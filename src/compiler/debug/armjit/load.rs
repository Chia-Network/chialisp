use std::collections::BTreeMap;
use elf_rs::*;

use crate::compiler::debug::armjit::memory::{NEG, TargetMemory};

const PC13_MASK: u32 = (1 << 13) - 1;

#[derive(Debug, Clone)]
struct ElfSym {
    st_name: u32,
    st_value: u32,
    st_size: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
}

#[derive(Debug, Clone)]
struct ElfRela {
    offset: u32,
    info: u32,
    addend: i32,
}

enum ElfRelaType {
    R_ARM_ABS32,
    R_ARM_LDR_PC_G0,
}

impl ElfRela {
    fn sym(&self) -> usize {
        (self.info >> 8) as usize
    }
    fn kind(&self) -> Option<ElfRelaType> {
        match self.info & 0xff {
            2 => Some(ElfRelaType::R_ARM_ABS32),
            4 => Some(ElfRelaType::R_ARM_LDR_PC_G0),
            _ => None
        }
    }
}

#[derive(Debug, Clone)]
struct ElfRelaSection {
    target: usize,
    content: Vec<ElfRela>
}

#[derive(Debug, Clone)]
struct ElfRelocations {
    pub rela: BTreeMap<usize, ElfRelaSection>
}

fn read_u16(content: &[u8], offset: usize) -> u16 {
    (content[offset] as u16) | ((content[offset + 1] as u16) << 8)
}

fn read_u24(content: &[u8], offset: usize) -> u32 {
    (content[offset] as u32) | ((content[offset + 1] as u32) << 8) | ((content[offset + 2] as u32) << 16)
}

fn read_u32(content: &[u8], offset: usize) -> u32 {
    read_u24(content, offset) | ((content[offset + 3] as u32) << 24)
}

fn read_i32(content: &[u8], offset: usize) -> i32 {
    let first_24 = read_u24(content, offset) as i32;
    let msb = content[offset + 3] as i32;
    if (msb & 0x80) != 0 {
        NEG + (first_24 | ((msb & 0x7f) << 24))
    } else {
        first_24 | (msb << 24)
    }
}

fn write_u32(content: &mut [u8], offset: usize, value: u32) {
    content[offset] = (value & 0xff) as u8;
    content[offset + 1] = ((value >> 8) & 0xff) as u8;
    content[offset + 2] = ((value >> 16) & 0xff) as u8;
    content[offset + 3] = ((value >> 24) & 0xff) as u8;
}

fn write_i32(content: &mut [u8], offset: usize, value: i32) {
    content[offset] = (value & 0xff) as u8;
    content[offset + 1] = ((value >> 8) & 0xff) as u8;
    content[offset + 2] = ((value >> 16) & 0xff) as u8;
    let stripped_msb = ((value >> 24) & 0x7f) as u8;
    let msb =
        if value < 0 {
            stripped_msb | 0x80
        } else {
            stripped_msb
        };
    content[offset + 3] = msb;
}

fn read_rela(content: &[u8], offset: usize) -> ElfRela {
    ElfRela {
        offset: read_u32(content, offset),
        info: read_u32(content, offset + 4),
        addend: read_i32(content, offset + 8)
    }
}

fn read_reloc_content(content: &[u8], entry_size: usize) -> Vec<ElfRela> {
    let mut result = Vec::new();
    for i in 0..(content.len() / entry_size) {
        result.push(read_rela(content, i * entry_size));
    }

    result
}

fn read_sym(content: &[u8], offset: usize) -> ElfSym {
    ElfSym {
        st_name: read_u32(content, offset),
        st_value: read_u32(content, offset + 4),
        st_size: read_u32(content, offset + 8),
        st_info: content[offset + 12],
        st_other: content[offset + 13],
        st_shndx: read_u16(content, offset + 14)
    }
}

fn read_sym_content(content: &[u8], entry_size: usize) -> Vec<ElfSym> {
    let mut result = Vec::new();
    for i in 0..(content.len() / entry_size) {
        result.push(read_sym(content, i * entry_size));
    }

    result
}

pub struct ElfLoader<'a> {
    elf: Elf<'a>,
}

impl<'a> ElfLoader<'a> {
    pub fn new(elf_bytes: &'a [u8]) -> Result<Self, Error> {
        Ok(ElfLoader {
            elf: Elf::from_bytes(&elf_bytes)?
        })
    }

    fn apply_reloc<M>(
        &self,
        memory: &mut M,
        target_addr: u32,
        sections: &[u32],
        symbols: &[ElfSym],
        in_section: usize,
        r: &ElfRela
    ) where M: TargetMemory {
        let reloc_at_addr = (sections[in_section] as u32) + r.offset;
        eprintln!("R {in_section} {reloc_at_addr:08x} reloc {r:?}");

        match r.kind() {
            Some(ElfRelaType::R_ARM_ABS32) => {
                // R_ARM_ABS32 = S + A
                let val_S = symbols[r.sym()].st_value as i32;
                let val_A = r.addend;
                eprintln!("S {val_S} A {val_A}");
                let final_value =
                    if val_A < 0 {
                        val_S - -val_A
                    } else {
                        val_S + val_A
                    };
                memory.write_i32(reloc_at_addr, final_value);
            }
            Some(ElfRelaType::R_ARM_LDR_PC_G0) => {
                // AKA R_ARM_PC13 = S - P + A
                let val_S = symbols[r.sym()].st_value as i32;
                let val_P = r.offset as i32;
                let val_A = r.addend;
                if r.addend >= -128 && r.addend < 127 {
                    // In range for simple encoding.
                    let mut existing = memory.read_u32(reloc_at_addr);
                    let mut low_8 = ((val_S - val_P + val_A) & 0xff) as u8;
                    let replace = (existing & !PC13_MASK) | low_8 as u32;
                    eprintln!("{:08x}: {:08x} => {:08x}", reloc_at_addr, existing, replace);
                    memory.write_u32(reloc_at_addr, replace);
                } else {
                    // Sparse encoding.
                    todo!();
                }
            }
            _ => todo!()
        }
    }

    pub fn load<M>(&self, memory: &mut M, target_addr: u32) where M: TargetMemory {
        let mut sections = Vec::new();
        let mut relocs = ElfRelocations {
            rela: BTreeMap::default()
        };
        let mut symbols = Vec::new();
        let mut section_addr = target_addr;

        // Collect relocation sections and set loaded data.
        for (i, s) in self.elf.section_header_iter().enumerate() {
            if s.flags().contains(SectionHeaderFlags::SHF_ALLOC) {
                if let Some(content) = s.content() {
                    memory.write_data(content, section_addr);
                }
                sections.push(section_addr);
                eprintln!("{i} {s:?}");
                eprintln!("load section {} at {section_addr:08x}", i);
                section_addr += s.size() as u32;
            } else {
                sections.push(0);
            }

            if matches!(s.sh_type(), SectionType::SHT_RELA) {
                if let Some(content) = s.content() {
                    let content = read_reloc_content(content, s.entsize() as usize);
                    let target_usize = s.info() as usize;
                    relocs.rela.insert(target_usize, ElfRelaSection {
                        content,
                        target: target_usize
                    });
                }
            } else if matches!(s.sh_type(), SectionType::SHT_SYMTAB) {
                if let Some(content) = s.content() {
                    if !symbols.is_empty() {
                        todo!();
                    }
                    symbols = read_sym_content(content, s.entsize() as usize);
                }
            }
        }

        for (rs, rd) in relocs.rela.iter() {
            for r in rd.content.iter() {
                self.apply_reloc(memory, target_addr, &sections, &symbols, *rs, r);
            }
        }
    }
}
