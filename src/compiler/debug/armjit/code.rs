use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::swap;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;

use faerie::{ArtifactBuilder, Decl, Link, SectionKind};
use gimli;
use gimli::{Encoding, Format, LineEncoding};
use gimli::constants::{DW_AT_low_pc, DW_AT_high_pc, DW_AT_name};
use gimli::write::{Address, Attribute, AttributeValue, DirectoryId, Dwarf, FileId, LineProgram, LineString, Location, LocationList, Range, RangeList, Section, Sections, Unit, UnitId};
use target_lexicon::triple;
use tempfile::NamedTempFile;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm::casts::bigint_to_bytes_clvm;
use crate::compiler::clvm::{sha256tree, truthy};
use crate::compiler::debug::armjit::load::{ElfLoader, write_u32};
use crate::compiler::sexp::{Atom, NodeSel, SelectNode, SExp, ThisNode};
use crate::compiler::srcloc::Srcloc;

const ENV_PTR: i32 = 0;
const NEXT_ALLOC_OFFSET: i32 = 4;

const SWI_DONE: usize = 0;
const SWI_THROW: usize = 1;
const SWI_DISPATCH_NEW_CODE: usize = 2;
const SWI_DISPATCH_INSTRUCTION: usize = 3;

pub const TARGET_ADDR: u32 = 0x1000;

//
// Compile each program to clvm, then decompose into arm assembly.
// If it isn't a proper program, just translate it as clvm.
//
// Initially do this with binutils then later with internally linked libraries.
//
// It's easy enough to run:
//
// - main.s -
// .align 4
//
//     .globl test
//     .globl _start
//
//     _start:
//     add   sp, sp, #0x7000
//     push	{fp, lr}
//     add	fp, sp, #4
//     sub	sp, sp, #8
//     mov	r0, #5
//     bl	test
//     str	r0, [fp, #-8]
//     ldr	r3, [fp, #-8]
// #   mov	r0, r3
//     sub	sp, fp, #4
//     pop	{fp, lr}
//     bx	lr
//
// linked with
//
// - test.c -
// int test(int x) {
//     return x + 3;
// }
//
// in an arm system emulator.  we can link one in rust via armv4t_emu or run with
//
// qemu-system-arm -S -s -machine virt -device loader,addr=0x8000,file=./test-prog,cpu-num=0
//
// after building like
// arm-none-eabi-as -o main.o main.s
// arm-none-eabi-gcc -c test.c
// arm-none-eabi-gcc -Ttext=0x8000 -static -nostdlib -o test-prog main.o test.o
//
// We can execute _start at 0x8000 and have the stack at 0x7000 in this scenario.
//
// CLVM operators will take a 'self' object pointer in r0, which will contain a
// pointer to the environment stack at the 0th offset, a pointer to the function
// table and pointers to other utilities, including a function which translates
// and runs untranslated CLVM, along with sending an event to gdb to load any
// symbols it's able to generate if possible.
//
// We'll have a table that contains the functions that were recovered from the
// provided CLVM and chialisp inputs.  Given the treehash of some upcoming code
// in the translation of an 'a' operation, we'll check whether we've cached the
// treehash by address (because each clvm value is written only once), then
// treehash and cache the translation, then jump to either the function which
// comes from the matching treehash or the emulator function.
//
// The entrypoint of the code contains a pointer to the actual environment, which
// is copied into the heap and then the main object is constructed around it.

// Aranges:
// I'm unsure if gimli can write this by itself.
// For now I'm going to generate it myself.
//
// Format:
// Header --
// WORD remaining section size
// HALF dwarf version
// WORD .debug_info offset
// BYTE bytes per address (n)
// Then a sequence of tuples after padding to n bytes --
// n-addr Start Address
// n-uint Length

#[derive(Clone, Debug)]
enum Register {
    SP,
    PC,
    FP,
    LR,
    R(usize),
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Register::SP => write!(f, "sp"),
            Register::PC => write!(f, "pc"),
            Register::FP => write!(f, "fp"),
            Register::LR => write!(f, "lr"),
            Register::R(n) => write!(f, "r{}", n),
        }
    }
}

trait ToU32 {
    fn to_u32(&self) -> u32;
}

impl ToU32 for Register {
    fn to_u32(&self) -> u32 {
        match self {
            Register::R(n) => *n as u32,
            Register::FP => 11,
            Register::SP => 13,
            Register::LR => 14,
            Register::PC => 15
        }
    }
}

#[derive(Clone, Debug)]
enum Instr {
    Align4,
    Section(String),
    Globl(String),
    Label(String),
    Space(usize,u8),
    Add(Register,Register,Register),
    Addi(Register,Register,i32),
    Sub(Register,Register,Register),
    Subi(Register,Register,i32),
    Andi(Register,Register,i32),
    Push(Vec<Register>),
    Pop(Vec<Register>),
    Mov(Register,i32),
    Str(Register,Register,i32),
    Ldr(Register,Register,i32),
    B(String),
    BEq(String),
    Bl(String),
    Bx(Register),
    Lea(Register,String),
    Swi(usize),
    SwiEq(usize),
    Cmpi(Register,usize),
    Long(usize),
    Addr(String),
    Bytes(Vec<u8>),
}

impl Instr {
    fn size(&self, current: usize) -> usize {
        match self {
            Instr::Align4 => {
                let next = (current + 3) & !3;
                next - current
            }
            Instr::Section(_) => 0,
            Instr::Space(size,_fill) => *size,
            Instr::Globl(_l) => 0,
            Instr::Label(_l) => 0,
            Instr::Bytes(v) => v.len(),
            _ => 4
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BeginEndBlock {
    BeginBlock,
    EndBlock
}

trait Encodable {
    fn encode(&self, v: &mut Vec<u8>, r: &mut Vec<Relocation>, function: &str);
}

enum ArmCond {
    Unconditional,
    Equal,
}

impl ToU32 for ArmCond {
    fn to_u32(&self) -> u32 {
        match self {
            ArmCond::Unconditional => 14 << 28,
            ArmCond::Equal => 0,
        }
    }
}

fn vec_from_u32(v: &mut Vec<u8>, data: u32) {
    v.push((data & 0xff) as u8);
    v.push(((data >> 8) & 0xff) as u8);
    v.push(((data >> 16) & 0xff) as u8);
    v.push((data >> 24) as u8);
}

enum ArmDataOp {
    Add,
    And,
    Sub,
    Mov,
    Cmp,
}

impl ToU32 for ArmDataOp {
    fn to_u32(&self) -> u32 {
        match self {
            ArmDataOp::Add => 4 << 21,
            ArmDataOp::And => 0,
            ArmDataOp::Sub => 2 << 21,
            ArmDataOp::Cmp => 10 << 21,
            ArmDataOp::Mov => 13 << 21,
        }
    }
}

enum ArmOp {
    Swi
}

impl ToU32 for ArmOp {
    fn to_u32(&self) -> u32 {
        match self {
            ArmOp::Swi => 15 << 24
        }
    }
}

struct Rn(Register);

impl ToU32 for Rn {
    fn to_u32(&self) -> u32 {
        self.0.to_u32() << 16
    }
}

struct Rd(Register);

impl ToU32 for Rd {
    fn to_u32(&self) -> u32 {
        self.0.to_u32() << 12
    }
}

struct Rm(u32, Register);

impl ToU32 for Rm {
    fn to_u32(&self) -> u32 {
        self.0 << 4 | self.1.to_u32()
    }
}

enum RelocationKind {
    Long,
    Branch,
}

struct Relocation {
    kind: RelocationKind,
    function: String,
    code_location: usize,
    reloc_target: String,
}

impl ToU32 for Vec<Register> {
    fn to_u32(&self) -> u32 {
        let mut out: u32 = 0;
        for r in self.iter() {
            out |= 1 << r.to_u32();
        }
        out
    }
}

impl Encodable for Instr {
    fn encode<'a>(&self, v: &mut Vec<u8>, r: &mut Vec<Relocation>, function: &str) {
        match self {
            Instr::Align4 => {
                while v.len() % 4 != 0 {
                    v.push(0);
                }
            }
            Instr::Space(n,val) => {
                for _ in 0..*n {
                    v.push(*val)
                }
            }
            Instr::Bytes(vs) => {
                v.extend(vs.clone());
            }
            Instr::Long(l) => {
                vec_from_u32(v, *l as u32);
            }
            Instr::Addr(target) => {
                r.push(Relocation {
                    kind: RelocationKind::Long,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                vec_from_u32(v, 0);
            }
            Instr::Add(r_d,r_s,r_a) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::Add.to_u32() | Rn(r_s.clone()).to_u32() | Rd(r_d.clone()).to_u32() | Rm(0,r_a.clone()).to_u32()),
            Instr::Addi(r_d,r_s,imm) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::Add.to_u32() | Rn(r_s.clone()).to_u32() | Rd(r_d.clone()).to_u32() | (1 << 25) | (*imm as u32)),
            Instr::Sub(r_d,r_s,r_a) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::Sub.to_u32() | Rn(r_s.clone()).to_u32() | Rd(r_d.clone()).to_u32() | Rm(0,r_a.clone()).to_u32()),
            Instr::Subi(r_d,r_s,imm) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::Sub.to_u32() | Rn(r_s.clone()).to_u32() | Rd(r_d.clone()).to_u32() | (1 << 25) | (*imm as u32)),
            Instr::Andi(r_d,r_s,imm) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::And.to_u32() | Rn(r_s.clone()).to_u32() | Rd(r_d.clone()).to_u32() | (1 << 25) | (*imm as u32)),
            Instr::Mov(r_d,imm) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::Mov.to_u32() | Rd(r_d.clone()).to_u32() | (1 << 25) | (*imm as u32)),
            Instr::Push(rs) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | 4 << 25 | Rn(Register::SP).to_u32() | 1 << 21 | rs.to_u32()),
            Instr::Pop(rs) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | 4 << 25 | Rn(Register::SP).to_u32() | 1 << 20 | 1 << 21 | 1 << 23 | 1 << 24 | rs.to_u32()),
            Instr::Str(rd,rs,off) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | 1 << 26 | 1 << 24 | 1 << 23 | Rn(rs.clone()).to_u32() | Rd(rd.clone()).to_u32() | (((65536 + off) & 0xfff) as u32)),
            Instr::Ldr(rd,rs,off) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | 1 << 26 | 1 << 24 | 1 << 23 | 1 << 20 | Rn(rs.clone()).to_u32() | Rd(rd.clone()).to_u32() | (((65536 + off) & 0xfff) as u32)),
            Instr::B(target) => {
                r.push(Relocation {
                    kind: RelocationKind::Branch,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 5 << 25);
            }
            Instr::BEq(target) => {
                r.push(Relocation {
                    kind: RelocationKind::Branch,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                vec_from_u32(v, ArmCond::Equal.to_u32() | 5 << 25);
            }
            Instr::Bl(target) => {
                r.push(Relocation {
                    kind: RelocationKind::Branch,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 5 << 25 | 1 << 24);
            }
            Instr::Bx(r) => {
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 0x12fff10 | r.to_u32());
            }
            Instr::Lea(rd,target) => {
                // Emit a load from +8 (0 as encoded).
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 1 << 26 | 1 << 24 | 1 << 23 | 1 << 20 | Rn(Register::PC).to_u32() | Rd(rd.clone()).to_u32());
                // Emit a jump to +8
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 5 << 25 | 0);
                r.push(Relocation {
                    kind: RelocationKind::Long,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone()
                });
                // Relocatable space.
                vec_from_u32(v, 0);
            }
            Instr::Swi(n) => {
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmOp::Swi.to_u32() | (*n as u32));
            }
            Instr::SwiEq(n) => {
                vec_from_u32(v, ArmCond::Equal.to_u32() | ArmOp::Swi.to_u32() | (*n as u32));
            }
            Instr::Cmpi(r,n) => {
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | ArmDataOp::Cmp.to_u32() | 1 << 25 | 1 << 20 | Rn(r.clone()).to_u32() | (*n as u32));
            }
            _ => {}
        }
    }
}

#[test]
fn test_arm_encoding_add_1_3_7() {
    let mut v = Vec::new();
    let mut r = Vec::new();
    Instr::Add(Register::R(1), Register::R(3), Register::R(7)).encode(&mut v, &mut r, "test");
    assert_eq!(b"\x07\x10\x83\xe0".to_vec(), v);
}

#[test]
fn test_arm_encoding_swi() {
    let mut v = Vec::new();
    let mut r = Vec::new();
    Instr::Swi(13).encode(&mut v, &mut r, "test");
    assert_eq!(b"\x0d\x00\x00\xef".to_vec(), v);
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Instr::Align4 => write!(f, "  .align 4"),
            Instr::Section(s) => write!(f, "  .section {s}"),
            Instr::Space(size,fill) => write!(f, "  .space {size},{fill}"),
            Instr::Globl(l) => write!(f, "  .globl {l}"),
            Instr::Label(l) => write!(f, "{l}:"),
            Instr::Add(r_d,r_s,r_a) => write!(f, "  add {r_d}, {r_s}, {r_a}"),
            Instr::Addi(r_d,r_s,imm) => write!(f, "  add {r_d}, {r_s}, #{imm}"),
            Instr::Andi(r_d,r_s,imm) => write!(f, "  and {r_d}, {r_s}, #{imm}"),
            Instr::Sub(r_d,r_s,r_a) => write!(f, "  sub {r_d}, {r_s}, {r_a}"),
            Instr::Subi(r_d,r_s,imm) => write!(f, "  sub {r_d}, {r_s}, #{imm}"),
            Instr::Cmpi(r,imm) => write!(f, "  cmp {r}, #{imm}"),
            Instr::Push(rs) => {
                write!(f, "  push {{")?;
                let mut sep = "";
                for r in rs.iter() {
                    write!(f, "{sep}{r}")?;
                    sep = ", ";
                }
                write!(f, "}}")
            },
            Instr::Pop(rs) => {
                write!(f, "  pop {{")?;
                let mut sep = "";
                for r in rs.iter() {
                    write!(f, "{sep}{r}")?;
                    sep = ", ";
                }
                write!(f, "}}")
            }
            Instr::Mov(r_d,imm) => write!(f, "  mov {r_d}, #{imm}"),
            Instr::Str(r_s,r_a,imm) => write!(f, "  str {r_s}, [{r_a}, #{imm}]"),
            Instr::Ldr(r_d,r_a,imm) => write!(f, "  ldr {r_d}, [{r_a}, #{imm}]"),
            Instr::B(l) => write!(f, "  b {l}"),
            Instr::Bl(l) => write!(f, "  bl {l}"),
            Instr::BEq(l) => write!(f, "  beq {l}"),
            Instr::Bx(r) => write!(f, "  bx {r}"),
            Instr::Lea(r,l) => write!(f, "  ldr {r}, ={l}"),
            Instr::Swi(n) => write!(f, "  swi {n}"),
            Instr::SwiEq(n) => write!(f, "  swieq {n}"),
            Instr::Long(n) => write!(f, "  .long {n}"),
            Instr::Addr(lbl) => write!(f, "  .long {lbl}"),
            Instr::Bytes(v) => {
                let mut sep = " ";
                write!(f, "  .byte")?;
                for b in v.iter() {
                    write!(f, "{sep}{b}")?;
                    sep = ", ";
                }
                Ok(())
            }
        }
    }
}

struct DwarfBuilder {
    unit_id: UnitId,
    file_to_id: HashMap<Vec<u8>, (DirectoryId, FileId)>,
    directory_to_id: HashMap<Vec<u8>, DirectoryId>,

    seq_addr_start: usize,

    dwarf: Dwarf
}

#[derive(Default, Clone, Debug)]
struct DwarfSectionWriter {
    pub written: Vec<u8>
}

impl gimli::write::Writer for DwarfSectionWriter {
    type Endian = gimli::LittleEndian;

    fn endian(&self) -> Self::Endian {
        return gimli::LittleEndian::default();
    }

    fn len(&self) -> usize {
        self.written.len()
    }

    fn write(&mut self, bytes: &[u8]) -> gimli::write::Result<()> {
        for b in bytes.iter() {
            self.written.push(*b);
        }

        Ok(())
    }

    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> gimli::write::Result<()> {
        let mut to_skip = 0;
        if offset < self.written.len() {
            to_skip = self.written.len() - offset;
            for (i, b) in bytes.iter().enumerate().take(to_skip) {
                self.written[offset + i] = *b;
            }
        }

        while offset > self.written.len() {
            self.written.push(0);
        }

        for b in bytes.iter().skip(to_skip) {
            self.written.push(*b);
        }

        Ok(())
    }
}

impl DwarfBuilder {
    fn new(filename: &str) -> Self {
        let mut path = PathBuf::new();
        path.push(filename);
        path.pop();
        let mut dirname = path.into_os_string().to_string_lossy().as_bytes().to_vec();

        if dirname.is_empty() {
            dirname = b".".to_vec();
        }

        path = PathBuf::new();
        path.push(filename);
        let filename = path.file_name().map(|f| f.to_string_lossy().as_bytes().to_vec()).unwrap_or_else(|| filename.as_bytes().to_vec());

        let line_encoding = LineEncoding {
            minimum_instruction_length: 4,
            maximum_operations_per_instruction: 1,
            default_is_stmt: false,
            line_base: 0,
            line_range: 1
        };

        let mut dwarf = Dwarf::default();
        let encoding = Encoding {
            address_size: 4,
            format: Format::Dwarf32,
            version: 2
        };

        let dirstring = LineString::String(dirname.clone());
        let filestring = LineString::String(filename.clone());
        let mut line_program = LineProgram::new(
            encoding.clone(),
            line_encoding,
            dirstring.clone(),
            filestring.clone(),
            None
        );
        let mut directory_to_id = HashMap::new();
        let directory_id = line_program.add_directory(dirstring);
        directory_to_id.insert(dirname.clone(), directory_id.clone());
        let mut file_to_id = HashMap::new();
        let file_id = line_program.add_file(
            filestring,
            directory_id.clone(),
            None
        );
        file_to_id.insert(filename.clone(), (directory_id, file_id));

        let mut unit = Unit::new(encoding, line_program);

        unit.ranges.add(RangeList(vec![
            Range::BaseAddress { address: Address::Constant(TARGET_ADDR as u64) }
        ]));
        unit.locations.add(LocationList(vec![
            Location::BaseAddress { address: Address::Constant(TARGET_ADDR as u64) }
        ]));
        let unit_ent = unit.get_mut(unit.root());
        unit_ent.set(DW_AT_low_pc, AttributeValue::Address(Address::Constant(TARGET_ADDR as u64)));
        // XXX Compute real upper bound.
        unit_ent.set(DW_AT_high_pc, AttributeValue::Address(Address::Constant(TARGET_ADDR as u64 + 0x100000)));
        unit_ent.set(DW_AT_name, AttributeValue::String(filename.clone()));

        let unit_id = dwarf.units.add(unit);

        let obj = DwarfBuilder {
            seq_addr_start: 0,
            unit_id,
            file_to_id,
            directory_to_id,
            dwarf,
        };

        obj
    }

    fn add_file_having_dirid(&mut self, dirid: DirectoryId, filename: &[u8]) -> (DirectoryId, FileId) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        let filestring = LineString::String(filename.to_vec());
        let fileid = unit.line_program.add_file(
            filestring,
            dirid.clone(),
            None
        );
        self.file_to_id.insert(filename.to_vec(), (dirid.clone(), fileid.clone()));
        (dirid, fileid)
    }

    fn add_file(&mut self, filename_str: &str) -> (DirectoryId, FileId) {
        let mut path = PathBuf::new();
        path.push(filename_str);
        let filename = path.file_name().map(|f| f.to_string_lossy().as_bytes().to_vec()).unwrap_or_else(|| filename_str.as_bytes().to_vec());
        if let Some((dirid, fileid)) = self.file_to_id.get(&filename) {
            return (*dirid, *fileid);
        }

        path = PathBuf::new();
        path.push(filename_str);
        path.pop();

        let dirname = path.into_os_string().to_string_lossy().as_bytes().to_vec();
        let use_dirname =
            if dirname.is_empty() {
                vec![b'.']
            } else {
                dirname.clone()
            };

        if let Some(dirid) = self.directory_to_id.get(&use_dirname) {
            return self.add_file_having_dirid(*dirid, &filename);
        }

        let dirstring = LineString::String(use_dirname.clone());
        let unit = self.dwarf.units.get_mut(self.unit_id);
        let dirid = unit.line_program.add_directory(dirstring);
        self.directory_to_id.insert(use_dirname, dirid.clone());
        self.add_file_having_dirid(dirid, &filename)
    }

    fn add_instr(&mut self, addr: usize, loc: &Srcloc, _instr: Instr, begin_end_block: Option<BeginEndBlock>) {
        let (_, file_id) = self.add_file(&loc.file);
        let unit = self.dwarf.units.get_mut(self.unit_id);
        if !unit.line_program.in_sequence() {
            return;
        }
        let row = unit.line_program.row();
        row.address_offset = (addr - self.seq_addr_start) as u64;
        row.file = file_id;
        row.line = loc.line as u64;
        row.column = loc.col as u64;
        row.is_statement = addr == self.seq_addr_start;
        eprintln!("line row {} at {:?}", row.address_offset, loc);
        row.basic_block = begin_end_block == Some(BeginEndBlock::BeginBlock);
        unit.line_program.generate_row();
    }

    fn start(&mut self, addr: usize) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        self.seq_addr_start = addr;
        unit.line_program.begin_sequence(Some(Address::Constant((addr + TARGET_ADDR as usize) as u64)));
    }

    fn end(&mut self, addr: usize) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        unit.line_program.end_sequence((addr - self.seq_addr_start) as u64);
    }

    fn write_section(&self, name: &str, section: &dyn Section<DwarfSectionWriter>, instrs: &mut Vec<Instr>) {
        instrs.push(Instr::Align4);
        instrs.push(Instr::Section(name.to_string()));
        instrs.push(Instr::Bytes(section.written.clone()));
    }

    fn write(&mut self, instrs: &mut Vec<Instr>) -> gimli::write::Result<()> {
        let mut sections = Sections::<DwarfSectionWriter>::default();
        self.dwarf.write(&mut sections)?;

        self.write_section(".debug_abbrev", &sections.debug_abbrev, instrs);
        self.write_section(".debug_info", &sections.debug_info, instrs);
        self.write_section(".debug_line", &sections.debug_line, instrs);
        self.write_section(".debug_line_str", &sections.debug_line_str, instrs);
        self.write_section(".debug_ranges", &sections.debug_ranges, instrs);
        self.write_section(".debug_rnglists", &sections.debug_rnglists, instrs);
        self.write_section(".debug_loc", &sections.debug_loc, instrs);
        self.write_section(".debug_loclists", &sections.debug_loclists, instrs);
        self.write_section(".debug_str", &sections.debug_str, instrs);
        self.write_section(".debug_frame", &sections.debug_frame, instrs);
        self.write_section(".eh_frame", &sections.eh_frame, instrs);

        return Ok(());
    }
}

enum Constant {
    Atom(String, Vec<u8>),
    Cons(String, String, String),
}

impl Constant {
    fn label(&self) -> String {
        match self {
            Constant::Atom(lbl,_) => lbl.clone(),
            Constant::Cons(lbl,_,_) => lbl.clone(),
        }
    }
}

pub struct Program {
    finished_insns: Vec<Instr>,
    first_label: String,
    env_label: String,
    encounters_of_code: HashMap<Vec<u8>, usize>,
    labels_by_hash: HashMap<Vec<u8>, String>,
    done_programs: HashSet<String>,
    waiting_programs: Vec<(String, Rc<SExp>)>,
    constants: HashMap<Vec<u8>, Constant>,
    symbol_table: HashMap<String, String>,
    current_addr: usize,
    dwarf_builder: DwarfBuilder,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let write_vec = |f: &mut Formatter, v: &[Instr]| -> fmt::Result {
            for i in v.iter() {
                write!(f, "{i}\n")?;
            }
            Ok(())
        };

        write_vec(f, &self.finished_insns)
    }
}

fn hexify(v: &[u8]) -> String{
    Bytes::new(Some(BytesFromType::Raw(v.to_vec()))).hex()
}

fn is_atom(a: Rc<SExp>) -> Option<(Srcloc, Vec<u8>)> {
    match a.borrow() {
        SExp::Cons(_,_,_) => None,
        SExp::Nil(l) => Some((l.clone(), Vec::new())),
        SExp::Atom(l,a) => Some((l.clone(), a.clone())),
        SExp::QuotedString(l,_,a) => Some((l.clone(), a.clone())),
        SExp::Integer(l,i) => {
            let bytes = bigint_to_bytes_clvm(&i);
            Some((l.clone(), bytes.data().clone()))
        }
    }
}

fn is_wrapped_atom(a: Rc<SExp>) -> Option<(Srcloc, Vec<u8>)> {
    if let Ok(NodeSel::Cons((l, a), n)) =
        NodeSel::Cons(Atom::Here(()), ThisNode).select_nodes(a)
    {
        if truthy(n) {
            return None;
        }

        return Some((l, a));
    }

    None
}

fn dequote(a: Rc<SExp>) -> Option<Rc<SExp>> {
    if let Ok(NodeSel::Cons(_q, v)) =
        NodeSel::Cons(Atom::Here("\x01"), ThisNode).select_nodes(a)
    {
        return Some(v.clone());
    }

    None
}

impl Program {
    fn get_code_label(&mut self, hash: &[u8]) -> String {
        let n =
            if let Some(n) = self.encounters_of_code.get(hash).clone() {
                *n
            } else {
                0
            };

        self.encounters_of_code.insert(hash.to_vec(), n+1);
        return format!("_{}_{n}", hexify(hash));
    }

    fn do_throw(&mut self, loc: &Srcloc, hash: &[u8]) {
        self.load_atom(loc, hash, hash);
        self.push(loc, Instr::Swi(SWI_THROW));
    }

    fn add_sexp(&mut self, loc: &Srcloc, hash: &[u8], s: Rc<SExp>) -> String {
        if let Some(lbl) = self.constants.get(hash) {
            return lbl.label();
        }

        match s.borrow() {
            SExp::Cons(_l,a,b) => {
                let a_hash = sha256tree(a.clone());
                let b_hash = sha256tree(b.clone());
                let a_label = self.add_sexp(loc, &a_hash, a.clone());
                let b_label = self.add_sexp(loc, &b_hash, b.clone());
                let label = format!("_{}", hexify(hash));
                self.constants.insert(hash.to_vec(), Constant::Cons(label.clone(), a_label.clone(), b_label.clone()));
                label

            },
            SExp::Nil(_) => self.add_atom(hash, &[]),
            SExp::Atom(_, a) => self.add_atom(hash, &a),
            SExp::QuotedString(_, _, a) => self.add_atom(hash, &a),
            SExp::Integer(_, i) => {
                let v = bigint_to_bytes_clvm(&i).data().clone();
                self.add_atom(hash, &v)
            }
        }
    }

    fn load_sexp(&mut self, loc: &Srcloc, hash: &[u8], s: Rc<SExp>) {
        let label = self.add_sexp(loc, hash, s);
        self.push(loc, Instr::Lea(Register::R(0), label));
    }

    fn first_rest(&mut self, loc: &Srcloc, hash: &[u8], lst: &[SExp], offset: i32) {
        if lst.len() != 1 {
            return self.do_throw(loc, hash);
        }

        let subexp = self.add(Rc::new(lst[0].clone()));
        self.push(
            loc,
            Instr::Addi(Register::R(0), Register::R(5), 0)
        );
        self.push(
            loc,
            Instr::Bl(subexp)
        );
        // Determine if the result is a cons.
        self.push(
            loc,
            Instr::Andi(Register::R(1), Register::R(0), 1)
        );
        self.push(
            loc,
            Instr::Cmpi(Register::R(1), 1)
        );
        self.push(
            loc,
            Instr::SwiEq(SWI_THROW)
        );
        self.push(
            loc,
            Instr::Ldr(Register::R(0), Register::R(0), offset)
        );
    }

    fn do_operator(&mut self, loc: &Srcloc, hash: &[u8], a: &[u8], b: Rc<SExp>, treat_as_quoted: bool) {
        if treat_as_quoted {
            todo!();
        }

        if a == b"" {
            return self.do_throw(loc, hash);
        }

        // Quote is special.
        if a == &[1] {
            self.add(b.clone());
            return self.load_sexp(loc, hash, b);
        }

        // Every other operator must have a proper list following it.
        let lst =
            if let Some(lst) = b.proper_list() {
                lst
            } else {
                return self.do_throw(loc, hash);
            };

        if a == &[2] {
            // Apply operator
            if lst.len() != 2 {
                return self.do_throw(loc, hash);
            }

            let env_comp = self.add(Rc::new(lst[1].clone()));

            self.push(
                loc,
                Instr::Bl(env_comp)
            );
            if let Some(quoted_code) = dequote(Rc::new(lst[0].clone())) {
                // Short circuit by reading out the quoted code and running it.
                let code_comp = self.add(quoted_code.clone());

                // Swap r0 (env) with r5[ENV_PTR]
                self.push(
                    loc,
                    Instr::Ldr(Register::R(4), Register::R(5), ENV_PTR),
                );
                self.push(
                    loc,
                    Instr::Str(Register::R(0), Register::R(5), ENV_PTR),
                );

                // r0 = env ptr
                self.push(
                    loc,
                    Instr::Addi(Register::R(0), Register::R(5), 0),
                );
                self.push(
                    loc,
                    Instr::Bl(code_comp)
                );

                // Reload the old env.
                self.push(
                    loc,
                    Instr::Str(Register::R(4), Register::R(5), ENV_PTR),
                );
                return;
            }

            let code_comp = self.add(Rc::new(lst[0].clone()));

            // Move env result to r4.
            self.push(
                loc,
                Instr::Addi(Register::R(4), Register::R(0), 0)
            );
            self.push(
                loc,
                Instr::Bl(code_comp)
            );

            // New code is in r0, new env in r4.  Swap r5[ENV_PTR] and r4.
            self.push(
                loc,
                Instr::Ldr(Register::R(1), Register::R(5), ENV_PTR),
            );
            self.push(
                loc,
                Instr::Str(Register::R(4), Register::R(5), ENV_PTR),
            );
            self.push(
                loc,
                Instr::Addi(Register::R(4), Register::R(1), 0),
            );

            // General case: don't know what code we have yet.
            //
            // Now r4 is the old env ptr (and will be preserved when executing
            // called code).  We'll restore it before leaving this apply.

            // r0 was code, move it to r6 (callee save).
            self.push(
                loc,
                Instr::Addi(Register::R(6), Register::R(0), 0),
            );
            // Load the env into arg0.
            self.push(
                loc,
                Instr::Addi(Register::R(0), Register::R(5), 0),
            );
            // Load the code into arg1.
            self.push(
                loc,
                Instr::Addi(Register::R(1), Register::R(6), 0),
            );
            // Dispatch the code.
            self.push(
                loc,
                Instr::Swi(SWI_DISPATCH_NEW_CODE)
            );
            self.push(
                loc,
                Instr::Ldr(Register::LR, Register::LR, 0),
            );
            self.push(
                loc,
                Instr::Bx(Register::LR)
            );

            // Reload the old env.
            self.push(
                loc,
                Instr::Str(Register::R(4), Register::R(5), ENV_PTR),
            );
            return;
        } else if a == &[3] {
            // If operator
            if lst.len() != 3 {
                return self.do_throw(loc, hash);
            }

            let else_clause = self.add(Rc::new(lst[2].clone()));
            let then_clause = self.add(Rc::new(lst[1].clone()));
            let cond_clause = self.add(Rc::new(lst[0].clone()));

            let else_label = self.get_code_label(hash);

            self.push(
                loc,
                Instr::Bl(cond_clause)
            );
            self.push(
                loc,
                Instr::Ldr(Register::R(1), Register::R(0), 0)
            );
            self.push(
                loc,
                Instr::Cmpi(Register::R(1), 1)
            );
            self.push(
                loc,
                Instr::BEq(else_label.clone())
            );
            self.push(
                loc,
                Instr::Bl(then_clause),
            );
            // Else clause acts as a function for relocation purposes.
            self.push(
                loc,
                Instr::Globl(else_label.clone())
            );
            self.push(
                loc,
                Instr::Label(else_label)
            );
            self.push(
                loc,
                Instr::B(else_clause),
            );
            return;
        } else if a == &[4] {
            // Cons operator
            if lst.len() != 2 {
                return self.do_throw(loc, hash);
            }

            let rest_label = self.add(Rc::new(lst[1].clone()));
            let first_label = self.add(Rc::new(lst[0].clone()));
            self.push(
                loc,
                Instr::Addi(Register::R(0), Register::R(5), 0),
            );
            self.push(
                loc,
                Instr::Bl(rest_label),
            );
            self.push(
                loc,
                Instr::Addi(Register::R(4), Register::R(0), 0),
            );
            self.push(
                loc,
                Instr::Addi(Register::R(0), Register::R(5), 0),
            );
            self.push(
                loc,
                Instr::Bl(first_label),
            );
            // R1 = next allocated address.
            self.push(
                loc,
                Instr::Ldr(Register::R(1), Register::R(5), NEXT_ALLOC_OFFSET)
            );
            // R2 = R1 + 8 (size of cons)
            self.push(
                loc,
                Instr::Addi(Register::R(2), Register::R(1), 8)
            );
            self.push(
                loc,
                Instr::Str(Register::R(2), Register::R(5), 0)
            );
            // Build cons
            self.push(
                loc,
                Instr::Str(Register::R(0), Register::R(1), 0)
            );
            self.push(
                loc,
                Instr::Str(Register::R(4), Register::R(1), 4)
            );
            // Move the result to r0
            self.push(
                loc,
                Instr::Addi(Register::R(0), Register::R(1), 0)
            );
            return;
        } else if a == &[5] {
            return self.first_rest(loc, hash, &lst, 0);
        } else if a == &[6] {
            return self.first_rest(loc, hash, &lst, 4);
        } else {
            // Dispatch instruction, push everything.
            for item in lst.iter().rev() {
                let clause_label = self.add(Rc::new(item.clone()));
                self.push(
                    loc,
                    Instr::Bl(clause_label)
                );
            }

            // Ensure we have this sexp loadable as data.
            let atom_hash = sha256tree(Rc::new(SExp::Atom(loc.clone(), a.to_vec())));
            let label = self.add_atom(&atom_hash, a);
            eprintln!("load {label} for general operator\n");
            self.push(
                loc,
                Instr::Lea(Register::R(1), label)
            );
            // Push number of objects on the stack.
            self.push(
                loc,
                Instr::Andi(Register::R(0), Register::R(0), 0)
            );
            self.push(
                loc,
                Instr::Addi(Register::R(0), Register::R(0), lst.len() as i32)
            );
            // Push an instruction dispatch.
            self.push(
                loc,
                Instr::Swi(SWI_DISPATCH_INSTRUCTION)
            );
        }
    }

    // R0 = the address of the env block.
    fn env_select(&mut self, loc: &Srcloc, hash: &[u8], v: &[u8]) {
        if v.is_empty() {
            self.load_atom(loc, hash, v);
            return;
        }

        // Let r0 be our pointer.
        self.push(
            loc,
            Instr::Ldr(Register::R(0), Register::R(5), ENV_PTR),
        );

        // Whole env ref.
        if v == &[1] {
            return;
        }

        for (i, byt) in v.iter().enumerate().rev() {
            for bit in 0..8 {
                let remaining = byt / (1 << bit);
                if remaining == 1 && i == 0 {
                    // We have the right value.
                    return;
                } else {
                    let offset = ((remaining & 1) * 4) as i32;

                    // Check for a cons.
                    self.push(
                        loc,
                        Instr::Andi(Register::R(1), Register::R(0), 1)
                    );
                    self.push(
                        loc,
                        Instr::Cmpi(Register::R(1), 1)
                    );
                    // Break if it was an atom.
                    self.push(
                        loc,
                        Instr::SwiEq(SWI_THROW)
                    );
                    // Load if it was a cons.
                    self.push(
                        loc,
                        Instr::Ldr(Register::R(0), Register::R(0), offset)
                    );
                }
            }
        }
    }

    fn add_atom(&mut self, hash: &[u8], v: &[u8]) -> String {
        if let Some(lbl) = self.constants.get(hash) {
            return lbl.label();
        }

        let label = format!("_{}", hexify(hash));
        self.constants.insert(hash.to_vec(), Constant::Atom(label.clone(), v.to_vec()));
        label
    }

    fn load_atom(&mut self, loc: &Srcloc, hash: &[u8], v: &[u8]) {
        let label = self.add_atom(hash, v);
        self.push(
            loc,
            Instr::Lea(Register::R(0), label)
        );
    }

    fn add(&mut self, sexp: Rc<SExp>) -> String {
        let hash = sha256tree(sexp.clone());

        // Note: get_code_label issues a fresh label for this hash every time.
        let body_label = self.get_code_label(&hash);
        self.waiting_programs.push((body_label.clone(), sexp.clone()));
        body_label
    }

    fn push_be(&mut self, srcloc: &Srcloc, instr: Instr, begin_end_block: Option<BeginEndBlock>) {
        let size = instr.size(self.current_addr);
        self.finished_insns.push(instr.clone());
        if size != 0 {
            let next_addr = self.current_addr + size;
            self.dwarf_builder.add_instr(self.current_addr, srcloc, instr, begin_end_block);
            self.current_addr = next_addr;
        }
    }

    fn push(&mut self, srcloc: &Srcloc, instr: Instr) {
        self.push_be(srcloc, instr, None);
    }

    fn emit_waiting(&mut self) {
        while let Some((label, sexp)) = self.waiting_programs.pop() {
            eprintln!("{} sexp {:?} {}", label, sexp.loc(), sexp);
            let hash = sha256tree(sexp.clone());

            self.labels_by_hash.insert(hash.clone(), label.clone());
            self.dwarf_builder.start(self.current_addr);

            self.push(
                &sexp.loc(),
                Instr::Globl(label.clone())
            );
            self.push(
                &sexp.loc(),
                Instr::Label(label.clone())
            );
            self.push_be(
                &sexp.loc(),
                Instr::Push(vec![Register::R(4), Register::R(5), Register::R(6), Register::LR]),
                Some(BeginEndBlock::BeginBlock)
            );

            // Grab the env pointer.
            self.push(
                &sexp.loc(),
                Instr::Addi(Register::R(5), Register::R(0), 0),
            );

            // Translate body.
            match sexp.borrow() {
                SExp::Cons(l,a,b) => {
                    if let Some((loc, a)) = is_atom(a.clone()) {
                        // do quoted operator
                        self.do_operator(&loc, &hash, &a, b.clone(), false);
                    } else if let Some((loc, a)) = is_wrapped_atom(a.clone()) {
                        // do unquoted operator
                        self.do_operator(&loc, &hash, &a, b.clone(), true);
                    } else {
                        // invalid head form, just throw.
                        self.do_throw(&l, &hash);
                    }
                },
                SExp::Nil(l) => {
                    self.load_atom(l, &hash, &[])
                }
                SExp::Atom(l, v) => {
                    if v.is_empty() {
                        return self.load_atom(l, &hash, &[]);
                    }
                    self.env_select(l, &hash, v);
                }
                SExp::QuotedString(l, _, v) => {
                    if v.is_empty() {
                        return self.load_atom(l, &hash, &[]);
                    }
                    self.env_select(l, &hash, v);
                }
                SExp::Integer(l, i) => {
                    let v = bigint_to_bytes_clvm(&i);
                    let v_ref = v.data();
                    if v_ref.is_empty() {
                        return self.load_atom(l, &hash, &[]);
                    }
                    self.env_select(l, &hash, v_ref);
                }
            }

            self.push(
                &sexp.loc(),
                Instr::Pop(vec![Register::R(4), Register::R(5), Register::R(6), Register::LR]),
            );
            self.push_be(
                &sexp.loc(),
                Instr::Bx(Register::LR),
                Some(BeginEndBlock::EndBlock)
            );
            self.dwarf_builder.end(self.current_addr);
        }
    }

    fn start_insns(&mut self) {
        let srcloc = Srcloc::start("*prolog*");
        for i in &[
            Instr::Section(".text".to_string()),
            Instr::Align4,
            Instr::Globl("_start".to_string()),
            Instr::Label("_start".to_string()),
            Instr::Lea(Register::R(0), "_run".to_string()),
            Instr::Bl(self.first_label.clone()),
            Instr::Swi(SWI_DONE),
            Instr::Label("_end".to_string()),
            Instr::B("_end".to_string())
        ] {
            self.push(&srcloc, i.clone());
        }
    }

    fn finish_insns(&mut self) -> Result<(), String> {
        let srcloc = Srcloc::start("*epilog*");
        for i in [
            Instr::Align4,
            Instr::Globl("_run".to_string()),
            Instr::Label("_run".to_string()),
            Instr::Addr(self.env_label.clone()),
            Instr::Long(0x10000000),
        ].iter() {
            self.push(&srcloc, i.clone());
        }

        // Write the constant data.
        let mut constants = HashMap::new();
        swap(&mut constants, &mut self.constants);
        for (_, c) in constants.iter() {
            if let Constant::Cons(label, a_label, b_label) = c {
                eprintln!("constant pair {label}");
                self.push(
                    &srcloc,
                    Instr::Align4
                );
                self.push(
                    &srcloc,
                    Instr::Globl(label.clone())
                );
                self.push(
                    &srcloc,
                    Instr::Label(label.clone())
                );
                self.push(
                    &srcloc,
                    Instr::Addr(a_label.clone())
                );
                self.push(
                    &srcloc,
                    Instr::Addr(b_label.clone())
                );
            }

        }

        self.push(
            &srcloc,
            Instr::Section(".data".to_string())
        );
        for (_, c) in constants.iter() {
            if let Constant::Atom(label, bytes) = c {
                self.push(
                    &srcloc,
                    Instr::Align4
                );
                self.push(
                    &srcloc,
                    Instr::Globl(label.clone())
                );
                self.push(
                    &srcloc,
                    Instr::Label(label.clone())
                );
                self.push(
                    &srcloc,
                    Instr::Long(bytes.len() * 2 + 1)
                );
                self.push(
                    &srcloc,
                    Instr::Bytes(bytes.clone())
                );
            }
        }
        swap(&mut constants, &mut self.constants);

        // Write the function table.
        self.push(
            &srcloc,
            Instr::Align4
        );
        self.push(
            &srcloc,
            Instr::Globl("_end".to_string())
        );
        self.push(
            &srcloc,
            Instr::Label("_end".to_string())
        );
        self.dwarf_builder.write(&mut self.finished_insns).unwrap();

        Ok(())
    }

    pub fn to_elf(&self, output: &str) -> Result<Vec<u8>, String> {
        let mut sections = Vec::new();
        let mut obj = ArtifactBuilder::new(triple!("arm-unknown-unknown-unknown-elf"))
            .name(output.to_owned())
            .finish();
        // Collect declarations
        let mut waiting_for_debug_info = None;
        let mut data_section = false;
        let mut data = "".to_string();
        let mut data_objects = Vec::new();

        let mut decls: Vec<(String, Decl)> = self.finished_insns.iter().filter_map(|i| {
            if let Instr::Section(name) = i {
                if name.starts_with(".debug") || name.starts_with(".eh") {
                    waiting_for_debug_info = Some(name.clone());
                    data_section = false;
                    return Some((name.to_string(), Decl::section(SectionKind::Debug).into()));
                } else if name == ".text" {
                    waiting_for_debug_info = None;
                    data_section = false;
                    // Predefined.
                    return None;
                } else {
                    eprintln!("data section {name}");
                    waiting_for_debug_info = None;
                    data_section = true;
                    return None;
                }
            } else if let Instr::Globl(name) = i {
                if data_section {
                    eprintln!("data label {name}");
                    data = name.clone();
                    return Some((data.clone(), Decl::data().global().into()));
                } else {
                    return Some((name.to_string(), Decl::function().global().into()))
                };
            } else if let Instr::Bytes(b) = i {
                // Define section in the faerie way.
                if let Some(waiting) = waiting_for_debug_info.clone() {
                    waiting_for_debug_info = None;
                    sections.push((waiting, b.clone()));
                } else if data_section {
                    eprintln!("data bytes named {data} = {b:?}");
                    data_objects.push((data.clone(), b.to_vec()));
                }
            }

            None
        }).collect();

        // Declare .debug_aranges
        decls.push((".debug_aranges".to_string(), Decl::section(SectionKind::Debug).into()));

        obj.declarations(decls.into_iter()).map_err(|e| format!("{e:?}"))?;

        let mut relocations = Vec::new();
        let mut function_body = Vec::new();
        let mut in_function = None;

        let mut produced_code = 0;
        let mut handle_def_end = |function_body: &mut Vec<u8>, in_function: &mut Option<String>| -> Result<(), String> {
            if let Some(defname) = in_function.as_ref() {
                if !function_body.is_empty() {
                    eprintln!("obj define {defname}");
                    produced_code += function_body.len();
                    obj.define(defname, function_body.clone()).map_err(|e| format!("{e:?}"))?;
                    *function_body = Vec::new();
                }
            }

            Ok(())
        };

        for i in self.finished_insns.iter() {
            if let Instr::Globl(name) = i {
                handle_def_end(&mut function_body, &mut in_function)?;
                in_function = Some(name.to_string());
            }

            if let Some(f) = in_function.as_ref() {
                i.encode(&mut function_body, &mut relocations, &f);
            }
        }

        handle_def_end(&mut function_body, &mut in_function)?;

        // Create .debug_aranges
        let mut debug_aranges: Vec<u8> = (0..0x20).map(|_| 0).collect();
        write_u32(&mut debug_aranges, 0, 0x1c);
        debug_aranges[4] = 2;
        write_u32(&mut debug_aranges, 6, 0);
        debug_aranges[10] = 4;
        write_u32(&mut debug_aranges, 16, TARGET_ADDR);
        write_u32(&mut debug_aranges, 20, TARGET_ADDR + produced_code as u32);
        sections.push((".debug_aranges".to_string(), debug_aranges));

        for (name, bytes) in sections.iter() {
            obj.define(name, bytes.clone()).map_err(|e| format!("{e:?}"))?;
        }

        for r in relocations.iter() {
            obj.link(Link { from: &r.function, to: &r.reloc_target, at: r.code_location as u64}).map_err(|e| format!("link {e:?}"))?;
        }

        let mut file = NamedTempFile::new().map_err(|e| format!("named temp {e:?}"))?;
        let name = file.path().to_str().unwrap().to_string();
        let mut reread_file = File::open(&name).map_err(|e| format!("reopen {e:?}"))?;
        obj.write(file.into_file()).map_err(|e| format!("obj write {e:?}"))?;
        reread_file.seek(SeekFrom::Start(0)).map_err(|e| format!("seek {e:?}"))?;
        let mut result_buf = Vec::new();
        reread_file.read_to_end(&mut result_buf).map_err(|e| format!("capture {e:?}"))?;

        // Patch up
        let create_patches = |result_buf: &mut [u8]| {
            let elf_loader = ElfLoader::new(result_buf, TARGET_ADDR).expect("should load");
            elf_loader.patch_sections()
        };

        let patches = create_patches(&mut result_buf);

        for (target, value) in patches.into_iter() {
            write_u32(&mut result_buf, target, value);
        }

        Ok(result_buf)
    }

    pub fn new(
        filename: &str,
        sexp: Rc<SExp>,
        env: Rc<SExp>,
        symbol_table: HashMap<String, String>
    ) -> Result<Self, String> {
        let dwarf_builder = DwarfBuilder::new(filename);
        let mut p: Program = Program {
            finished_insns: Vec::new(),
            first_label: Default::default(),
            env_label: Default::default(),
            encounters_of_code: Default::default(),
            labels_by_hash: Default::default(),
            done_programs: Default::default(),
            waiting_programs: Default::default(),
            constants: Default::default(),
            symbol_table: Default::default(),
            current_addr: 12,
            dwarf_builder
        };

        p.symbol_table = symbol_table;
        let loc = Srcloc::start("*env*");
        let envhash = sha256tree(env.clone());
        p.first_label = p.add(sexp);
        p.start_insns();
        p.env_label = p.add_sexp(&loc, &envhash, env);
        p.emit_waiting();
        p.finish_insns()?;
        Ok(p)
    }
}
