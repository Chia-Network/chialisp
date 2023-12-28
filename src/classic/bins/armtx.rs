use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::swap;
use std::path::PathBuf;
use std::str::FromStr;
use std::rc::Rc;

use argh::FromArgs;
use clvmr::Allocator;
use elf_rs::{Elf, ProgramHeaderFlags, ProgramType};
use faerie::{ArtifactBuilder, Decl, Link, SectionKind};
use gimli::{Encoding, Format, LineEncoding, LittleEndian};
use gimli::write::{Address, DirectoryId, Dwarf, FileId, LineProgram, LineString, Section, Sections, UnitId, Unit};
use target_lexicon::triple;
use tempfile::NamedTempFile;

use clvm_tools_rs::classic::clvm::casts::bigint_to_bytes_clvm;
use clvm_tools_rs::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use clvm_tools_rs::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};

use clvm_tools_rs::compiler::clvm::{sha256tree, truthy};
use clvm_tools_rs::compiler::comptypes::CompilerOpts;
use clvm_tools_rs::compiler::compiler::{DefaultCompilerOpts, compile_file};
use clvm_tools_rs::compiler::sexp::{Atom, decode_string, NodeSel, SelectNode, SExp, ThisNode, parse_sexp};
use clvm_tools_rs::compiler::srcloc::Srcloc;

const ENV_PTR: i32 = 4;
const STACK_TOP: i32 = 8;
const NEXT_ALLOC_OFFSET: i32 = 12;

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
    Adr(Register,String),
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
}

impl ToU32 for ArmDataOp {
    fn to_u32(&self) -> u32 {
        match self {
            ArmDataOp::Add => 4 << 21,
            ArmDataOp::And => 0,
            ArmDataOp::Sub => 2 << 21,
            ArmDataOp::Mov => 13 << 21,
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
            out |= r.to_u32();
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
            Instr::Str(rd,rs,off) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | 1 << 26 | 1 << 25 | 1 << 24 | 1 << 23 | Rn(rs.clone()).to_u32() | Rd(rd.clone()).to_u32() | (((65536 + off) & 0xffff) as u32)),
            Instr::Ldr(rd,rs,off) => vec_from_u32(v, ArmCond::Unconditional.to_u32() | 1 << 26 | 1 << 25 | 1 << 24 | 1 << 23 | 1 << 20 | Rn(rs.clone()).to_u32() | Rd(rd.clone()).to_u32() | (((65536 + off) & 0xffff) as u32)),
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
            Instr::Adr(_r,_target) => {
                
                todo!();
            }
            Instr::Lea(_r,_target) => {
                todo!();
            }
            Instr::Swi(_n) => {
                todo!();
            }
            Instr::SwiEq(_n) => {
                todo!();
            }
            Instr::Cmpi(_r,_n) => {
                todo!();
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
            Instr::Adr(r,l) => write!(f, "  adr {r}, {l}"),
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
    type Endian = LittleEndian;

    fn endian(&self) -> Self::Endian {
        return LittleEndian::default();
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
        let dirname = path.into_os_string().to_string_lossy().as_bytes().to_vec();

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

        let unit = Unit::new(encoding, line_program);

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
        row.basic_block = begin_end_block == Some(BeginEndBlock::BeginBlock);
        unit.line_program.generate_row();
    }

    fn start(&mut self, addr: usize) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        self.seq_addr_start = addr;
        unit.line_program.begin_sequence(Some(Address::Constant(addr as u64)));
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

struct Program {
    finished_insns: Vec<Instr>,
    first_label: String,
    env_label: String,
    encounters_of_code: HashMap<Vec<u8>, usize>,
    labels_by_hash: HashMap<Vec<u8>, String>,
    waiting_programs: HashMap<String, Rc<SExp>>,
    constants: HashMap<Vec<u8>, Constant>,
    symbol_table: HashMap<String, String>,
    current_addr: usize,
    dispatch_addr: u32,
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

// Read a mappble object file and obtain an address we can use for its entry
// point.
fn find_entry_point_address(object_file: &str) -> Result<u32, String> {
    let file = fs::read(object_file).map_err(|e| format!("error reading {object_file}: {e:?}"))?;
    let elf = Elf::from_bytes(&file).map_err(|e| format!("{object_file} elf error: {e:?}"))?;

    let elf32 =
        match elf {
            Elf::Elf32(elf) => elf,
            _ => {
                return Err("need a 32 bit arm elf binary for support".to_string());
            }
        };

    // Find the entry point address of the target binary.
    for phdr in elf32.program_header_iter() {
        if matches!(phdr.ph_type(), ProgramType::LOAD) && phdr.flags().contains(ProgramHeaderFlags::EXECUTE) {
            return Ok(phdr.vaddr() as u32);
        }
    }

    Err(format!("could not find an executable program header in {object_file}"))
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
        self.push(loc, Instr::Swi(0));
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
            Instr::SwiEq(0)
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
                Instr::Adr(Register::LR, "dispatch_code".to_string())
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
            self.push(
                loc,
                Instr::Label(else_label),
            );
            self.push(
                loc,
                Instr::Bl(else_clause),
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
        }

        // Generic operator emulation.
        todo!("generic op {a:?}");
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
                        Instr::SwiEq(0)
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
        self.waiting_programs.insert(body_label.clone(), sexp.clone());
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
        let mut current_waiting = HashMap::new();
        swap(&mut current_waiting, &mut self.waiting_programs);

        while !current_waiting.is_empty() {
            for (label, sexp) in current_waiting.iter() {
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

            current_waiting.clear();
            swap(&mut current_waiting, &mut self.waiting_programs);
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
            Instr::Ldr(Register::SP, Register::R(0), STACK_TOP),
            Instr::Bl(self.first_label.clone()),
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
            Instr::Long(2),
            Instr::Addr(self.env_label.clone()),
            Instr::Long(0x7ff0),
            Instr::Long(0x1000000),
            Instr::Addr("_funaddrs".to_string()),

            Instr::Align4,
            Instr::Globl("dispatch_code".to_string()),
            Instr::Label("dispatch_code".to_string()),
            Instr::Long(self.dispatch_addr as usize),

            // Write the function table.
            Instr::Align4,
            Instr::Globl("_funaddrs".to_string()),
            Instr::Label("_funaddrs".to_string())
        ].iter() {
            self.push(&srcloc, i.clone());
        }

        let mut labels_by_hash = HashMap::new();
        swap(&mut labels_by_hash, &mut self.labels_by_hash);
        for (k, v) in labels_by_hash.iter() {
            self.push(
                &srcloc,
                Instr::Addr(v.clone())
            );
            self.push(
                &srcloc,
                Instr::Bytes(k.to_vec())
            );
        }
        swap(&mut labels_by_hash, &mut self.labels_by_hash);

        self.push(
            &srcloc,
            Instr::Long(0)
        );

        // Write the constant data.
        let mut constants = HashMap::new();
        swap(&mut constants, &mut self.constants);
        for (_, c) in constants.iter() {
            match c {
                Constant::Atom(label, bytes) => {
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
                Constant::Cons(label, a_label, b_label) => {
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
        }
        swap(&mut constants, &mut self.constants);

        self.dwarf_builder.write(&mut self.finished_insns).unwrap();

        Ok(())
    }

    fn to_elf(&self, output: &str) -> Result<Vec<u8>, String> {
        let mut sections = Vec::new();
        let mut obj = ArtifactBuilder::new(triple!("arm-unknown-unknown-unknown-elf"))
            .name(output.to_owned())
            .finish();
        // Collect declarations
        let decls: Vec<(String, Decl)> = self.finished_insns.iter().filter_map(|i| {
            if let Instr::Section(name) = i {
                sections.push(name.clone());
                if name.starts_with(".debug") {
                    return Some((name.to_string(), Decl::section(SectionKind::Debug).into()));
                } else if name == ".text" {
                    return Some((name.to_string(), Decl::section(SectionKind::Text).into()));
                } else {
                    return Some((name.to_string(), Decl::section(SectionKind::Data).into()));
                }
            } else if let Instr::Globl(name) = i {
                return Some((name.to_string(), Decl::function().global().into()));
            }

            None
        }).collect();
        obj.declarations(decls.into_iter()).map_err(|e| format!("{e:?}"))?;

        let mut relocations = Vec::new();
        let mut function_body = Vec::new();
        let mut in_function = None;
        for i in self.finished_insns.iter() {
            if let Instr::Globl(name) = i {
                if !function_body.is_empty() {
                    obj.define(&name, function_body).map_err(|e| format!("{e:?}"))?;
                    function_body = Vec::new();
                }

                in_function = Some(name.to_string());
            }

            if let Some(f) = in_function.as_ref() {
                i.encode(&mut function_body, &mut relocations, &f);
            }
        }

        for r in relocations.iter() {
            obj.link(Link { from: &r.function, to: &r.reloc_target, at: r.code_location as u64}).map_err(|e| format!("{e:?}"))?;
        }

        let file = NamedTempFile::new().map_err(|e| format!("{e:?}"))?;
        let name = file.path().to_str().unwrap().to_string();
        obj.write(file.into_file()).map_err(|e| format!("{e:?}"))?;
        let mut file = File::open(&name).map_err(|e| format!("{e:?}"))?;
        file.seek(SeekFrom::Start(0)).map_err(|e| format!("{e:?}"))?;
        let mut result_buf = Vec::new();
        file.read_to_end(&mut result_buf).map_err(|e| format!("{e:?}"))?;
        Ok(result_buf)
    }

    fn new(
        filename: &str,
        dispatch_addr: u32,
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
            waiting_programs: Default::default(),
            constants: Default::default(),
            symbol_table: Default::default(),
            current_addr: 12,
            dispatch_addr,
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

/// Translate a chialisp program to debug as an arm elf executable.
#[derive(FromArgs)]
struct Args {
    /// include paths
    #[argh(option, short='i')]
    pub include: Vec<String>,

    #[argh(option, short='s', description="support program for clvm execution")]
    pub support: String,

    #[argh(option, short='o', description="output file")]
    pub output: String,

    /// file name
    #[argh(positional)]
    pub filename: String,

    /// initial env
    #[argh(positional)]
    pub env: String,
}

fn main() {
     let args: Args = argh::from_env();

    let mut search_paths = args.include.clone();

    let argfile =
        if let Ok(res) = fs::read(&args.filename) {
            res
        } else {
            panic!("error reading {}", args.filename);
        };

    let srcloc = Srcloc::start(&args.filename);
    let mut allocator = Allocator::new();
    let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
    let mut symbol_table = HashMap::new();
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(&args.filename)).set_search_paths(&search_paths);
    let compiled =
        match compile_file(
            &mut allocator,
            runner,
            opts,
            &decode_string(&argfile),
            &mut symbol_table,
        ) {
            Ok(res) => res,
            Err(e) => panic!("{:?}", e)
        };
    let mut env =
        match parse_sexp(srcloc.clone(), args.env.bytes()) {
            Ok(res) => res,
            Err(e) => panic!("{:?}", e)
        };

    if env.is_empty() {
        env.push(Rc::new(SExp::Nil(srcloc.clone())));
    }

    let dispatch_addr = find_entry_point_address(&args.support).expect("should be readable");
    let program = Program::new(
        &args.filename,
        dispatch_addr,
        Rc::new(compiled),
        env[0].clone(),
        symbol_table
    ).expect("should generate");

    let elf_out = program.to_elf(&args.output).expect("should be writable");
    fs::write(&args.output, &elf_out).expect("should be able to write file");
    todo!();
}
