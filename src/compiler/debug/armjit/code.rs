use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Formatter;
use std::mem::swap;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;

use sha2::Digest;
use sha2::Sha256;

use faerie::{ArtifactBuilder, Decl, Link, SectionKind};
use gimli;
use gimli::constants::{
    DW_AT_byte_size, DW_AT_encoding, DW_AT_frame_base, DW_AT_high_pc, DW_AT_language,
    DW_AT_location, DW_AT_low_pc, DW_AT_name, DW_AT_type, DW_TAG_base_type,
    DW_TAG_formal_parameter, DW_TAG_pointer_type, DW_TAG_subprogram, DW_TAG_variable, DW_LANG_C99,
};
use gimli::write::{
    Address, Attribute, AttributeValue, CallFrameInstruction, CieId, CommonInformationEntry,
    DirectoryId, Dwarf, DwarfUnit, Expression, FileId, FrameDescriptionEntry, FrameTable,
    LineProgram, LineString, Location, LocationList, Range, RangeList, Section, Sections, Unit,
    UnitEntryId, UnitId,
};
use gimli::Arm;
use gimli::{DW_ATE_unsigned, DwAte, Encoding, Format, LineEncoding};
use target_lexicon::triple;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero, Bytes, BytesFromType};
use crate::compiler::debug::armjit::load::{write_u32, ElfLoader};
use crate::compiler::debug::{
    debug_decode_string as decode_string, debug_dequote as dequote,
    debug_find_all_by_hash as find_all_by_hash, debug_is_atom as is_atom,
    debug_is_wrapped_atom as is_wrapped_atom, debug_parse_sexp as parse_sexp,
    debug_sha256tree as sha256tree, debug_start_loc, ConcreteSExp, DebugSExp, DebugSExpValue,
    Srcloc,
};
use crate::util::Number;

pub const NEXT_ALLOC_OFFSET: i32 = 0;

pub const SWI_DONE: usize = 0;
pub const SWI_THROW: usize = 1;
pub const SWI_DISPATCH_NEW_CODE: usize = 2;
pub const SWI_DISPATCH_INSTRUCTION: usize = 3;
pub const SWI_PRINT_EXPR: usize = 4;

pub const TARGET_ADDR: u32 = 0x1000;

pub struct ElfObject {
    pub object_file: Vec<u8>,
    pub synthetic_source: String,
}

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
pub enum Register {
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
            Register::PC => 15,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Instr {
    Align4,
    Section(String),
    Globl(String),
    Label(String),
    Space(usize, u8),
    Add(Register, Register, Register),
    Addi(Register, Register, i32),
    AddiEq(Register, Register, i32),
    Sub(Register, Register, Register),
    Subi(Register, Register, i32),
    Andi(Register, Register, i32),
    Push(Vec<Register>),
    Pop(Vec<Register>),
    Mov(Register, i32),
    Str(Register, Register, i32),
    Ldr(Register, Register, i32),
    B(String),
    Bl(String),
    Bx(Register),
    Blx(Register),
    Lea(Register, String),
    Swi(usize),
    SwiEq(usize),
    Cmpi(Register, usize),
    Long(usize),
    Addr(String, bool),
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
            Instr::Space(size, _fill) => *size,
            Instr::Globl(_l) => 0,
            Instr::Label(_l) => 0,
            Instr::Lea(_, _) => 12,
            Instr::Bytes(v) => v.len(),
            _ => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BeginEndBlock {
    BeginBlock,
    EndBlock,
}

pub trait Encodable {
    fn encode(&self, v: &mut Vec<u8>, r: &mut Vec<Relocation>, function: &str);
}

enum ArmCond {
    Unconditional,
    Equal,
    GreaterOrEqual,
}

impl ToU32 for ArmCond {
    fn to_u32(&self) -> u32 {
        match self {
            ArmCond::Unconditional => 14 << 28,
            ArmCond::Equal => 0,
            ArmCond::GreaterOrEqual => 10 << 28,
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
    Swi,
}

impl ToU32 for ArmOp {
    fn to_u32(&self) -> u32 {
        match self {
            ArmOp::Swi => 15 << 24,
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

pub enum RelocationKind {
    Long,
    Branch,
}

pub struct Relocation {
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
            Instr::Space(n, val) => {
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
            Instr::Addr(target, text) => {
                r.push(Relocation {
                    kind: RelocationKind::Long,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                let offset = if *text { 4 } else { 0 };
                vec_from_u32(v, offset);
            }
            Instr::Add(r_d, r_s, r_a) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | ArmDataOp::Add.to_u32()
                    | Rn(r_s.clone()).to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | Rm(0, r_a.clone()).to_u32(),
            ),
            Instr::Addi(r_d, r_s, imm) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | ArmDataOp::Add.to_u32()
                    | Rn(r_s.clone()).to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | (1 << 25)
                    | (*imm as u32),
            ),
            Instr::AddiEq(r_d, r_s, imm) => vec_from_u32(
                v,
                ArmCond::Equal.to_u32()
                    | ArmDataOp::Add.to_u32()
                    | Rn(r_s.clone()).to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | (1 << 25)
                    | (*imm as u32),
            ),
            Instr::Sub(r_d, r_s, r_a) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | ArmDataOp::Sub.to_u32()
                    | Rn(r_s.clone()).to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | Rm(0, r_a.clone()).to_u32(),
            ),
            Instr::Subi(r_d, r_s, imm) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | ArmDataOp::Sub.to_u32()
                    | Rn(r_s.clone()).to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | (1 << 25)
                    | (*imm as u32),
            ),
            Instr::Andi(r_d, r_s, imm) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | ArmDataOp::And.to_u32()
                    | Rn(r_s.clone()).to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | (1 << 25)
                    | (*imm as u32),
            ),
            Instr::Mov(r_d, imm) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | ArmDataOp::Mov.to_u32()
                    | Rd(r_d.clone()).to_u32()
                    | (1 << 25)
                    | (*imm as u32),
            ),
            Instr::Push(rs) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | 4 << 25
                    | Rn(Register::SP).to_u32()
                    | 1 << 21
                    | 1 << 24
                    | rs.to_u32(),
            ),
            Instr::Pop(rs) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | 4 << 25
                    | Rn(Register::SP).to_u32()
                    | 1 << 20
                    | 1 << 21
                    | 1 << 23
                    | rs.to_u32(),
            ),
            Instr::Str(rd, rs, off) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | 1 << 26
                    | 1 << 24
                    | 1 << 23
                    | Rn(rs.clone()).to_u32()
                    | Rd(rd.clone()).to_u32()
                    | (((65536 + off) & 0xff) as u32),
            ),
            Instr::Ldr(rd, rs, off) => vec_from_u32(
                v,
                ArmCond::Unconditional.to_u32()
                    | 1 << 26
                    | 1 << 24
                    | 1 << 23
                    | 1 << 20
                    | Rn(rs.clone()).to_u32()
                    | Rd(rd.clone()).to_u32()
                    | (((65536 + off) & 0xff) as u32),
            ),
            Instr::B(target) => {
                r.push(Relocation {
                    kind: RelocationKind::Branch,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 5 << 25);
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
            Instr::Blx(r) => {
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 0x12fff30 | r.to_u32());
            }
            Instr::Lea(rd, target) => {
                // Emit a load from +8 (0 as encoded).
                vec_from_u32(
                    v,
                    ArmCond::Unconditional.to_u32()
                        | 1 << 26
                        | 1 << 24
                        | 1 << 23
                        | 1 << 20
                        | Rn(Register::PC).to_u32()
                        | Rd(rd.clone()).to_u32(),
                );
                // Emit a jump to +8
                vec_from_u32(v, ArmCond::Unconditional.to_u32() | 5 << 25 | 0);
                r.push(Relocation {
                    kind: RelocationKind::Long,
                    function: function.to_string(),
                    code_location: v.len(),
                    reloc_target: target.clone(),
                });
                // Relocatable space.
                vec_from_u32(v, 4);
            }
            Instr::Swi(n) => {
                vec_from_u32(
                    v,
                    ArmCond::Unconditional.to_u32() | ArmOp::Swi.to_u32() | (*n as u32),
                );
            }
            Instr::SwiEq(n) => {
                vec_from_u32(
                    v,
                    ArmCond::Equal.to_u32() | ArmOp::Swi.to_u32() | (*n as u32),
                );
            }
            Instr::Cmpi(r, n) => {
                vec_from_u32(
                    v,
                    ArmCond::Unconditional.to_u32()
                        | ArmDataOp::Cmp.to_u32()
                        | 1 << 25
                        | 1 << 20
                        | Rn(r.clone()).to_u32()
                        | (*n as u32),
                );
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
            Instr::Space(size, fill) => write!(f, "  .space {size},{fill}"),
            Instr::Globl(l) => write!(f, "  .globl {l}"),
            Instr::Label(l) => write!(f, "{l}:"),
            Instr::Add(r_d, r_s, r_a) => write!(f, "  add {r_d}, {r_s}, {r_a}"),
            Instr::Addi(r_d, r_s, imm) => write!(f, "  add {r_d}, {r_s}, #{imm}"),
            Instr::AddiEq(r_d, r_s, imm) => write!(f, "  addeq {r_d}, {r_s}, #{imm}"),
            Instr::Andi(r_d, r_s, imm) => write!(f, "  and {r_d}, {r_s}, #{imm}"),
            Instr::Sub(r_d, r_s, r_a) => write!(f, "  sub {r_d}, {r_s}, {r_a}"),
            Instr::Subi(r_d, r_s, imm) => write!(f, "  sub {r_d}, {r_s}, #{imm}"),
            Instr::Cmpi(r, imm) => write!(f, "  cmp {r}, #{imm}"),
            Instr::Push(rs) => {
                write!(f, "  push {{")?;
                let mut sep = "";
                for r in rs.iter() {
                    write!(f, "{sep}{r}")?;
                    sep = ", ";
                }
                write!(f, "}}")
            }
            Instr::Pop(rs) => {
                write!(f, "  pop {{")?;
                let mut sep = "";
                for r in rs.iter() {
                    write!(f, "{sep}{r}")?;
                    sep = ", ";
                }
                write!(f, "}}")
            }
            Instr::Mov(r_d, imm) => write!(f, "  mov {r_d}, #{imm}"),
            Instr::Str(r_s, r_a, imm) => write!(f, "  str {r_s}, [{r_a}, #{imm}]"),
            Instr::Ldr(r_d, r_a, imm) => write!(f, "  ldr {r_d}, [{r_a}, #{imm}]"),
            Instr::B(l) => write!(f, "  b {l}"),
            Instr::Bl(l) => write!(f, "  bl {l}"),
            Instr::Bx(r) => write!(f, "  bx {r}"),
            Instr::Blx(r) => write!(f, "  blx {r}"),
            Instr::Lea(r, l) => write!(f, "  ldr {r}, ={l}"),
            Instr::Swi(n) => write!(f, "  swi {n}"),
            Instr::SwiEq(n) => write!(f, "  swieq {n}"),
            Instr::Long(n) => write!(f, "  .long {n}"),
            Instr::Addr(lbl, _) => write!(f, "  .long {lbl}"),
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
    synthetic_source_path: String,
    synthetic_file_id: Option<FileId>,
    synthetic_source_lines: Vec<String>,
    synthetic_expr_line_by_key: HashMap<Vec<u8>, u64>,
    pointer_type: UnitEntryId,
    u32_type: UnitEntryId,

    seq_addr_start: usize,
    target_addr: u32,

    symbol_table: Rc<HashMap<String, String>>,

    dwarf: Dwarf,
    frame_table: FrameTable,
    cfi_cie_id: Option<CieId>,
    last_row_source: Option<(String, u64, u64)>,
    last_statement_source_line: Option<(String, u64)>,
}

struct VariableLocationInfo<'a> {
    pub beginning: u64,
    pub end: u64,
    start_expr: &'a dyn Fn() -> Expression,
}

#[derive(Default, Clone, Debug)]
struct DwarfSectionWriter {
    pub written: Vec<u8>,
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
    fn new(
        filename: &str,
        elf_output: &str,
        target_addr: u32,
        symbol_table: Rc<HashMap<String, String>>,
    ) -> Self {
        let mut path = PathBuf::new();
        path.push(filename);
        path.pop();
        let mut dirname = path.into_os_string().to_string_lossy().as_bytes().to_vec();

        if dirname.is_empty() {
            dirname = b".".to_vec();
        }

        path = PathBuf::new();
        path.push(filename);
        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy().as_bytes().to_vec())
            .unwrap_or_else(|| filename.as_bytes().to_vec());

        let line_encoding = LineEncoding {
            minimum_instruction_length: 4,
            maximum_operations_per_instruction: 1,
            default_is_stmt: false,
            line_base: 0,
            line_range: 1,
        };

        let mut dwarf = Dwarf::default();
        let encoding = Encoding {
            address_size: 4,
            format: Format::Dwarf32,
            version: 2,
        };

        let dirstring = LineString::String(dirname.clone());
        let filestring = LineString::String(filename.clone());
        let mut line_program = LineProgram::new(
            encoding.clone(),
            line_encoding,
            dirstring.clone(),
            filestring.clone(),
            None,
        );
        let mut directory_to_id = HashMap::new();
        let directory_id = line_program.add_directory(dirstring);
        directory_to_id.insert(dirname.clone(), directory_id.clone());
        let mut file_to_id = HashMap::new();
        let file_id = line_program.add_file(filestring, directory_id.clone(), None);
        file_to_id.insert(filename.clone(), (directory_id, file_id));

        let mut unit = Unit::new(encoding, line_program);

        unit.ranges.add(RangeList(vec![Range::BaseAddress {
            address: Address::Constant(target_addr as u64),
        }]));
        unit.locations.add(LocationList(vec![Location::BaseAddress {
            address: Address::Constant(target_addr as u64),
        }]));
        let unit_ent = unit.get_mut(unit.root());
        unit_ent.set(
            DW_AT_low_pc,
            AttributeValue::Address(Address::Constant(target_addr as u64)),
        );
        unit_ent.set(DW_AT_name, AttributeValue::String(filename.clone()));
        unit_ent.set(DW_AT_language, AttributeValue::Language(DW_LANG_C99));

        let unit_id = dwarf.units.add(unit);
        let mut mutable_unit = dwarf.units.get_mut(unit_id);
        let base_type_id = mutable_unit.add(mutable_unit.root(), DW_TAG_base_type);
        let mut base_ent = mutable_unit.get_mut(base_type_id);
        base_ent.set(DW_AT_byte_size, AttributeValue::Data1(4));
        base_ent.set(DW_AT_encoding, AttributeValue::Encoding(DW_ATE_unsigned));
        base_ent.set(DW_AT_name, AttributeValue::String(b"word".to_vec()));
        let type_id = mutable_unit.add(mutable_unit.root(), DW_TAG_pointer_type);
        let mut type_ent = mutable_unit.get_mut(type_id);
        type_ent.set(DW_AT_name, AttributeValue::String(b"sexp".to_vec()));
        type_ent.set(DW_AT_byte_size, AttributeValue::Udata(4));
        type_ent.set(DW_AT_type, AttributeValue::UnitRef(base_type_id));

        let mut obj = DwarfBuilder {
            seq_addr_start: 0,
            target_addr,
            unit_id,
            file_to_id,
            directory_to_id,
            synthetic_source_path: format!("{elf_output}.clsp"),
            synthetic_file_id: None,
            synthetic_source_lines: Vec::new(),
            synthetic_expr_line_by_key: HashMap::new(),
            pointer_type: type_id,
            u32_type: base_type_id,
            symbol_table,
            dwarf,
            frame_table: FrameTable::default(),
            cfi_cie_id: None,
            last_row_source: None,
            last_statement_source_line: None,
        };
        let cfi_encoding = Encoding {
            address_size: 4,
            format: Format::Dwarf32,
            version: 1,
        };
        let mut cie = CommonInformationEntry::new(cfi_encoding, 1, -4, Arm::R14);
        cie.add_instruction(CallFrameInstruction::Cfa(Arm::R13, 0));
        obj.cfi_cie_id = Some(obj.frame_table.add_cie(cie));

        let synthetic_source_path = obj.synthetic_source_path.clone();
        let (_, synthetic_file_id) = obj.add_file(&synthetic_source_path);
        obj.synthetic_file_id = Some(synthetic_file_id);

        obj
    }

    fn add_file_having_dirid(
        &mut self,
        dirid: DirectoryId,
        filename: &[u8],
    ) -> (DirectoryId, FileId) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        let filestring = LineString::String(filename.to_vec());
        let fileid = unit.line_program.add_file(filestring, dirid.clone(), None);
        self.file_to_id
            .insert(filename.to_vec(), (dirid.clone(), fileid.clone()));
        (dirid, fileid)
    }

    fn add_file(&mut self, filename_str: &str) -> (DirectoryId, FileId) {
        let mut path = PathBuf::new();
        path.push(filename_str);
        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy().as_bytes().to_vec())
            .unwrap_or_else(|| filename_str.as_bytes().to_vec());
        if let Some((dirid, fileid)) = self.file_to_id.get(&filename) {
            return (*dirid, *fileid);
        }

        path = PathBuf::new();
        path.push(filename_str);
        path.pop();

        let dirname = path.into_os_string().to_string_lossy().as_bytes().to_vec();
        let use_dirname = if dirname.is_empty() {
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

    fn synthetic_expr_key(loc: &Srcloc, source_sexp: &impl fmt::Display) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(loc.file.as_bytes());
        hasher.update((loc.line as u64).to_le_bytes());
        hasher.update((loc.col as u64).to_le_bytes());
        if let Some(until) = loc.until.as_ref() {
            hasher.update((until.line as u64).to_le_bytes());
            hasher.update((until.col as u64).to_le_bytes());
        }
        hasher.update(source_sexp.to_string().as_bytes());
        hasher.finalize().to_vec()
    }

    fn add_synthetic_line(&mut self, loc: &Srcloc, source_sexp: &impl fmt::Display) -> u64 {
        let synthetic_key = Self::synthetic_expr_key(loc, source_sexp);
        if let Some(line) = self.synthetic_expr_line_by_key.get(&synthetic_key) {
            return *line;
        }

        const MAX_SYNTHETIC_EXPR_CHARS: usize = 220;
        let expression = source_sexp.to_string();
        let truncated: String = expression.chars().take(MAX_SYNTHETIC_EXPR_CHARS).collect();
        let display_expr = if expression.chars().count() > MAX_SYNTHETIC_EXPR_CHARS {
            format!("{truncated}...")
        } else {
            truncated
        };
        self.synthetic_source_lines
            .push(format!("{loc} => {display_expr}"));
        let line = self.synthetic_source_lines.len() as u64;
        self.synthetic_expr_line_by_key.insert(synthetic_key, line);
        line
    }

    fn synthetic_source(&self) -> String {
        let mut synthetic_source = self.synthetic_source_lines.join("\n");
        if !synthetic_source.is_empty() {
            synthetic_source.push('\n');
        }
        synthetic_source
    }

    fn add_instr(
        &mut self,
        addr: usize,
        loc: &Srcloc,
        source_sexp: &impl fmt::Display,
        instr: Instr,
        begin_end_block: Option<BeginEndBlock>,
    ) {
        if !self
            .dwarf
            .units
            .get_mut(self.unit_id)
            .line_program
            .in_sequence()
        {
            return;
        }
        let source_file = loc.file.to_string();
        let source_key = (source_file.clone(), loc.line as u64, loc.col as u64);
        let source_line_key = (source_file, loc.line as u64);
        let synthetic_file_id = self
            .synthetic_file_id
            .expect("synthetic source file registered");
        let mut using_synthetic_file = loc.file.starts_with('*');
        let (mut file_id, mut line, mut col) = if using_synthetic_file {
            (
                synthetic_file_id,
                self.add_synthetic_line(loc, source_sexp),
                1_u64,
            )
        } else {
            let (_, file_id) = self.add_file(loc.file.as_str());
            (file_id, loc.line as u64, loc.col as u64)
        };
        let source_changed = self
            .last_row_source
            .as_ref()
            .map(|prev| prev != &source_key)
            .unwrap_or(true);
        let control_flow_or_dispatch = matches!(
            instr,
            Instr::B(_)
                | Instr::Bl(_)
                | Instr::Bx(_)
                | Instr::Blx(_)
                | Instr::Swi(_)
                | Instr::SwiEq(_)
        );
        let is_statement = addr == self.seq_addr_start
            || begin_end_block == Some(BeginEndBlock::BeginBlock)
            || source_changed
            || control_flow_or_dispatch;
        if is_statement && !using_synthetic_file {
            let same_statement_source_line = self
                .last_statement_source_line
                .as_ref()
                .map(|prev| prev == &source_line_key)
                .unwrap_or(false);
            // Keep the first statement point on a real source line, and map
            // repeated statement boundaries on that line to the synthetic file.
            if same_statement_source_line {
                file_id = synthetic_file_id;
                line = self.add_synthetic_line(loc, source_sexp);
                col = 1;
                using_synthetic_file = true;
            }
        }
        let unit = self.dwarf.units.get_mut(self.unit_id);
        let row = unit.line_program.row();
        row.address_offset = (addr - self.seq_addr_start) as u64;
        row.file = file_id;
        row.line = line;
        row.column = col;
        row.is_statement = is_statement;
        eprintln!("line row {} at {:?}", row.address_offset, loc);
        row.basic_block = begin_end_block == Some(BeginEndBlock::BeginBlock);
        let emitted_statement = row.is_statement;
        unit.line_program.generate_row();
        if emitted_statement {
            if using_synthetic_file {
                self.last_statement_source_line = None;
            } else {
                self.last_statement_source_line = Some(source_line_key);
            }
        }
        self.last_row_source = Some(source_key);
    }

    fn start(&mut self, addr: usize) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        self.seq_addr_start = addr;
        self.last_row_source = None;
        self.last_statement_source_line = None;
        unit.line_program.begin_sequence(Some(Address::Constant(
            (addr + self.target_addr as usize) as u64,
        )));
    }

    fn end(&mut self, addr: usize) {
        let unit = self.dwarf.units.get_mut(self.unit_id);
        unit.line_program
            .end_sequence((addr - self.seq_addr_start) as u64);
    }

    fn match_function(&self, label: &str) -> Option<(String, String)> {
        let symbol_hash = if let Some(stripped) = label.strip_prefix('_') {
            let hash = stripped
                .chars()
                .take_while(|c| *c != '_')
                .collect::<String>();
            if self.symbol_table.contains_key(&hash) {
                Some(hash)
            } else {
                None
            }
        } else {
            self.symbol_table
                .iter()
                .find(|(hash, name)| {
                    *name == label
                        && hash.len() == 64
                        && hash.as_bytes().iter().all(|b| b.is_ascii_hexdigit())
                })
                .map(|(hash, _)| hash.clone())
        };
        if let Some(symbol_hash) = symbol_hash {
            let mut stripped = symbol_hash.as_bytes().to_vec();
            let name = self
                .symbol_table
                .get(&symbol_hash)
                .cloned()
                .unwrap_or_else(|| label.to_string());
            let mut left_stripped = stripped.clone();
            left_stripped.append(&mut b"_left_env".to_vec());
            let left_env = if let Some(res) = self
                .symbol_table
                .get(&decode_string(&left_stripped))
                .cloned()
            {
                res == "1"
            } else {
                false
            };
            eprintln!("{name} left_env {left_env}");
            stripped.append(&mut b"_arguments".to_vec());
            let args = self
                .symbol_table
                .get(&decode_string(&stripped))
                .map(|s| {
                    if left_env {
                        format!("(() . {s})")
                    } else {
                        s.to_string()
                    }
                })
                .unwrap_or_else(|| {
                    if left_env {
                        "(() . ENV)".to_string()
                    } else {
                        "ENV".to_string()
                    }
                });
            return Some((name, args));
        }

        None
    }

    fn add_arguments(
        &mut self,
        subprogram_id: UnitEntryId,
        locations: &[VariableLocationInfo],
        here: Number,
        path: Number,
        args: impl DebugSExp,
    ) {
        eprintln!("add_arguments {here} {path} {args}");
        match args.explode() {
            DebugSExpValue::Cons(_, a, b) => {
                self.add_arguments(subprogram_id, locations, here.clone() << 1, path.clone(), a);
                self.add_arguments(subprogram_id, locations, here.clone() << 1, path | here, b);
            }
            DebugSExpValue::Atom(_, a) => {
                let argname = decode_string(&a);
                let unit = self.dwarf.units.get_mut(self.unit_id);

                let mut loclist = Vec::new();

                for l in locations.iter() {
                    let mut expr = (l.start_expr)();

                    let mut i = bi_one();
                    while i < here {
                        if (path.clone() & i.clone()) != bi_zero() {
                            expr.op_plus_uconst(4);
                        }
                        expr.op_deref();
                        i <<= 1;
                    }

                    loclist.push(Location::StartEnd {
                        begin: Address::Constant(l.beginning),
                        end: Address::Constant(l.end),
                        data: expr,
                    });
                }

                let loc_list_id = unit.locations.add(LocationList(loclist));

                let at_id = unit.add(subprogram_id, DW_TAG_formal_parameter);
                let at_ent = unit.get_mut(at_id);
                at_ent.set(
                    DW_AT_name,
                    AttributeValue::String(argname.as_bytes().to_vec()),
                );
                at_ent.set(DW_AT_type, AttributeValue::UnitRef(self.pointer_type));
                at_ent.set(DW_AT_location, AttributeValue::LocationListRef(loc_list_id));

                /*
                let at_id2 = unit.add(subprogram_id, DW_TAG_variable);
                let at_ent = unit.get_mut(at_id2);
                at_ent.set(
                    DW_AT_name,
                    AttributeValue::String(argname.as_bytes().to_vec()),
                );
                at_ent.set(DW_AT_type, AttributeValue::UnitRef(self.pointer_type));
                */
            }
            _ => {}
        }
    }

    // Create dwarf traffic needed to ensure that gdb can find the locals.
    fn decorate_function(
        &mut self,
        label: &str,
        addr: usize,
        size: usize,
        preferred_name: Option<&str>,
    ) -> Option<String> {
        let mut fde = FrameDescriptionEntry::new(
            Address::Constant((self.target_addr as usize + addr) as u64),
            size as u32,
        );
        // Function prolog:
        //   push {fp, lr}
        //   add  fp, sp, #4
        //   sub  sp, sp, #0x18
        //
        // Function epilog:
        //   sub  sp, fp, #4
        //   pop  {fp, lr}
        //   bx   lr
        //
        // The canonical frame address tracks caller SP through the body:
        // CFA = fp + 4. Saved FP/LR are at CFA-8/CFA-4.
        fde.add_instruction(4, CallFrameInstruction::Cfa(Arm::R13, 8));
        fde.add_instruction(4, CallFrameInstruction::Offset(Arm::R11, -8));
        fde.add_instruction(4, CallFrameInstruction::Offset(Arm::R14, -4));
        fde.add_instruction(8, CallFrameInstruction::Cfa(Arm::R11, 4));
        // After `pop {fp, lr}` and before `bx lr`, execution has returned to the
        // entry-state calling convention for this frame.
        let post_pop_offset = (size.saturating_sub(4)) as u32;
        fde.add_instruction(post_pop_offset, CallFrameInstruction::Cfa(Arm::R13, 0));
        fde.add_instruction(post_pop_offset, CallFrameInstruction::Restore(Arm::R11));
        fde.add_instruction(post_pop_offset, CallFrameInstruction::Restore(Arm::R14));
        let cfi_cie_id = self
            .cfi_cie_id
            .expect("CFI CIE should be initialized for unwind info");
        self.frame_table.add_fde(cfi_cie_id, fde);

        let matched_signature = self.match_function(&label);
        let name = preferred_name
            .map(str::to_string)
            .or_else(|| matched_signature.as_ref().map(|(name, _)| name.clone()))
            .unwrap_or_else(|| label.to_string());
        let args = matched_signature
            .as_ref()
            .map(|(_, args)| args.clone())
            .unwrap_or_else(|| "ENV".to_string());

        // We'll make 3 subprograms to represent where the current arguments can be arrived
        // at from, then decorate all of them with the argument retriever below.

        eprintln!("get subprogram");
        let mut subprogram_names = vec![name.clone()];
        if name != label && preferred_name.is_none() && matched_signature.is_none() {
            // Keep a typed DIE for both the colloquial and emitted symbol names so
            // either one resolves to the same pointer return type in debuggers.
            subprogram_names.push(label.to_string());
        }
        let subprogram_ids = {
            let unit = self.dwarf.units.get_mut(self.unit_id);
            let mut subprogram_ids = Vec::with_capacity(subprogram_names.len());
            for subprogram_name in subprogram_names.iter() {
                let subprogram_id = unit.add(unit.root(), DW_TAG_subprogram);

                // Frame pointer for the function.
                let mut fbexpr_mid = Expression::new();
                fbexpr_mid.op_breg(gimli::Register(13), 0);
                let mut loclist = Vec::new();
                loclist.push(Location::StartEnd {
                    begin: Address::Constant(addr as u64),
                    end: Address::Constant((addr + size) as u64),
                    data: fbexpr_mid,
                });
                let loc_list_id = unit.locations.add(LocationList(loclist));
                let mut sub_ent = unit.get_mut(subprogram_id);
                sub_ent.set(
                    DW_AT_name,
                    AttributeValue::String(subprogram_name.as_bytes().to_vec()),
                );
                sub_ent.set(DW_AT_type, AttributeValue::UnitRef(self.pointer_type));
                sub_ent.set(
                    DW_AT_low_pc,
                    AttributeValue::Address(Address::Constant(
                        (self.target_addr as usize + addr) as u64,
                    )),
                );
                sub_ent.set(DW_AT_high_pc, AttributeValue::Udata(size as u64));
                sub_ent.set(
                    DW_AT_frame_base,
                    AttributeValue::LocationListRef(loc_list_id),
                );
                subprogram_ids.push(subprogram_id);
            }
            subprogram_ids
        };
        eprintln!("about to parse args");
        let srcloc = debug_start_loc("*args*");
        if let Ok(parsed) = parse_sexp(srcloc.clone(), args.bytes()) {
            let self_u32_type = self.u32_type;
            let early_reg_closure = move || {
                let mut early_reg_expr = Expression::new();
                early_reg_expr.op_regval_type(gimli::Register(7), self_u32_type);
                early_reg_expr
            };
            let frame_closure = || {
                let mut frame_expr = Expression::new();
                frame_expr.op_fbreg(12);
                frame_expr.op_deref();
                frame_expr
            };

            let locations = vec![
                VariableLocationInfo {
                    beginning: addr as u64,
                    end: addr as u64 + 0x1c,
                    start_expr: &early_reg_closure,
                },
                VariableLocationInfo {
                    beginning: addr as u64 + 0x1c,
                    end: (addr + size) as u64 - 2 * 4,
                    start_expr: &frame_closure,
                },
                VariableLocationInfo {
                    beginning: (addr + size) as u64 - 2 * 4,
                    end: (addr + size) as u64,
                    start_expr: &early_reg_closure,
                },
            ];
            if !parsed.is_empty() {
                for subprogram_id in subprogram_ids.iter().copied() {
                    self.add_arguments(
                        subprogram_id,
                        &locations,
                        bi_one(),
                        bi_zero(),
                        parsed[0].clone(),
                    );
                }
            }
        }

        eprintln!("function {name}");
        Some(name)
    }

    fn write_section(
        &self,
        name: &str,
        section: &dyn Section<DwarfSectionWriter>,
        instrs: &mut Vec<Instr>,
    ) {
        instrs.push(Instr::Align4);
        instrs.push(Instr::Section(name.to_string()));
        instrs.push(Instr::Bytes(section.written.clone()));
    }

    fn write(&mut self, current_addr: usize, instrs: &mut Vec<Instr>) -> gimli::write::Result<()> {
        let mut unit = self.dwarf.units.get_mut(self.unit_id);
        let unit_ent = unit.get_mut(unit.root());
        unit_ent.set(
            DW_AT_high_pc,
            AttributeValue::Address(Address::Constant(
                self.target_addr as u64 + current_addr as u64,
            )),
        );

        let mut sections = Sections::<DwarfSectionWriter>::default();
        self.dwarf.write(&mut sections)?;
        self.frame_table
            .write_debug_frame(&mut sections.debug_frame)?;
        self.frame_table.write_eh_frame(&mut sections.eh_frame)?;

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
            Constant::Atom(lbl, _) => lbl.clone(),
            Constant::Cons(lbl, _, _) => lbl.clone(),
        }
    }
}

pub struct Program<T: DebugSExp = ConcreteSExp> {
    program: HashMap<String, Srcloc>,
    target_addr: u32,
    finished_insns: Vec<Instr>,
    first_label: String,
    env_label: String,
    encounters_of_code: HashMap<Vec<u8>, usize>,
    labels_by_hash: HashMap<Vec<u8>, String>,
    done_programs: HashSet<String>,
    waiting_programs: Vec<(String, T)>,
    constants: HashMap<Vec<u8>, Constant>,
    symbol_table: Rc<HashMap<String, String>>,
    current_symbol: Option<String>,
    current_symbol_name: Option<String>,
    function_symbols: HashMap<String, String>,
    renamed_symbols: HashMap<String, String>,
    start_addr: usize,
    current_addr: usize,
    dwarf_builder: DwarfBuilder,
}

impl<T: DebugSExp> fmt::Display for Program<T> {
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

fn hexify(v: &[u8]) -> String {
    Bytes::new(Some(BytesFromType::Raw(v.to_vec()))).hex()
}

pub fn swi_print(register: usize, label: usize) -> usize {
    SWI_PRINT_EXPR | register << 4 | label << 8
}

impl<T: DebugSExp> Program<T> {
    fn get_renamed_function_label(&self, hash: &[u8]) -> Option<String> {
        let hash_string = hex::encode(hash);
        self.renamed_symbols.get(&hash_string).cloned()
    }

    fn label_is_taken(&self, label: &str) -> bool {
        self.labels_by_hash
            .values()
            .any(|existing| existing == label)
    }

    fn get_renamed_name_for_label(&self, label: &str) -> Option<String> {
        let hash = label.strip_prefix('_').and_then(|s| s.split('_').next())?;
        self.renamed_symbols.get(hash).cloned()
    }

    fn get_code_label(&mut self, hash: &[u8]) -> String {
        let n = if let Some(n) = self.encounters_of_code.get(hash).clone() {
            *n
        } else {
            0
        };

        self.encounters_of_code.insert(hash.to_vec(), n + 1);
        return format!("_{}_{n}", hexify(hash));
    }

    fn do_throw(&mut self, source_sexp: T, loc: &Srcloc, hash: &[u8]) {
        self.load_atom(source_sexp.clone(), loc, hash, hash);
        self.push(source_sexp.clone(), loc, Instr::Swi(SWI_PRINT_EXPR));
        self.push(source_sexp, loc, Instr::Swi(SWI_THROW));
    }

    fn add_sexp(&mut self, loc: &Srcloc, hash: &[u8], s: T) -> String {
        if let Some(lbl) = self.constants.get(hash) {
            return lbl.label();
        }

        match s.explode() {
            DebugSExpValue::Cons(_l, a, b) => {
                let a_hash = sha256tree(a.clone());
                let b_hash = sha256tree(b.clone());
                let a_label = self.add_sexp(loc, &a_hash, a);
                let b_label = self.add_sexp(loc, &b_hash, b);
                let label = format!("_{}", hexify(hash));
                self.constants.insert(
                    hash.to_vec(),
                    Constant::Cons(label.clone(), a_label.clone(), b_label.clone()),
                );
                label
            }
            _ => self.add_atom(
                hash,
                &s.atom_bytes()
                    .expect("non-cons debug sexp should atomize")
                    .1,
            ),
        }
    }

    fn load_sexp(&mut self, source_sexp: T, loc: &Srcloc, hash: &[u8], s: T) {
        let label = self.add_sexp(loc, hash, s);
        self.push(source_sexp, loc, Instr::Lea(Register::R(0), label));
    }

    fn first_rest(&mut self, source_sexp: T, loc: &Srcloc, hash: &[u8], lst: &[T], offset: i32) {
        if lst.len() != 1 {
            return self.do_throw(source_sexp, loc, hash);
        }

        let subexp = self.add(lst[0].clone());
        for i in &[
            Instr::Addi(Register::R(0), Register::R(7), 0),
            // Determine if the result is a cons.
            Instr::Bl(subexp),
            Instr::Cmpi(Register::R(0), 0),
            Instr::SwiEq(SWI_THROW),
            Instr::Ldr(Register::R(1), Register::R(0), 0),
            Instr::Andi(Register::R(1), Register::R(1), 1),
            Instr::Cmpi(Register::R(1), 1),
            Instr::SwiEq(SWI_THROW),
            Instr::Ldr(Register::R(0), Register::R(0), offset),
        ] {
            self.push(source_sexp.clone(), loc, i.clone());
        }
    }

    fn do_operator(
        &mut self,
        loc: &Srcloc,
        hash: &[u8],
        a: &[u8],
        b: T,
        treat_as_quoted: bool,
        source_sexp: T,
    ) {
        if treat_as_quoted {
            todo!();
        }

        if a == b"" {
            return self.do_throw(source_sexp, loc, hash);
        }

        // Quote is special.
        if a == &[1] {
            eprintln!("do_operator, quoted {b}");
            self.add(b.clone());
            let b_hash = sha256tree(b.clone());
            return self.load_sexp(source_sexp, loc, &b_hash, b);
        }

        // Every other operator must have a proper list following it.
        let lst = if let Some(lst) = b.proper_list() {
            lst
        } else {
            return self.do_throw(source_sexp, loc, hash);
        };

        if a == &[2] {
            // Apply operator
            if lst.len() != 2 {
                return self.do_throw(source_sexp, loc, hash);
            }

            let env_comp = self.add(lst[1].clone());
            for i in &[
                Instr::Addi(Register::R(0), Register::R(7), 0),
                Instr::Bl(env_comp),
                Instr::Addi(Register::R(4), Register::R(0), 0),
            ] {
                self.push(source_sexp.clone(), loc, i.clone());
            }

            if let Some(quoted_code) = dequote(lst[0].clone()) {
                // Short circuit by reading out the quoted code and running it.
                let code_comp = self.add(quoted_code.clone());

                for i in &[
                    Instr::Addi(Register::R(7), Register::R(4), 0),
                    Instr::Addi(Register::R(0), Register::R(7), 0),
                    Instr::Bl(code_comp),
                ] {
                    self.push(source_sexp.clone(), loc, i.clone());
                }
            } else {
                let code_comp = self.add(lst[0].clone());

                for i in &[
                    Instr::Addi(Register::R(0), Register::R(7), 0),
                    Instr::Bl(code_comp),
                    Instr::Addi(Register::R(7), Register::R(4), 0),
                    Instr::Swi(SWI_DISPATCH_NEW_CODE),
                    Instr::Bx(Register::R(1)),
                ] {
                    self.push(source_sexp.clone(), loc, i.clone());
                }
            }

            for i in &[
                // Reload the old env.
                Instr::Ldr(Register::R(7), Register::SP, 12),
            ] {
                self.push(source_sexp.clone(), loc, i.clone());
            }
            return;
        } else if a == &[3] {
            // If operator
            if lst.len() != 3 {
                return self.do_throw(source_sexp, loc, hash);
            }

            let else_clause = self.add(lst[2].clone());
            let then_clause = self.add(lst[1].clone());
            let cond_clause = self.add(lst[0].clone());

            for i in &[
                Instr::Addi(Register::R(0), Register::R(7), 0),
                Instr::Bl(else_clause),
                Instr::Addi(Register::R(6), Register::R(0), 0),
                Instr::Addi(Register::R(0), Register::R(7), 0),
                Instr::Bl(then_clause),
                Instr::Addi(Register::R(4), Register::R(0), 0),
                Instr::Addi(Register::R(0), Register::R(7), 0),
                Instr::Bl(cond_clause),
                Instr::Cmpi(Register::R(0), 0),
                Instr::AddiEq(Register::R(4), Register::R(6), 0),
                Instr::Ldr(Register::R(1), Register::R(0), 0),
                Instr::Cmpi(Register::R(1), 1),
                Instr::AddiEq(Register::R(4), Register::R(6), 0),
                Instr::Addi(Register::R(0), Register::R(4), 0),
            ] {
                self.push(source_sexp.clone(), loc, i.clone());
            }
            return;
        } else if a == &[4] {
            // Cons operator
            if lst.len() != 2 {
                return self.do_throw(source_sexp, loc, hash);
            }

            let rest_label = self.add(lst[1].clone());
            let first_label = self.add(lst[0].clone());

            for i in &[
                Instr::Addi(Register::R(0), Register::R(7), 0),
                Instr::Bl(rest_label),
                Instr::Addi(Register::R(4), Register::R(0), 0),
                Instr::Addi(Register::R(0), Register::R(7), 0),
                Instr::Bl(first_label),
                // R1 = next allocated address.
                Instr::Ldr(Register::R(1), Register::R(5), NEXT_ALLOC_OFFSET),
                // R2 = R1 + 8 (size of cons)
                Instr::Addi(Register::R(2), Register::R(1), 8),
                Instr::Str(Register::R(2), Register::R(5), NEXT_ALLOC_OFFSET),
                // Build cons
                Instr::Str(Register::R(0), Register::R(1), 0),
                Instr::Str(Register::R(4), Register::R(1), 4),
                // Move the result to r0
                Instr::Addi(Register::R(0), Register::R(1), 0),
            ] {
                self.push(source_sexp.clone(), loc, i.clone());
            }
            return;
        } else if a == &[5] {
            return self.first_rest(source_sexp, loc, hash, &lst, 0);
        } else if a == &[6] {
            return self.first_rest(source_sexp, loc, hash, &lst, 4);
        } else {
            // Ensure we have this sexp loadable as data.
            let operator_sexp = T::atom(loc.clone(), a);
            let atom_hash = sha256tree(operator_sexp.clone());
            let label = self.add_atom(&atom_hash, a);
            eprintln!("load {label} for general operator {operator_sexp}\n");

            // Load a nil into R4.
            for i in &[Instr::Andi(Register::R(4), Register::R(4), 0)] {
                self.push(source_sexp.clone(), loc, i.clone());
            }

            // For each subexpression, call it and replace R4 with (cons R0 R4)
            for item in lst.iter().rev() {
                eprintln!("load clause {item} for operator {operator_sexp}");
                let clause_label = self.add(item.clone());
                for i in &[
                    // Load the allocator ptr into R0.
                    Instr::Ldr(Register::R(0), Register::R(5), NEXT_ALLOC_OFFSET),
                    // Allocate a cons (new addr in R6)
                    Instr::Addi(Register::R(6), Register::R(0), 8),
                    // Store back the pointer.
                    Instr::Str(Register::R(6), Register::R(5), NEXT_ALLOC_OFFSET),
                    // R6 = alloc ptr
                    Instr::Addi(Register::R(6), Register::R(0), 0),
                    // Reload R0 with the env ptr.
                    Instr::Addi(Register::R(0), Register::R(7), 0),
                    // Call the arg code
                    Instr::Bl(clause_label),
                    // Store R0 into the cons.
                    Instr::Str(Register::R(0), Register::R(6), 0),
                    // Store R4 into the cons.
                    Instr::Str(Register::R(4), Register::R(6), 4),
                    // Replace R4 with R6 (the new cons)
                    Instr::Addi(Register::R(4), Register::R(6), 0),
                ] {
                    self.push(source_sexp.clone(), loc, i.clone());
                }
            }

            for i in &[
                // Load the sexp for the operator into R0
                Instr::Lea(Register::R(0), label),
                // Set R1 to the tail exp.
                Instr::Addi(Register::R(1), Register::R(4), 0),
                // Call to do the operator.
                Instr::Swi(SWI_DISPATCH_INSTRUCTION),
            ] {
                self.push(source_sexp.clone(), loc, i.clone());
            }
        }
    }

    // R0 = the address of the env block.
    fn env_select(&mut self, source_sexp: T, loc: &Srcloc, hash: &[u8], v: &[u8]) {
        if v.is_empty() {
            self.load_atom(source_sexp, loc, hash, v);
            return;
        }

        // Let r0 be our pointer.
        self.push(
            source_sexp.clone(),
            loc,
            Instr::Addi(Register::R(0), Register::R(7), 0),
        );

        // Whole env ref.
        if v == &[1] {
            return;
        }

        for (i, byt) in v.iter().enumerate().rev() {
            for bit in 0..8 {
                let remaining = byt >> bit;
                if remaining == 1 && i == 0 {
                    // We have the right value.
                    return;
                } else {
                    let offset = ((remaining & 1) * 4) as i32;

                    for i in &[
                        // Check for a cons.
                        Instr::Cmpi(Register::R(0), 0),
                        Instr::SwiEq(SWI_THROW),
                        Instr::Ldr(Register::R(1), Register::R(0), 0),
                        Instr::Andi(Register::R(1), Register::R(1), 1),
                        Instr::Cmpi(Register::R(1), 1),
                        // Break if it was an atom.
                        Instr::SwiEq(SWI_THROW),
                        // Load if it was a cons.
                        Instr::Ldr(Register::R(0), Register::R(0), offset),
                    ] {
                        self.push(source_sexp.clone(), loc, i.clone());
                    }
                }
            }
        }
    }

    fn add_atom(&mut self, hash: &[u8], v: &[u8]) -> String {
        if let Some(lbl) = self.constants.get(hash) {
            return lbl.label();
        }

        let label = format!("_{}", hexify(hash));
        self.constants
            .insert(hash.to_vec(), Constant::Atom(label.clone(), v.to_vec()));
        label
    }

    fn load_atom(&mut self, source_sexp: T, loc: &Srcloc, hash: &[u8], v: &[u8]) {
        let label = self.add_atom(hash, v);
        self.push(source_sexp, loc, Instr::Lea(Register::R(0), label));
    }

    fn add(&mut self, sexp: T) -> String {
        let hash = sha256tree(sexp.clone());
        if let Some(existing_label) = self.labels_by_hash.get(&hash) {
            return existing_label.clone();
        }

        // Note: get_code_label issues a fresh label for this hash every time.
        let generated_body_label = self.get_code_label(&hash);
        let body_label = self
            .get_renamed_function_label(&hash)
            .filter(|label| !self.label_is_taken(label))
            .unwrap_or(generated_body_label);
        eprintln!("label {body_label} for {sexp} at {}", sexp.loc());

        self.labels_by_hash.insert(hash, body_label.clone());
        self.waiting_programs
            .push((body_label.clone(), sexp.clone()));
        body_label
    }

    fn push_be(
        &mut self,
        source_sexp: T,
        srcloc: &Srcloc,
        instr: Instr,
        begin_end_block: Option<BeginEndBlock>,
    ) {
        let size = instr.size(self.current_addr);

        let insert_instr = if let Instr::Globl(g) = &instr {
            eprintln!("instr {instr:?}");
            // Two things: ensure we switch to real function names when we
            // have them.
            //
            // Ensure we set the current symbol.
            self.current_symbol = Some(g.clone());
            self.current_symbol_name = self.get_renamed_name_for_label(g);

            instr
        } else {
            instr
        };

        self.finished_insns.push(insert_instr.clone());
        let start_block = matches!(begin_end_block, Some(BeginEndBlock::BeginBlock));
        let end_block = matches!(begin_end_block, Some(BeginEndBlock::EndBlock));

        if start_block {
            self.current_addr = (self.current_addr + 15) & !15;
            self.start_addr = self.current_addr;
        }

        if end_block {
            self.current_addr = (self.current_addr + 15) & !15;
            if let Some(label) = self.current_symbol.as_ref() {
                eprintln!(
                    "end block for label {label} {:x}-{:x}",
                    self.start_addr, self.current_addr
                );
                let preferred_name = self.current_symbol_name.clone();
                if let Some(function_name) = self.dwarf_builder.decorate_function(
                    label,
                    self.start_addr,
                    self.current_addr - self.start_addr,
                    preferred_name.as_deref(),
                ) {
                    eprintln!("end block with function {function_name}");
                    if label == &function_name || !self.label_is_taken(&function_name) {
                        self.function_symbols.insert(label.clone(), function_name);
                    }
                }
                self.current_symbol = None;
                self.current_symbol_name = None;
            }
        }

        if size != 0 {
            let next_addr = self.current_addr + size;
            self.dwarf_builder.add_instr(
                self.current_addr,
                srcloc,
                &source_sexp,
                insert_instr,
                begin_end_block,
            );
            self.current_addr = next_addr;
        }
    }

    fn push(&mut self, source_sexp: T, srcloc: &Srcloc, instr: Instr) {
        self.push_be(source_sexp, srcloc, instr, None);
    }

    fn emit_waiting(&mut self) {
        while let Some((label, sexp)) = self.waiting_programs.pop() {
            eprintln!("{} sexp {:?} {}", label, sexp.loc(), sexp);
            let hash = sha256tree(sexp.clone());

            self.labels_by_hash.insert(hash.clone(), label.clone());
            self.dwarf_builder.start(self.current_addr);

            self.push(sexp.clone(), &sexp.loc(), Instr::Globl(label.clone()));
            self.push(sexp.clone(), &sexp.loc(), Instr::Label(label.clone()));
            self.push_be(
                sexp.clone(),
                &sexp.loc(),
                Instr::Push(vec![Register::FP, Register::LR]),
                Some(BeginEndBlock::BeginBlock),
            );
            for i in &[
                Instr::Addi(Register::FP, Register::SP, 4),
                Instr::Subi(Register::SP, Register::SP, 0x18),
                Instr::Str(Register::R(4), Register::SP, 0),
                Instr::Str(Register::R(5), Register::SP, 4),
                Instr::Str(Register::R(6), Register::SP, 8),
                Instr::Str(Register::R(7), Register::SP, 12),
                Instr::Addi(Register::R(7), Register::R(0), 0),
            ] {
                self.push(sexp.clone(), &sexp.loc(), i.clone());
            }

            // Translate body.
            match sexp.explode() {
                DebugSExpValue::Cons(l, a, b) => {
                    if let Some((loc, a)) = is_atom(a.clone()) {
                        // do quoted operator
                        self.do_operator(&loc, &hash, &a, b.clone(), false, sexp.clone());
                    } else if let Some((loc, a)) = is_wrapped_atom(a.clone()) {
                        // do unquoted operator
                        self.do_operator(&loc, &hash, &a, b.clone(), true, sexp.clone());
                    } else {
                        // invalid head form, just throw.
                        self.do_throw(sexp.clone(), &l, &hash);
                    }
                }
                DebugSExpValue::Nil(l) => self.load_atom(sexp.clone(), &l, &hash, &[]),
                DebugSExpValue::Atom(l, v) => {
                    if v.is_empty() {
                        return self.load_atom(sexp.clone(), &l, &hash, &[]);
                    }
                    self.env_select(sexp.clone(), &l, &hash, &v);
                }
                DebugSExpValue::QuotedString(l, _, v) => {
                    if v.is_empty() {
                        return self.load_atom(sexp.clone(), &l, &hash, &[]);
                    }
                    self.env_select(sexp.clone(), &l, &hash, &v);
                }
                DebugSExpValue::Integer(_, _) => {
                    let (l, v) = sexp
                        .atom_bytes()
                        .expect("non-cons debug sexp should atomize");
                    if v.is_empty() {
                        return self.load_atom(sexp.clone(), &l, &hash, &[]);
                    }
                    self.env_select(sexp.clone(), &l, &hash, &v);
                }
            }

            for i in &[
                Instr::Ldr(Register::R(4), Register::SP, 0),
                Instr::Ldr(Register::R(5), Register::SP, 4),
                Instr::Ldr(Register::R(6), Register::SP, 8),
                Instr::Ldr(Register::R(7), Register::SP, 12),
                Instr::Subi(Register::SP, Register::FP, 4),
                Instr::Pop(vec![Register::FP, Register::LR]),
            ] {
                self.push(sexp.clone(), &sexp.loc(), i.clone());
            }

            self.push_be(
                sexp.clone(),
                &sexp.loc(),
                Instr::Bx(Register::LR),
                Some(BeginEndBlock::EndBlock),
            );
            self.dwarf_builder.end(self.current_addr);
        }
    }

    fn start_insns(&mut self) {
        let srcloc = debug_start_loc("*prolog*");
        let source_sexp = T::atom(srcloc.clone(), b"prolog");
        for i in &[
            Instr::Section(".text".to_string()),
            Instr::Align4,
            Instr::Globl("_start".to_string()),
            Instr::Label("_start".to_string()),
            Instr::Lea(Register::R(5), "_run".to_string()),
            Instr::Ldr(Register::R(7), Register::R(5), 4),
            Instr::Addi(Register::R(0), Register::R(7), 0),
            Instr::Bl(self.first_label.clone()),
            // Print the last value.
            Instr::Swi(SWI_PRINT_EXPR),
            Instr::Swi(SWI_DONE),
        ] {
            self.push(source_sexp.clone(), &srcloc, i.clone());
        }
    }

    fn finish_insns(&mut self) -> Result<(), String> {
        let srcloc = debug_start_loc("*epilog*");
        let source_sexp = T::atom(srcloc.clone(), b"epilog");
        let mut constants = HashMap::new();
        swap(&mut constants, &mut self.constants);

        for i in [
            Instr::Align4,
            Instr::Globl("_run".to_string()),
            Instr::Label("_run".to_string()),
            Instr::Long(0x10000000),
            Instr::Addr(self.env_label.clone(), true),
            // Write the constant data.
            Instr::Align4,
            Instr::Section(".data".to_string()),
        ]
        .iter()
        {
            self.push(source_sexp.clone(), &srcloc, i.clone());
        }

        for (_, c) in constants.iter() {
            match c {
                Constant::Cons(label, a_label, b_label) => {
                    eprintln!("constant pair {label}");
                    for i in &[
                        Instr::Align4,
                        Instr::Globl(label.clone()),
                        Instr::Label(label.clone()),
                        Instr::Addr(a_label.clone(), false),
                        Instr::Addr(b_label.clone(), false),
                    ] {
                        self.push(source_sexp.clone(), &srcloc, i.clone());
                    }
                }
                Constant::Atom(label, bytes) => {
                    for i in &[
                        Instr::Align4,
                        Instr::Globl(label.clone()),
                        Instr::Label(label.clone()),
                        Instr::Long(bytes.len() * 2 + 1),
                        Instr::Bytes(bytes.clone()),
                    ] {
                        self.push(source_sexp.clone(), &srcloc, i.clone());
                    }
                }
            }
        }
        swap(&mut constants, &mut self.constants);

        // Export remapped function names by hash so the emulator can resolve
        // a tree hash to a renamed symbol after loading the ELF.
        let mut renamed_symbols: Vec<_> = self
            .renamed_symbols
            .iter()
            .map(|(hash, target_name)| (hash.clone(), target_name.clone()))
            .collect();
        renamed_symbols.sort_by(|a, b| a.0.cmp(&b.0));
        for (hash, target_name) in renamed_symbols.into_iter() {
            let mut target_name_bytes = target_name.as_bytes().to_vec();
            target_name_bytes.push(0);
            let remap_label = format!("_$_{hash}");
            for i in &[
                Instr::Align4,
                Instr::Globl(remap_label.clone()),
                Instr::Label(remap_label),
                Instr::Bytes(target_name_bytes.clone()),
            ] {
                self.push(source_sexp.clone(), &srcloc, i.clone());
            }
        }

        self.dwarf_builder
            .write(self.current_addr, &mut self.finished_insns)
            .unwrap();

        Ok(())
    }

    pub fn to_elf(&self, output: &str) -> Result<ElfObject, String> {
        let synthetic_source = self.dwarf_builder.synthetic_source();
        let mut sections = Vec::new();
        let mut obj = ArtifactBuilder::new(triple!("arm-unknown-unknown-unknown-elf"))
            .name(output.to_owned())
            .finish();
        // Collect declarations
        let mut waiting_for_debug_info = None;
        let mut data_section = false;
        let mut data = "".to_string();

        let mut decls: Vec<(String, Decl)> = self
            .finished_insns
            .iter()
            .filter_map(|i| {
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
                        return Some((name.to_string(), Decl::function().global().into()));
                    };
                } else if let Instr::Bytes(b) = i {
                    // Define section in the faerie way.
                    if let Some(waiting) = waiting_for_debug_info.clone() {
                        waiting_for_debug_info = None;
                        sections.push((waiting, b.clone()));
                    }
                }

                None
            })
            .collect();

        // Declare .debug_aranges
        decls.push((
            ".debug_aranges".to_string(),
            Decl::section(SectionKind::Debug).into(),
        ));

        obj.declarations(decls.into_iter())
            .map_err(|e| format!("{e:?}"))?;

        let mut relocations = Vec::new();
        let mut function_body = Vec::new();
        let mut in_function = None;

        let mut produced_code = 0;
        let mut handle_def_end = |target_addr: u32,
                                  function_body: &mut Vec<u8>,
                                  in_function: &mut Option<String>|
         -> Result<(), String> {
            if let Some(defname) = in_function.as_ref() {
                if !function_body.is_empty() {
                    eprintln!("obj define {defname}");
                    produced_code += function_body.len();
                    obj.define(defname, function_body.clone())
                        .map_err(|e| format!("{e:?}"))?;
                    *function_body = Vec::new();
                }
            }

            Ok(())
        };

        for i in self.finished_insns.iter() {
            if let Instr::Globl(name) = i {
                handle_def_end(self.target_addr, &mut function_body, &mut in_function)?;
                in_function = Some(name.to_string());
            }

            if let Some(f) = in_function.as_ref() {
                i.encode(&mut function_body, &mut relocations, &f);
            }
        }

        handle_def_end(self.target_addr, &mut function_body, &mut in_function)?;
        // Create .debug_aranges
        let mut debug_aranges: Vec<u8> = (0..0x20).map(|_| 0).collect();
        write_u32(&mut debug_aranges, 0, 0x1c);
        debug_aranges[4] = 2;
        write_u32(&mut debug_aranges, 6, 0);
        debug_aranges[10] = 4;
        write_u32(&mut debug_aranges, 16, self.target_addr);
        eprintln!(
            "produced_code {produced_code} target_addr {}",
            self.target_addr
        );
        write_u32(&mut debug_aranges, 20, (produced_code as u32));
        sections.push((".debug_aranges".to_string(), debug_aranges));

        for (name, bytes) in sections.iter() {
            obj.define(name, bytes.clone())
                .map_err(|e| format!("{e:?}"))?;
        }

        for r in relocations.iter() {
            obj.link(Link {
                from: &r.function,
                to: &r.reloc_target,
                at: r.code_location as u64,
            })
            .map_err(|e| format!("link {e:?}"))?;
        }

        let mut result_buf = obj.emit().map_err(|e| format!("obj emit {e:?}"))?;

        // Patch up
        eprintln!("reload elf");
        let create_patches = |result_buf: &mut [u8]| {
            let elf_loader = ElfLoader::new(result_buf, self.target_addr).expect("should load");
            elf_loader.patch_sections()
        };

        eprintln!("create patches");
        let patches = create_patches(&mut result_buf);
        eprintln!("patches made");

        for (i, (target, value)) in patches.into_iter().enumerate() {
            eprintln!("section {i} target {target:x} value {value:x}");
            write_u32(&mut result_buf, target, value);
        }

        eprintln!("code succeeded");
        Ok(ElfObject {
            object_file: result_buf,
            synthetic_source,
        })
    }

    pub fn new(
        program: HashMap<String, Srcloc>,
        filename: &str,
        elf_output: &str,
        sexp: T,
        env: T,
        target_addr: u32,
        symbol_table: Rc<HashMap<String, String>>,
    ) -> Result<Self, String> {
        let remap_hashes: Vec<_> = symbol_table
            .iter()
            .filter_map(|(hash, name)| {
                if name.contains(':') || name.contains('_') || name == "source_file" {
                    return None;
                }
                if !program.contains_key(name) {
                    return None;
                }

                if let Ok(byte_hash) = hex::decode(hash) {
                    let all_matches = find_all_by_hash(&byte_hash, sexp.clone());
                    let selected = if let Some(target_loc) = program.get(name) {
                        all_matches
                            .iter()
                            .find(|matched| target_loc.overlap(&matched.loc()))
                            .cloned()
                    } else {
                        None
                    }
                    .or_else(|| all_matches.first().cloned());
                    if let Some(selected) = selected {
                        return Some((byte_hash, name.clone(), selected));
                    }
                }
                None
            })
            .collect();

        let dwarf_builder =
            DwarfBuilder::new(filename, elf_output, target_addr, symbol_table.clone());
        let mut p: Program<T> = Program {
            program,
            finished_insns: Vec::new(),
            first_label: Default::default(),
            env_label: Default::default(),
            encounters_of_code: Default::default(),
            labels_by_hash: Default::default(),
            done_programs: Default::default(),
            waiting_programs: Default::default(),
            constants: Default::default(),
            symbol_table: Default::default(),
            function_symbols: Default::default(),
            renamed_symbols: Default::default(),
            current_addr: 0,
            start_addr: 0,
            target_addr,
            current_symbol: None,
            current_symbol_name: None,
            dwarf_builder,
        };

        p.symbol_table = symbol_table;
        let loc = debug_start_loc("*env*");
        let envhash = sha256tree(env.clone());
        for (remap_hash, name, remap_sexp) in remap_hashes.into_iter() {
            let remap_hash_hex = hex::encode(&remap_hash);
            eprintln!("should remap hash {} to {name}", remap_hash_hex);
            p.renamed_symbols
                .insert(remap_hash_hex.clone(), name.clone());
            p.add_sexp(&remap_sexp.loc(), &remap_hash, remap_sexp.clone());
            let remap_label = p.add(remap_sexp);
            eprintln!("remap selected label {remap_label} for {name}");
        }
        p.first_label = p.add(sexp.clone());
        p.start_insns();
        p.env_label = p.add_sexp(&loc, &envhash, env);
        p.emit_waiting();
        p.finish_insns()?;
        Ok(p)
    }
}
