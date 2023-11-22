use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::fs;
use std::mem::swap;
use std::rc::Rc;

use gimli::write::Dwarf;

use clvm_tools_rs::classic::clvm::casts::bigint_to_bytes_clvm;
use clvm_tools_rs::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use clvm_tools_rs::compiler::clvm::{sha256tree, truthy};
use clvm_tools_rs::compiler::sexp::{Atom, NodeSel, SelectNode, SExp, ThisNode, parse_sexp};
use clvm_tools_rs::compiler::srcloc::Srcloc;

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

#[derive(Clone, Debug)]
enum Instr {
    Align4,
    Text,
    Data,
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
    Swi(usize),
    SwiEq(usize),
    Cmpi(Register,usize),
    Long(usize),
    Addr(String),
    Bytes(Vec<u8>),
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Instr::Align4 => write!(f, "  .align 4"),
            Instr::Text => write!(f, "  .text"),
            Instr::Data => write!(f, "  .data"),
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

#[derive(Debug, Clone)]
struct SrclocAndInstr {
    loc: Srcloc,
    instr: Instr,
}

impl fmt::Display for SrclocAndInstr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.instr)
    }
}

impl SrclocAndInstr {
    fn new(loc: Srcloc, instr: Instr) -> Self {
        SrclocAndInstr { loc, instr }
    }
}

#[derive(Default)]
struct DwarfBuilder {
    dwarf: Dwarf
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

#[derive(Default)]
struct Program {
    finished_insns: Vec<SrclocAndInstr>,
    first_label: String,
    encounters_of_code: HashMap<Vec<u8>, usize>,
    waiting_programs: HashMap<String, Rc<SExp>>,
    constants: HashMap<Vec<u8>, Constant>,
    dwarf_builder: DwarfBuilder,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let write_vec = |f: &mut Formatter, v: &[SrclocAndInstr]| -> fmt::Result {
            for i in v.iter() {
                write!(f, "{i}\n")?;
            }
            Ok(())
        };

        writeln!(f, "{}", Instr::Text);
        writeln!(f, "{}", Instr::Align4);
        writeln!(f, "{}", Instr::Globl("_start".to_string()));
        writeln!(f, "{}", Instr::Label("_start".to_string()));
        writeln!(f, "{}", Instr::Adr(Register::R(0), "_run".to_string()));
        writeln!(f, "{}", Instr::Ldr(Register::SP, Register::R(0), 8));
        writeln!(f, "{}", Instr::B(self.first_label.clone()));

        write_vec(f, &self.finished_insns)?;

        writeln!(f, "{}", Instr::Align4);
        writeln!(f, "{}", Instr::Globl("_run".to_string()));
        writeln!(f, "{}", Instr::Label("_run".to_string()));
        writeln!(f, "{}", Instr::Long(0x1000000));
        writeln!(f, "{}", Instr::Addr("_4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a".to_string()));
        writeln!(f, "{}", Instr::Long(0x7ff0));

        writeln!(f, "{}", Instr::Align4);
        for (k, c) in self.constants.iter() {
            match c {
                Constant::Atom(label, bytes) => {
                    writeln!(f, "{}", Instr::Align4);
                    writeln!(f, "{}", Instr::Globl(label.clone()));
                    writeln!(f, "{}", Instr::Label(label.clone()));
                    writeln!(f, "{}", Instr::Long(bytes.len() * 2 + 1));
                    writeln!(f, "{}", Instr::Bytes(bytes.clone()));
                }
                Constant::Cons(label, a_label, b_label) => {
                    writeln!(f, "{}", Instr::Align4);
                    writeln!(f, "{}", Instr::Globl(label.clone()));
                    writeln!(f, "{}", Instr::Label(label.clone()));
                    writeln!(f, "{}", Instr::Addr(a_label.clone()));
                    writeln!(f, "{}", Instr::Addr(b_label.clone()));
                }
            }
        }

        Ok(())
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
        self.finished_insns.push(SrclocAndInstr::new(loc.clone(), Instr::Swi(0)));
    }

    fn add_sexp(&mut self, loc: &Srcloc, hash: &[u8], s: Rc<SExp>) -> String {
        if let Some(lbl) = self.constants.get(hash) {
            return lbl.label();
        }

        match s.borrow() {
            SExp::Cons(l,a,b) => {
                let a_hash = sha256tree(a.clone());
                let b_hash = sha256tree(b.clone());
                let a_label = self.add_sexp(loc, &a_hash, a.clone());
                let b_label = self.add_sexp(loc, &b_hash, b.clone());
                let label = format!("_{}", hexify(hash));
                self.constants.insert(hash.to_vec(), Constant::Cons(label.clone(), a_label.clone(), b_label.clone()));
                label

            },
            SExp::Nil(l) => self.add_atom(l, hash, &[]),
            SExp::Atom(l, a) => self.add_atom(l, hash, &a),
            SExp::QuotedString(l, _, a) => self.add_atom(l, hash, &a),
            SExp::Integer(l, i) => {
                let v = bigint_to_bytes_clvm(&i).data().clone();
                self.add_atom(l, hash, &v)
            }
        }
    }

    fn load_sexp(&mut self, loc: &Srcloc, hash: &[u8], s: Rc<SExp>) {
        let label = self.add_sexp(loc, hash, s);
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::Adr(Register::R(0), label)
        ));
    }

    fn first_rest(&mut self, loc: &Srcloc, hash: &[u8], lst: &[SExp], offset: i32) {
        if lst.len() != 1 {
            return self.do_throw(loc, hash);
        }

        let subexp = self.add(Rc::new(lst[0].clone()));
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::Addi(Register::R(0), Register::R(5), 0)
        ));
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::Bl(subexp)
        ));
        // Determine if the result is a cons.
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::Andi(Register::R(1), Register::R(0), 1)
        ));
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::Cmpi(Register::R(1), 1)
        ));
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::SwiEq(0)
        ));
        self.finished_insns.push(SrclocAndInstr::new(
            loc.clone(),
            Instr::Ldr(Register::R(0), Register::R(0), offset)
        ));
    }

    fn do_operator(&mut self, loc: &Srcloc, hash: &[u8], a: &[u8], b: Rc<SExp>, quoted: bool) {
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

            todo!();
        } else if a == &[3] {
            // If operator
            if lst.len() != 3 {
                return self.do_throw(loc, hash);
            }

            let else_clause = self.add(Rc::new(lst[2].clone()));
            let then_clause = self.add(Rc::new(lst[1].clone()));
            let cond_clause = self.add(Rc::new(lst[0].clone()));

            let else_label = self.get_code_label(hash);

            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Bl(cond_clause)
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Ldr(Register::R(1), Register::R(0), 0)
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Cmpi(Register::R(1), 1)
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::BEq(else_label.clone())
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Bl(then_clause),
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Label(else_label),
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Bl(else_clause),
            ));
            return;
        } else if a == &[4] {
            // Cons operator
            if lst.len() != 2 {
                return self.do_throw(loc, hash);
            }

            let rest_label = self.add(Rc::new(lst[1].clone()));
            let first_label = self.add(Rc::new(lst[0].clone()));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Addi(Register::R(0), Register::R(5), 0),
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Bl(rest_label),
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Addi(Register::R(4), Register::R(0), 0),
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Addi(Register::R(0), Register::R(5), 0),
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Bl(first_label),
            ));
            // R1 = next allocated address.
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Ldr(Register::R(1), Register::R(5), 0)
            ));
            // R2 = R1 + 8 (size of cons)
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Addi(Register::R(2), Register::R(1), 8)
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Str(Register::R(2), Register::R(5), 0)
            ));
            // Build cons
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Str(Register::R(0), Register::R(1), 0)
            ));
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Str(Register::R(4), Register::R(1), 4)
            ));
            // Move the result to r0
            self.finished_insns.push(SrclocAndInstr::new(
                loc.clone(),
                Instr::Addi(Register::R(0), Register::R(1), 0)
            ));
            return;
        } else if a == &[5] {
            return self.first_rest(loc, hash, &lst, 0);
        } else if a == &[6] {
            return self.first_rest(loc, hash, &lst, 4);
        }

        // Generic operator emulation.
        todo!("generic op {a:?}");
    }

    // R0 = the address of the target item in the environment.  If it's not
    // reachable, this results in a throw.
    fn env_select(&mut self, loc: &Srcloc, hash: &[u8], v: &[u8]) {
        todo!();
    }

    fn add_atom(&mut self, loc: &Srcloc, hash: &[u8], v: &[u8]) -> String {
        if let Some(lbl) = self.constants.get(hash) {
            return lbl.label();
        }

        let label = format!("_{}", hexify(hash));
        self.constants.insert(hash.to_vec(), Constant::Atom(label.clone(), v.to_vec()));
        label
    }

    fn load_atom(&mut self, loc: &Srcloc, hash: &[u8], v: &[u8]) {
        let s = hexify(hash);
        let label = self.add_atom(loc, hash, v);
        self.finished_insns.push(SrclocAndInstr::new(loc.clone(), Instr::Adr(Register::R(0), label)));
    }

    fn add(&mut self, sexp: Rc<SExp>) -> String {
        let hash = sha256tree(sexp.clone());

        // Note: get_code_label issues a fresh label for this hash every time.
        let body_label = self.get_code_label(&hash);
        self.waiting_programs.insert(body_label.clone(), sexp.clone());
        body_label
    }

    fn emit_waiting(&mut self) {
        let mut current_waiting = HashMap::new();
        swap(&mut current_waiting, &mut self.waiting_programs);

        while !current_waiting.is_empty() {
            for (label, sexp) in current_waiting.iter() {
                let hash = sha256tree(sexp.clone());

                self.finished_insns.push(SrclocAndInstr::new(
                    sexp.loc(),
                    Instr::Globl(label.clone())
                ));
                self.finished_insns.push(SrclocAndInstr::new(
                    sexp.loc(),
                    Instr::Label(label.clone())
                ));
                self.finished_insns.push(SrclocAndInstr::new(
                    sexp.loc(),
                    Instr::Push(vec![Register::LR, Register::R(4), Register::R(5)])
                ));

                // Grab the env pointer.
                self.finished_insns.push(SrclocAndInstr::new(
                    sexp.loc(),
                    Instr::Addi(Register::R(5), Register::R(0), 0),
                ));

                // Translate body.
                match sexp.borrow() {
                    SExp::Cons(l,a,b) => {
                        if let Some((loc, a)) = is_atom(a.clone()) {
                            // do quoted operator
                            self.do_operator(&loc, &hash, &a, b.clone(), true);
                        } else if let Some((loc, a)) = is_wrapped_atom(a.clone()) {
                            // do unquoted operator
                            self.do_operator(&loc, &hash, &a, b.clone(), false);
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

                self.finished_insns.push(SrclocAndInstr::new(
                    sexp.loc(),
                    Instr::Pop(vec![Register::LR, Register::R(4), Register::R(5)]),
                ));
                self.finished_insns.push(SrclocAndInstr::new(
                    sexp.loc(),
                    Instr::Bx(Register::LR),
                ));
            }

            current_waiting.clear();
            swap(&mut current_waiting, &mut self.waiting_programs);
        }
    }

    fn new(sexp: Rc<SExp>) -> Self {
        let mut p: Program = Program::default();
        p.first_label = p.add(sexp);
        p.emit_waiting();
        p
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("must provide programs to convert");
        return;
    }

    let argfile =
        if let Ok(res) = fs::read(&args[0]) {
            res
        } else {
            panic!("error reading {}", args[0]);
        };

    let srcloc = Srcloc::start(&args[0]);
    let parsed =
        if let Ok(res) = parse_sexp(srcloc, argfile.iter().copied()) {
            res
        } else {
            panic!("error parsing {}", args[0]);
        };

    let program = Program::new(parsed[0].clone());
    println!("{program}");
}
