use num_bigint::ToBigInt;
use rand::Rng;
use std::env;
use std::rc::Rc;

use clvmr::allocator::Allocator;
use subprocess::Exec;

use crate::classic::clvm::__type_compatibility__::Stream;
use crate::classic::clvm::serialize::sexp_to_stream;
use crate::classic::clvm_tools::binutils::assemble;
use crate::compiler::fuzz::{FuzzGenerator, FuzzTypeParams, Rule};
use crate::compiler::sexp::{self, enlist, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::tests::classic::run::do_basic_run;
use crate::tests::compiler::fuzz::simple_seeded_rng;

const GENERATED_PROGRAMS_TO_COMPARE: u32 = 100;
const MAX_EXPANSIONS_BEFORE_TERMINATING: usize = 28;
const MAX_EXPANSIONS_TOTAL: usize = 160;
const MAX_HELPERS: usize = 6;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Scope {
    depth: usize,
    vars: Vec<String>,
    funcs: Vec<String>,
    consts: Vec<String>,
}

impl Scope {
    fn child(&self) -> Self {
        let mut next = self.clone();
        next.depth = next.depth.saturating_sub(1);
        next
    }

    fn with_var(&self, name: String) -> Self {
        let mut next = self.clone();
        next.vars.push(name);
        next
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Cl26Tag {
    Start,
    Program(Scope),
    Tail(Scope),
    OpArgs(Scope, usize),
    AssignTail(Scope, usize),
    LetBindings(Scope, Vec<String>),
}

fn encode_list(items: &[String]) -> String {
    items.join(",")
}

fn decode_list(items: &str) -> Vec<String> {
    if items.is_empty() {
        Vec::new()
    } else {
        items.split(',').map(|item| item.to_string()).collect()
    }
}

fn encode_scope(scope: &Scope) -> String {
    format!(
        "{}|{}|{}|{}",
        scope.depth,
        encode_list(&scope.vars),
        encode_list(&scope.funcs),
        encode_list(&scope.consts)
    )
}

fn decode_scope(parts: &[&str]) -> Option<Scope> {
    if parts.len() < 4 {
        return None;
    }

    Some(Scope {
        depth: parts[0].parse().ok()?,
        vars: decode_list(parts[1]),
        funcs: decode_list(parts[2]),
        consts: decode_list(parts[3]),
    })
}

fn encode_tag(tag: &Cl26Tag) -> String {
    match tag {
        Cl26Tag::Start => "start".to_string(),
        Cl26Tag::Program(scope) => format!("program|{}", encode_scope(scope)),
        Cl26Tag::Tail(scope) => format!("tail|{}", encode_scope(scope)),
        Cl26Tag::OpArgs(scope, remaining) => {
            format!("op-args|{}|{}", encode_scope(scope), remaining)
        }
        Cl26Tag::AssignTail(scope, remaining) => {
            format!("assign-tail|{}|{}", encode_scope(scope), remaining)
        }
        Cl26Tag::LetBindings(scope, names) => {
            format!(
                "let-bindings|{}|{}",
                encode_scope(scope),
                encode_list(names)
            )
        }
    }
}

fn decode_tag(tag: &[u8]) -> Option<Cl26Tag> {
    let tag = std::str::from_utf8(tag).ok()?;
    if tag == "start" {
        return Some(Cl26Tag::Start);
    }

    let parts: Vec<&str> = tag.split('|').collect();
    if parts.len() < 5 {
        return None;
    }

    let scope = decode_scope(&parts[1..5])?;
    match parts[0] {
        "program" if parts.len() == 5 => Some(Cl26Tag::Program(scope)),
        "tail" if parts.len() == 5 => Some(Cl26Tag::Tail(scope)),
        "op-args" if parts.len() == 6 => Some(Cl26Tag::OpArgs(scope, parts[5].parse().ok()?)),
        "assign-tail" if parts.len() == 6 => {
            Some(Cl26Tag::AssignTail(scope, parts[5].parse().ok()?))
        }
        "let-bindings" if parts.len() == 6 => {
            Some(Cl26Tag::LetBindings(scope, decode_list(parts[5])))
        }
        _ => None,
    }
}

fn atom(loc: &Srcloc, name: &str) -> Rc<SExp> {
    Rc::new(SExp::Atom(loc.clone(), name.as_bytes().to_vec()))
}

fn integer(loc: &Srcloc, value: i64) -> Rc<SExp> {
    Rc::new(SExp::Integer(loc.clone(), value.to_bigint().unwrap()))
}

fn nil(loc: &Srcloc) -> Rc<SExp> {
    Rc::new(SExp::Nil(loc.clone()))
}

fn cons(loc: &Srcloc, left: Rc<SExp>, right: Rc<SExp>) -> Rc<SExp> {
    Rc::new(SExp::Cons(loc.clone(), left, right))
}

fn list(loc: &Srcloc, items: &[Rc<SExp>]) -> Rc<SExp> {
    Rc::new(enlist(loc.clone(), items))
}

fn placeholder(loc: &Srcloc, idx: usize, tag: Cl26Tag) -> Rc<SExp> {
    atom(loc, &format!("${{{idx}:{}}}", encode_tag(&tag)))
}

struct Cl26ProgramFuzz;

impl FuzzTypeParams for Cl26ProgramFuzz {
    type Tag = Vec<u8>;
    type Expr = Rc<SExp>;
    type Error = String;
    type State = Cl26ProgramState;
}

#[derive(Clone, Debug)]
struct Cl26ProgramState {
    srcloc: Srcloc,
    top_args: Vec<String>,
    next_const: usize,
    next_func: usize,
    next_var: usize,
    next_int: i64,
    next_op: usize,
}

impl Cl26ProgramState {
    fn new<R: Rng>(rng: &mut R) -> Self {
        let srcloc = Srcloc::start("*random-cl26*");
        let mut top_args = Vec::new();
        for prefix in ["A", "B", "C"] {
            let random_suffix = sexp::random_atom_name(rng, 2);
            top_args.push(format!(
                "{}{}",
                prefix,
                String::from_utf8(random_suffix).expect("random names are ASCII")
            ));
        }

        Cl26ProgramState {
            srcloc,
            top_args,
            next_const: 0,
            next_func: 0,
            next_var: 0,
            next_int: 1,
            next_op: 0,
        }
    }

    fn top_scope(&self) -> Scope {
        Scope {
            depth: 4,
            vars: self.top_args.clone(),
            funcs: Vec::new(),
            consts: Vec::new(),
        }
    }

    fn fresh_const(&mut self) -> String {
        let name = format!("C{}", self.next_const);
        self.next_const += 1;
        name
    }

    fn fresh_func(&mut self) -> String {
        let name = format!("F{}", self.next_func);
        self.next_func += 1;
        name
    }

    fn fresh_var(&mut self) -> String {
        let name = format!("V{}", self.next_var);
        self.next_var += 1;
        name
    }

    fn fresh_int(&mut self) -> Rc<SExp> {
        let value = ((self.next_int * 17 + 11) % 97) - 31;
        self.next_int += 1;
        integer(&self.srcloc, value)
    }

    fn next_operator(&mut self) -> &'static str {
        let op = match self.next_op % 3 {
            0 => "+",
            1 => "-",
            _ => "*",
        };
        self.next_op += 1;
        op
    }
}

struct StartRule;

impl Rule<Cl26ProgramFuzz> for StartRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        if decode_tag(tag) != Some(Cl26Tag::Start) {
            return Ok(None);
        }

        let loc = &state.srcloc;
        let args: Vec<Rc<SExp>> = state.top_args.iter().map(|arg| atom(loc, arg)).collect();
        let include = list(loc, &[atom(loc, "include"), atom(loc, "*standard-cl-26*")]);
        let program = placeholder(loc, idx, Cl26Tag::Program(state.top_scope()));

        Ok(Some(cons(
            loc,
            atom(loc, "mod"),
            cons(loc, list(loc, &args), cons(loc, include, program)),
        )))
    }
}

struct ProgramTailRule;

impl Rule<Cl26ProgramFuzz> for ProgramTailRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Program(scope)) = decode_tag(tag) else {
            return Ok(None);
        };

        Ok(Some(cons(
            &state.srcloc,
            placeholder(&state.srcloc, idx, Cl26Tag::Tail(scope)),
            nil(&state.srcloc),
        )))
    }
}

struct ProgramDefunRule;

impl Rule<Cl26ProgramFuzz> for ProgramDefunRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Program(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 || state.next_func >= MAX_HELPERS {
            return Ok(None);
        }

        let func = state.fresh_func();
        let arg = state.fresh_var();
        let mut funcs_for_next_forms = scope.funcs.clone();
        funcs_for_next_forms.push(func.clone());
        let body_scope = Scope {
            depth: scope.depth - 1,
            vars: vec![arg.clone()],
            funcs: scope.funcs.clone(),
            consts: scope.consts.clone(),
        };
        let next_scope = Scope {
            funcs: funcs_for_next_forms,
            ..scope
        };
        let loc = &state.srcloc;
        let helper = list(
            loc,
            &[
                atom(loc, "defun"),
                atom(loc, &func),
                list(loc, &[atom(loc, &arg)]),
                placeholder(loc, idx, Cl26Tag::Tail(body_scope)),
            ],
        );
        let next_program = placeholder(loc, idx + 1, Cl26Tag::Program(next_scope));

        Ok(Some(cons(loc, helper, next_program)))
    }
}

struct ProgramDefconstantRule;

impl Rule<Cl26ProgramFuzz> for ProgramDefconstantRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Program(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || state.next_const >= MAX_HELPERS {
            return Ok(None);
        }

        let name = state.fresh_const();
        let primitive = state.fresh_int();
        let mut next_scope = scope.clone();
        next_scope.consts.push(name.clone());
        let loc = &state.srcloc;
        let helper = list(
            loc,
            &[atom(loc, "defconstant"), atom(loc, &name), primitive],
        );
        let next_program = placeholder(loc, idx, Cl26Tag::Program(next_scope));

        Ok(Some(cons(loc, helper, next_program)))
    }
}

struct ProgramDefconstRule;

impl Rule<Cl26ProgramFuzz> for ProgramDefconstRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Program(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 || state.next_const >= MAX_HELPERS {
            return Ok(None);
        }

        let name = state.fresh_const();
        let mut next_scope = scope.clone();
        next_scope.consts.push(name.clone());
        let value_scope = Scope {
            depth: 0,
            vars: Vec::new(),
            funcs: Vec::new(),
            consts: scope.consts.clone(),
        };
        let loc = &state.srcloc;
        let helper = list(
            loc,
            &[
                atom(loc, "defconst"),
                atom(loc, &name),
                placeholder(loc, idx, Cl26Tag::Tail(value_scope)),
            ],
        );
        let next_program = placeholder(loc, idx + 1, Cl26Tag::Program(next_scope));

        Ok(Some(cons(loc, helper, next_program)))
    }
}

struct TailPrimitiveRule;

impl Rule<Cl26ProgramFuzz> for TailPrimitiveRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(_scope)) = decode_tag(tag) else {
            return Ok(None);
        };

        Ok(Some(state.fresh_int()))
    }
}

struct TailVarRule;

impl Rule<Cl26ProgramFuzz> for TailVarRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if scope.vars.is_empty() {
            return Ok(None);
        }

        let chosen = &scope.vars[idx % scope.vars.len()];
        Ok(Some(atom(&state.srcloc, chosen)))
    }
}

struct TailConstRule;

impl Rule<Cl26ProgramFuzz> for TailConstRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if scope.consts.is_empty() {
            return Ok(None);
        }

        let chosen = &scope.consts[idx % scope.consts.len()];
        Ok(Some(atom(&state.srcloc, chosen)))
    }
}

struct TailFunctionCallRule;

impl Rule<Cl26ProgramFuzz> for TailFunctionCallRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 || scope.funcs.is_empty() {
            return Ok(None);
        }

        let loc = &state.srcloc;
        let chosen = &scope.funcs[idx % scope.funcs.len()];
        Ok(Some(list(
            loc,
            &[
                atom(loc, chosen),
                placeholder(loc, idx, Cl26Tag::Tail(scope.child())),
            ],
        )))
    }
}

struct TailIfRule;

impl Rule<Cl26ProgramFuzz> for TailIfRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 {
            return Ok(None);
        }

        let loc = &state.srcloc;
        let child = scope.child();
        Ok(Some(list(
            loc,
            &[
                atom(loc, "if"),
                placeholder(loc, idx, Cl26Tag::Tail(child.clone())),
                placeholder(loc, idx + 1, Cl26Tag::Tail(child.clone())),
                placeholder(loc, idx + 2, Cl26Tag::Tail(child)),
            ],
        )))
    }
}

struct TailAssignRule;

impl Rule<Cl26ProgramFuzz> for TailAssignRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 {
            return Ok(None);
        }

        let remaining_bindings = 1 + (idx % 2);
        Ok(Some(cons(
            &state.srcloc,
            atom(&state.srcloc, "assign"),
            placeholder(
                &state.srcloc,
                idx,
                Cl26Tag::AssignTail(scope.child(), remaining_bindings),
            ),
        )))
    }
}

struct AssignTailBindingRule;

impl Rule<Cl26ProgramFuzz> for AssignTailBindingRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::AssignTail(scope, remaining)) = decode_tag(tag) else {
            return Ok(None);
        };
        if remaining == 0 {
            return Ok(None);
        }

        let name = state.fresh_var();
        let value_scope = scope.clone();
        let body_scope = scope.with_var(name.clone());
        let loc = &state.srcloc;
        Ok(Some(cons(
            loc,
            atom(loc, &name),
            cons(
                loc,
                placeholder(loc, idx, Cl26Tag::Tail(value_scope)),
                placeholder(loc, idx + 1, Cl26Tag::AssignTail(body_scope, remaining - 1)),
            ),
        )))
    }
}

struct AssignTailFinalRule;

impl Rule<Cl26ProgramFuzz> for AssignTailFinalRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::AssignTail(scope, remaining)) = decode_tag(tag) else {
            return Ok(None);
        };
        if remaining != 0 {
            return Ok(None);
        }

        Ok(Some(cons(
            &state.srcloc,
            placeholder(&state.srcloc, idx, Cl26Tag::Tail(scope)),
            nil(&state.srcloc),
        )))
    }
}

struct TailLetRule;

impl Rule<Cl26ProgramFuzz> for TailLetRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 {
            return Ok(None);
        }

        let names: Vec<String> = (0..(1 + (idx % 2))).map(|_| state.fresh_var()).collect();
        let mut body_scope = scope.child();
        body_scope.vars.extend(names.iter().cloned());
        let loc = &state.srcloc;
        Ok(Some(list(
            loc,
            &[
                atom(loc, "let"),
                placeholder(loc, idx, Cl26Tag::LetBindings(scope.child(), names)),
                placeholder(loc, idx + 1, Cl26Tag::Tail(body_scope)),
            ],
        )))
    }
}

struct LetBindingsRule;

impl Rule<Cl26ProgramFuzz> for LetBindingsRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::LetBindings(scope, names)) = decode_tag(tag) else {
            return Ok(None);
        };
        let loc = &state.srcloc;
        let Some((name, rest)) = names.split_first() else {
            return Ok(Some(nil(loc)));
        };

        let binding = list(
            loc,
            &[
                atom(loc, name),
                placeholder(loc, idx, Cl26Tag::Tail(scope.clone())),
            ],
        );
        Ok(Some(cons(
            loc,
            binding,
            placeholder(loc, idx + 1, Cl26Tag::LetBindings(scope, rest.to_vec())),
        )))
    }
}

struct TailOperatorRule;

impl Rule<Cl26ProgramFuzz> for TailOperatorRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::Tail(scope)) = decode_tag(tag) else {
            return Ok(None);
        };
        if terminate || scope.depth == 0 {
            return Ok(None);
        }

        let op = state.next_operator();
        let loc = &state.srcloc;
        Ok(Some(cons(
            loc,
            atom(loc, op),
            placeholder(loc, idx, Cl26Tag::OpArgs(scope.child(), 2)),
        )))
    }
}

struct OpArgsRule;

impl Rule<Cl26ProgramFuzz> for OpArgsRule {
    fn check(
        &self,
        state: &mut Cl26ProgramState,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(Cl26Tag::OpArgs(scope, remaining)) = decode_tag(tag) else {
            return Ok(None);
        };
        let loc = &state.srcloc;
        if remaining == 0 {
            return Ok(Some(nil(loc)));
        }

        Ok(Some(cons(
            loc,
            placeholder(loc, idx, Cl26Tag::Tail(scope.clone())),
            placeholder(loc, idx + 1, Cl26Tag::OpArgs(scope, remaining - 1)),
        )))
    }
}

/// Generate a random, scope-correct CL26 program using the SExp fuzz hooks.
pub fn random_cl26_program<R: Rng + Sized>(rng: &mut R) -> String {
    let mut state = Cl26ProgramState::new(rng);
    let topnode = placeholder(&state.srcloc, 0, Cl26Tag::Start);
    let rules: Vec<Rc<dyn Rule<Cl26ProgramFuzz>>> = vec![
        Rc::new(StartRule),
        Rc::new(ProgramTailRule),
        Rc::new(ProgramDefunRule),
        Rc::new(ProgramDefconstantRule),
        Rc::new(ProgramDefconstRule),
        Rc::new(TailPrimitiveRule),
        Rc::new(TailVarRule),
        Rc::new(TailConstRule),
        Rc::new(TailFunctionCallRule),
        Rc::new(TailIfRule),
        Rc::new(TailAssignRule),
        Rc::new(AssignTailBindingRule),
        Rc::new(AssignTailFinalRule),
        Rc::new(TailLetRule),
        Rc::new(LetBindingsRule),
        Rc::new(TailOperatorRule),
        Rc::new(OpArgsRule),
    ];
    let mut fuzzer = FuzzGenerator::new(topnode, &rules);
    let mut expansions = 0;
    while fuzzer
        .expand(
            &mut state,
            expansions > MAX_EXPANSIONS_BEFORE_TERMINATING,
            rng,
        )
        .expect("CL26 program grammar should keep expanding")
    {
        expansions += 1;
        assert!(
            expansions < MAX_EXPANSIONS_TOTAL,
            "CL26 program generation should terminate"
        );
    }

    fuzzer.result().to_string()
}

fn compiler_output_to_hex(compiler_name: &str, program: &str, compiled: &str) -> String {
    let mut allocator = Allocator::new();
    let assembled = assemble(&mut allocator, compiled.trim()).unwrap_or_else(|err| {
        panic!("{compiler_name} output did not assemble: {err:?}\nprogram:\n{program}\ncompiled:\n{compiled}")
    });
    let mut stream_out = Stream::new(None);
    sexp_to_stream(&mut allocator, assembled, &mut stream_out);
    hex::encode(&stream_out.get_value().data())
}

fn compile_current_branch_to_hex(program: &str) -> String {
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    compiler_output_to_hex("current compiler", program, &compiled)
}

fn compile_chialisp_043_to_hex(program: &str) -> String {
    let program_run = Exec::cmd(format!("{}/.cargo/bin/run", env::var("HOME").unwrap()))
        .arg(program)
        .capture()
        .expect("should run");
    eprintln!("{}", program_run.stderr_str());
    compiler_output_to_hex("current compiler", program, &program_run.stdout_str())
}

#[test]
fn random_cl26_programs_match_chialisp_043_hex() {
    for seed in 0..GENERATED_PROGRAMS_TO_COMPARE {
        let mut rng = simple_seeded_rng(0xC126_0000 | seed);
        let program = random_cl26_program(&mut rng);
        let current_hex = compile_current_branch_to_hex(&program);
        let chialisp_043_hex = compile_chialisp_043_to_hex(&program);
        assert_eq!(
            current_hex.trim(),
            chialisp_043_hex.trim(),
            "compiled hex mismatch for generated CL26 program seed {seed}:\n{program}"
        );
    }
}
