use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use clvm_rs::allocator::Allocator;
use num_bigint::ToBigInt;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::cldb::{CldbNoOverride, CldbRun, CldbRunEnv};
use crate::compiler::clvm;
use crate::compiler::clvm::{sha256tree, truthy, RunStep};
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::{decode_string, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;

#[derive(Clone, Debug)]
pub struct RunStepRelevantInfo {
    name: String,
    hash: Vec<u8>,
    prog: Rc<SExp>,
    formal_parameters: Rc<SExp>,
    runtime_argument_values: Rc<SExp>,
    left_env: bool,
}

#[derive(Clone, Debug)]
pub enum RunPurpose {
    ComputeArgument,
    Main,
}

pub struct HierarchyFrame {
    pub purpose: RunPurpose,

    pub prog: Rc<SExp>,
    pub env: Rc<SExp>,

    pub function_hash: Vec<u8>,
    pub function_name: String,
    pub function_arguments: Rc<SExp>,
    pub function_left_env: bool,

    pub source: Srcloc,
    pub named_args: HashMap<String, Rc<SExp>>,

    pub run: CldbRun,
}

pub struct HierarchialRunner {
    allocator: Allocator,
    runner: Rc<dyn TRunProgram>,
    prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    symbol_table: Rc<HashMap<String, String>>,
    error: bool,
    input_file: Option<String>,
    program_lines: Rc<Vec<String>>,
    prog: Rc<SExp>,
    flags: u32,

    pub running: Vec<HierarchyFrame>,
}

#[derive(Clone, Debug)]
pub enum HierarchialStepResult {
    ShapeChange,
    Info(Option<BTreeMap<String, String>>),
    Done(Option<BTreeMap<String, String>>),
}

pub enum RunClass {
    Primitive(Rc<SExp>),
    Application(RunStepRelevantInfo),
}

pub fn hex_of_hash(hash: &[u8]) -> String {
    Bytes::new(Some(BytesFromType::Raw(hash.to_vec()))).hex()
}

pub fn is_op(v: u8, op: Rc<SExp>) -> bool {
    match op.borrow() {
        SExp::Integer(_, i) => *i == v.to_bigint().unwrap(),
        SExp::Atom(_, n) => *n == vec![v],
        SExp::QuotedString(_, _, n) => *n == vec![v],
        _ => false,
    }
}

pub fn is_apply_op(op: Rc<SExp>) -> bool {
    is_op(2, op)
}

pub fn get_fun_hash(op: Rc<SExp>, sexp: Rc<SExp>) -> Option<(Vec<u8>, Rc<SExp>, Rc<SExp>)> {
    if let SExp::Cons(_, prog, args) = sexp.borrow() {
        if is_apply_op(op) {
            if let SExp::Cons(_, env, _) = args.borrow() {
                return Some((clvm::sha256tree(prog.clone()), prog.clone(), env.clone()));
            }
        }
    }

    None
}

pub fn relevant_run_step_info(
    symbol_table: &HashMap<String, String>,
    step: &RunStep,
) -> Option<RunStepRelevantInfo> {
    if let RunStep::Op(op, _, args, Some(remaining_args), _) = step {
        if remaining_args.is_empty() {
            return get_fun_hash(op.clone(), args.clone())
                .and_then(|(hash, prog, env)| make_relevant_info(symbol_table, &hash, prog, env));
        }
    }

    None
}

fn sexp_from_symbol_table(
    symbol_table: &HashMap<String, String>,
    item_name: &str,
) -> Option<Rc<SExp>> {
    let loc = Srcloc::compiler_internal_srcloc();
    symbol_table.get(item_name).and_then(|data| {
        parse_sexp(loc.clone(), data.as_bytes().iter().copied())
            .ok()
            .and_then(|p| {
                if p.is_empty() {
                    None
                } else {
                    Some(Rc::new(p[0].atomize()))
                }
            })
    })
}

fn uses_left_env(symbol_table: &HashMap<String, String>, hash: &[u8]) -> bool {
    let hex_hash = hex_of_hash(hash);
    let item_name = format!("{hex_hash}_left_env");
    sexp_from_symbol_table(symbol_table, &item_name)
        .map(truthy)
        .unwrap_or_else(|| false)
}

fn make_relevant_info(
    symbol_table: &HashMap<String, String>,
    hash: &[u8],
    prog: Rc<SExp>,
    env: Rc<SExp>,
) -> Option<RunStepRelevantInfo> {
    let hex_hash = hex_of_hash(hash);
    let args_name = format!("{hex_hash}_arguments");
    let fun_args = sexp_from_symbol_table(symbol_table, &args_name).unwrap_or_else(|| {
        let name: Vec<u8> = args_name.as_bytes().to_vec();
        Rc::new(SExp::Atom(prog.loc(), name))
    });
    symbol_table
        .get(&hex_hash)
        .map(|fun_name| RunStepRelevantInfo {
            hash: hash.to_vec(),
            name: fun_name.clone(),
            formal_parameters: fun_args,
            prog: prog.clone(),
            runtime_argument_values: env.clone(),
            left_env: uses_left_env(symbol_table, hash),
        })
}

fn get_args_from_env(
    arg_map: &mut HashMap<String, Rc<SExp>>,
    lines: Rc<Vec<String>>,
    args: Rc<SExp>,
    env: Rc<SExp>,
    left_env: bool,
) {
    match (args.atomize(), env.borrow()) {
        (SExp::Cons(_, a, b), SExp::Cons(_, x, y)) => {
            if left_env {
                get_args_from_env(arg_map, lines, args.clone(), y.clone(), false);
                return;
            }

            get_args_from_env(arg_map, lines.clone(), a, x.clone(), false);
            get_args_from_env(arg_map, lines, b, y.clone(), false);
        }
        (SExp::Atom(_, n), _) => {
            arg_map.insert(decode_string(&n), env);
        }
        _ => {}
    }
}

impl HierarchialRunner {
    pub fn new(
        runner: Rc<dyn TRunProgram>,
        prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
        input_file: Option<String>,
        program_lines: Rc<Vec<String>>,
        symbol_table: Rc<HashMap<String, String>>,
        prog: Rc<SExp>,
        env: Rc<SExp>,
    ) -> Self {
        let step = clvm::start_step(prog.clone(), env.clone());
        let run = CldbRun::new(
            runner.clone(),
            prim_map.clone(),
            Box::new(CldbRunEnv::new(
                input_file.clone(),
                program_lines.clone(),
                Box::new(CldbNoOverride::new()),
            )),
            step,
        );

        let fun_args = sexp_from_symbol_table(symbol_table.borrow(), "__chia__main_arguments")
            .unwrap_or_else(|| Rc::new(SExp::Nil(prog.loc())));

        let mut program_args = HashMap::new();
        get_args_from_env(
            &mut program_args,
            program_lines.clone(),
            fun_args.clone(),
            env.clone(),
            false,
        );

        HierarchialRunner {
            allocator: Allocator::new(),
            runner,
            prim_map,
            symbol_table,
            input_file: input_file.clone(),
            program_lines,
            error: false,
            prog: prog.clone(),
            flags: 0,

            running: vec![HierarchyFrame {
                purpose: RunPurpose::Main,

                prog: prog.clone(),
                env,

                function_hash: sha256tree(prog.clone()),
                function_name: input_file.unwrap_or_else(|| {
                    format!(
                        "clvm_program_{}",
                        hex_of_hash(&clvm::sha256tree(prog.clone()))
                    )
                }),
                function_arguments: fun_args,
                function_left_env: false,

                named_args: program_args,
                source: prog.loc(),

                run,
            }],
        }
    }

    pub fn set_flags(&mut self, flags: u32) {
        self.flags = flags;
        for r in self.running.iter_mut() {
            r.run.set_flags(flags);
        }
    }

    pub fn is_ended(&self) -> bool {
        self.running.is_empty()
            || self.error
            || self.running.len() == 1 && self.running[0].run.is_ended()
    }

    fn push_synthetic_stack_frame(&mut self, current_env: Rc<SExp>, info: &RunStepRelevantInfo) {
        let arg_step = clvm::start_step(info.prog.clone(), info.runtime_argument_values.clone());

        let mut arg_run = CldbRun::new(
            self.runner.clone(),
            self.prim_map.clone(),
            Box::new(CldbRunEnv::new(
                self.input_file.clone(),
                self.program_lines.clone(),
                Box::new(CldbNoOverride::new()),
            )),
            arg_step,
        );

        arg_run.set_flags(self.flags);

        let mut named_args = HashMap::new();
        get_args_from_env(
            &mut named_args,
            self.program_lines.clone(),
            info.formal_parameters.clone(),
            info.runtime_argument_values.clone(),
            info.left_env,
        );

        let arg_frame = HierarchyFrame {
            purpose: RunPurpose::ComputeArgument,

            prog: info.prog.clone(),
            env: info.runtime_argument_values.clone(),

            function_hash: info.hash.clone(),
            function_name: info.name.clone(),
            function_arguments: info.runtime_argument_values.clone(),
            function_left_env: info.left_env,

            named_args: named_args.clone(),

            run: arg_run,
            source: info.prog.loc(),
        };

        // Make an empty frame to repopulate (maybe option here?).
        let step = clvm::start_step(info.prog.clone(), current_env.clone());
        let mut run = CldbRun::new(
            self.runner.clone(),
            self.prim_map.clone(),
            Box::new(CldbRunEnv::new(
                self.input_file.clone(),
                self.program_lines.clone(),
                Box::new(CldbNoOverride::new()),
            )),
            step,
        );

        run.set_flags(self.flags);

        self.running.push(HierarchyFrame {
            purpose: RunPurpose::Main,

            prog: info.prog.clone(),
            env: current_env,

            function_hash: info.hash.clone(),
            function_name: info.name.clone(),
            function_arguments: info.formal_parameters.clone(),
            function_left_env: info.left_env,
            source: info.prog.loc(),

            named_args,

            run,
        });

        self.running.push(arg_frame);
    }

    pub fn step(&mut self) -> Result<HierarchialStepResult, RunFailure> {
        if self.running.is_empty() {
            return Err(RunFailure::RunErr(
                self.prog.loc(),
                "no running code".to_string(),
            ));
        }

        let mut idx = self.running.len() - 1;
        let current_env = self.running[idx].env.clone();
        let current_step = self.running[idx].run.current_step();
        if let Some(outcome) = self.running[idx].run.final_result() {
            if self.running.pop().is_none() {
                return Err(RunFailure::RunErr(
                    self.prog.loc(),
                    "running had no frame to pop".to_string(),
                ));
            }

            if self.running.is_empty() {
                return Ok(HierarchialStepResult::Done(None));
            }

            idx -= 1;

            self.running[idx].env = outcome;

            let step = clvm::step_return_value(
                &self.running[idx].run.current_step(),
                self.running[idx].env.clone(),
            );

            self.running[idx].run = CldbRun::new(
                self.runner.clone(),
                self.prim_map.clone(),
                Box::new(CldbRunEnv::new(
                    self.input_file.clone(),
                    self.program_lines.clone(),
                    Box::new(CldbNoOverride::new()),
                )),
                step,
            );

            self.running[idx].run.set_flags(self.flags);

            Ok(HierarchialStepResult::ShapeChange)
        } else if let Some(info) = relevant_run_step_info(&self.symbol_table, &current_step) {
            // Create a frame based on the last argument.
            self.push_synthetic_stack_frame(current_env, &info);

            Ok(HierarchialStepResult::ShapeChange)
        } else {
            // Not final result, we'll step the top of the stack.
            let info = self.running[idx].run.step(&mut self.allocator);
            if let Some(i) = &info {
                self.error |= i.get("Failure").is_some();
            }
            Ok(HierarchialStepResult::Info(info))
        }
    }
}
