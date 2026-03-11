use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::mem::swap;
use std::rc::Rc;

use clvm_rs::allocator;
use clvm_rs::allocator::{Allocator, NodePtr};
use clvm_rs::error::EvalErr;
use num_bigint::ToBigInt;

use crate::classic::clvm::__type_compatibility__::{
    Bytes, BytesFromType, Stream, UnvalidatedBytesFromType,
};
use crate::classic::clvm::serialize::{sexp_from_stream, SimpleCreateCLVMObject};
//use crate::classic::clvm::syntax_error::SyntaxErr;
use crate::classic::clvm_tools::sha256tree::sha256tree;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::clvm;
use crate::compiler::clvm::{convert_from_clvm_rs, run_step, RunStep};
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::util::u8_from_number;
use crate::util::Number;

pub const FAVOR_HEX: u32 = 1;

fn print_atom() -> SExp {
    SExp::Atom(Srcloc::start("*print*"), b"$print$".to_vec())
}

#[derive(Clone, Debug)]
pub struct PriorResult {
    reference: usize,
    // value: Rc<SExp>, // In future, we'll want to know the value produced.
}

fn format_arg_inputs(args: &[PriorResult]) -> String {
    let value_strings: Vec<String> = args.iter().map(|pr| pr.reference.to_string()).collect();
    value_strings.join(", ")
}

fn get_arg_associations(
    associations: &HashMap<Number, PriorResult>,
    args: Rc<SExp>,
) -> Vec<PriorResult> {
    let mut arg_exp: Rc<SExp> = args;
    let mut result: Vec<PriorResult> = Vec::new();
    loop {
        if let SExp::Cons(_, arg, rest) = arg_exp.borrow() {
            if let Some(n) = arg
                .get_number()
                .ok()
                .as_ref()
                .and_then(|n| associations.get(n))
            {
                result.push(n.clone());
            }
            arg_exp = rest.clone();
        } else {
            return result;
        }
    }
}

/// An interface which allows consumers to inject their own functionality into
/// cldb runs, including possibly mocking functions, performing tracing and
/// other desired things.  The result of the operation can be dictated when
/// the runnable is asked to replace the step state.
pub trait CldbRunnable {
    fn replace_step(&self, step: &RunStep) -> Option<Result<RunStep, RunFailure>>;
}

/// A CldbEnvironment is a container for a function-oriented view of clvm programs
/// when running in Cldb.
pub trait CldbEnvironment {
    fn add_context(
        &self,
        s: &SExp,
        c: &SExp,
        args: Option<Rc<SExp>>,
        context_result: &mut BTreeMap<String, String>,
    );
    fn add_function(&self, s: &SExp, context_result: &mut BTreeMap<String, String>);
    fn get_override(&self, s: &RunStep) -> Option<Result<RunStep, RunFailure>>;
}

/// CldbRun is the main object used to run CLVM code in a stepwise way.  The main
/// advantage of CldbRun over clvmr's runner is that the caller observes a new
/// step being returned after it asks for each step to be run.  The progress of
/// evaulation is observable and hopefully understandable and in an order which,
/// combined with observing the RunStep can help with debugging.
///
/// CldmbRun contains a RunStep and moves evaluation forward every time its step
/// method is called, along with having some convenience methods, like being able
/// to ask whether the run ended and what the final result was (if it completed).
///
/// The result is a map of key value pairs indicating various information about
/// the run.
pub struct CldbRun {
    runner: Rc<dyn TRunProgram>,
    prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    env: Box<dyn CldbEnvironment>,

    step: RunStep,

    ended: bool,
    final_result: Option<Rc<SExp>>,
    to_print: BTreeMap<String, String>,
    in_expr: bool,
    row: usize,
    print_only: bool,
    flags: u32,

    outputs_to_step: HashMap<Number, PriorResult>,
}

fn humanize(a: Rc<SExp>) -> Rc<SExp> {
    match a.borrow() {
        SExp::Integer(l, i) => {
            // If it has a nice string representation then show that.
            let bytes_of_int = u8_from_number(i.clone());
            if bytes_of_int.len() > 2 && bytes_of_int.iter().all(|b| *b >= 32 && *b < 127) {
                Rc::new(SExp::QuotedString(l.clone(), b'\'', bytes_of_int))
            } else {
                a.clone()
            }
        }
        SExp::Cons(l, a, b) => {
            let new_a = humanize(a.clone());
            let new_b = humanize(b.clone());
            Rc::new(SExp::Cons(l.clone(), new_a, new_b))
        }
        _ => a.clone(),
    }
}

pub fn improve_presentation(a: Rc<SExp>, flags: u32) -> Rc<SExp> {
    match a.borrow() {
        SExp::Atom(l, av) => {
            if av.len() > 2 && (flags & FAVOR_HEX) != 0 {
                Rc::new(SExp::QuotedString(l.clone(), b'x', av.clone()))
            } else {
                a.clone()
            }
        }
        SExp::Integer(l, i) => {
            // If it has a nice string representation then show that.
            let bytes_of_int = u8_from_number(i.clone());
            if bytes_of_int.len() > 2 && (flags & FAVOR_HEX) != 0 {
                Rc::new(SExp::QuotedString(l.clone(), b'x', bytes_of_int))
            } else {
                a.clone()
            }
        }
        SExp::Cons(loc, l, r) => {
            let new_l = improve_presentation(l.clone(), flags);
            let new_r = improve_presentation(r.clone(), flags);
            if Rc::as_ptr(&new_l) == Rc::as_ptr(l) && Rc::as_ptr(&new_r) == Rc::as_ptr(r) {
                return a.clone();
            }

            Rc::new(SExp::Cons(loc.clone(), new_l, new_r))
        }
        _ => a.clone(),
    }
}

fn is_print_request(a: &SExp) -> Option<(Srcloc, Rc<SExp>)> {
    if let SExp::Cons(l, f, r) = a {
        if &print_atom() == f.borrow() {
            return Some((l.clone(), humanize(r.clone())));
        }
    }

    None
}

impl CldbRun {
    /// Create a new CldbRun for running a program.
    /// Takes an CldbEnvironment and a prepared RunStep, which will be stepped
    /// through.  The CldbEnvironment specifies places where the consumer has the
    /// ability to examine the run step and possibly alter the result of execution.
    pub fn new(
        runner: Rc<dyn TRunProgram>,
        prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
        env: Box<dyn CldbEnvironment>,
        step: RunStep,
    ) -> Self {
        CldbRun {
            runner,
            prim_map,
            env,
            step,
            ended: false,
            final_result: None,
            to_print: BTreeMap::new(),
            in_expr: false,
            row: 0,
            outputs_to_step: HashMap::<Number, PriorResult>::new(),
            print_only: false,
            flags: 0,
        }
    }

    pub fn set_flags(&mut self, flags: u32) {
        self.flags = flags;
    }

    pub fn set_print_only(&mut self, pronly: bool) {
        self.print_only = pronly;
    }

    pub fn is_ended(&self) -> bool {
        self.ended
    }

    pub fn final_result(&self) -> Option<Rc<SExp>> {
        self.final_result.clone()
    }

    pub fn should_print_basic_output(&self) -> bool {
        !self.print_only
    }

    pub fn step(&mut self, allocator: &mut Allocator) -> Option<BTreeMap<String, String>> {
        let mut produce_result = false;
        let mut result = BTreeMap::new();
        let new_step = match self.env.get_override(&self.step) {
            Some(v) => v,
            _ => run_step(
                allocator,
                self.runner.clone(),
                self.prim_map.clone(),
                &self.step,
                None,
            ),
        };

        // Allow overrides by consumers.

        match &new_step {
            Ok(RunStep::OpResult(l, x, _p)) => {
                if self.in_expr {
                    if self.should_print_basic_output() {
                        self.to_print
                            .insert("Result-Location".to_string(), l.to_string());
                        self.to_print.insert(
                            "Value".to_string(),
                            improve_presentation(x.clone(), self.flags).to_string(),
                        );
                        self.to_print
                            .insert("Row".to_string(), self.row.to_string());

                        if let Ok(n) = x.get_number() {
                            self.outputs_to_step.insert(
                                n,
                                PriorResult {
                                    reference: self.row,
                                    // value: x.clone(), // for future
                                },
                            );
                        }
                        swap(&mut self.to_print, &mut result);
                        produce_result = true;
                    }

                    self.in_expr = false;
                }
            }
            Ok(RunStep::Done(l, x)) => {
                let final_value = improve_presentation(x.clone(), self.flags);
                self.to_print
                    .insert("Final-Location".to_string(), l.to_string());
                self.to_print
                    .insert("Final".to_string(), final_value.to_string());

                self.ended = true;
                self.final_result = Some(final_value);
                swap(&mut self.to_print, &mut result);
                produce_result = true;
            }
            Ok(RunStep::Step(_sexp, _c, _p)) => {}
            Ok(RunStep::Op(sexp, c, a, None, _p)) => {
                let should_print_basic_output = self.should_print_basic_output();
                if should_print_basic_output {
                    self.to_print
                        .insert("Operator-Location".to_string(), a.loc().to_string());
                    self.to_print
                        .insert("Operator".to_string(), sexp.to_string());
                }

                if let Ok(v) = sexp.get_number() {
                    if v == 11_u32.to_bigint().unwrap() && should_print_basic_output {
                        // Build source tree for hashes.
                        let arg_associations =
                            get_arg_associations(&self.outputs_to_step, a.clone());
                        let args = format_arg_inputs(&arg_associations);
                        self.to_print.insert("Argument-Refs".to_string(), args);
                    } else if v == 34_u32.to_bigint().unwrap() {
                        // Handle diagnostic output.
                        if let Some((loc, outputs)) = is_print_request(a) {
                            self.to_print
                                .insert("Print-Location".to_string(), loc.to_string());
                            self.to_print
                                .insert("Print".to_string(), outputs.to_string());
                            swap(&mut self.to_print, &mut result);
                            produce_result = true;
                        }
                    }
                }
                if should_print_basic_output {
                    self.env.add_context(
                        sexp.borrow(),
                        improve_presentation(c.clone(), self.flags).borrow(),
                        Some(improve_presentation(a.clone(), self.flags)),
                        &mut self.to_print,
                    );
                    self.env.add_function(sexp, &mut self.to_print);
                    self.in_expr = true;
                }
            }
            Ok(RunStep::Op(_sexp, _c, _a, Some(_v), _p)) => {}
            Err(RunFailure::RunExn(l, s)) => {
                self.to_print
                    .insert("Throw-Location".to_string(), l.to_string());
                self.to_print.insert(
                    "Throw".to_string(),
                    improve_presentation(s.clone(), self.flags).to_string(),
                );

                swap(&mut self.to_print, &mut result);
                self.ended = true;
                produce_result = true;
            }
            Err(RunFailure::RunErr(l, s)) => {
                self.to_print
                    .insert("Failure-Location".to_string(), l.to_string());
                self.to_print.insert("Failure".to_string(), s.to_string());

                swap(&mut self.to_print, &mut result);
                self.ended = true;
                produce_result = true;
            }
        }

        self.step = new_step.unwrap_or_else(|_| self.step.clone());

        if produce_result {
            self.row += 1;
            Some(result)
        } else {
            None
        }
    }

    pub fn current_step(&self) -> RunStep {
        self.step.clone()
    }
}

/// A simple implementation of CldbEnvironment that does not override anything.
pub struct CldbNoOverride {}

impl CldbRunnable for CldbNoOverride {
    fn replace_step(&self, _step: &RunStep) -> Option<Result<RunStep, RunFailure>> {
        None
    }
}

impl CldbNoOverride {
    pub fn new() -> Self {
        CldbNoOverride {}
    }

    pub fn new_symbols(_symbol_table: HashMap<String, String>) -> Self {
        CldbNoOverride {}
    }
}

impl Default for CldbNoOverride {
    fn default() -> Self {
        CldbNoOverride::new()
    }
}

/// Allow the caller to examine environment and return an expression that
/// will be quoted, used in conjunction with CldbEnvironment.
pub trait CldbSingleBespokeOverride {
    fn get_override(&self, env: Rc<SExp>) -> Result<Rc<SExp>, RunFailure>;
}

/// Provides a collection of overrides to be used with CldbEnvironment and
/// CldbRun to support use cases like examining the arguments given to a
/// specific function while CLVM code is executing or to mock functions in
/// a CLVM program.
pub struct CldbOverrideBespokeCode {
    symbol_table: HashMap<String, String>,
    overrides: HashMap<String, Box<dyn CldbSingleBespokeOverride>>,
}

impl CldbOverrideBespokeCode {
    /// Given the symbol table of a compiled CLVM program and a hashmap from
    /// function names to override specifications, provie a ClvmEnvironment that
    /// overrides the targeted functions with the given overrides, which are
    /// objects the consumer implements CldbSingleBespokeOverride for.
    ///
    /// These can do whatever the user likes, from inspecting the arguments
    /// to replacing the result.
    pub fn new(
        symbol_table: HashMap<String, String>,
        overrides: HashMap<String, Box<dyn CldbSingleBespokeOverride>>,
    ) -> Self {
        CldbOverrideBespokeCode {
            symbol_table,
            overrides,
        }
    }

    fn find_function_and_override_if_needed(
        &self,
        sexp: Rc<SExp>,
        _c: Rc<SExp>,
        f: Rc<SExp>,
        args: Rc<SExp>,
        p: Rc<RunStep>,
    ) -> Option<Result<RunStep, RunFailure>> {
        let fun_hash = clvm::sha256tree(f);
        let fun_hash_str = Bytes::new(Some(BytesFromType::Raw(fun_hash))).hex();

        self.symbol_table
            .get(&fun_hash_str)
            .and_then(|funname| self.overrides.get(funname))
            .map(|override_fn| {
                override_fn
                    .get_override(args.clone())
                    .map(|new_exp| RunStep::OpResult(sexp.loc(), new_exp, p.clone()))
            })
    }
}

impl CldbRunnable for CldbOverrideBespokeCode {
    fn replace_step(&self, step: &RunStep) -> Option<Result<RunStep, RunFailure>> {
        match step {
            RunStep::Op(sexp, context, arguments, None, parent) => match sexp.borrow() {
                SExp::Integer(_, i) => {
                    if *i == 2_u32.to_bigint().unwrap() {
                        match arguments.borrow() {
                            SExp::Cons(_, first, args) => self
                                .find_function_and_override_if_needed(
                                    sexp.clone(),
                                    context.clone(),
                                    first.clone(),
                                    args.clone(),
                                    parent.clone(),
                                ),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }
}

/// A small collection of information about the running program, including the
/// name of the source file and the lines of the program.  When present, this
/// allows names to be picked out of the source base and locations to be accurate.
///
/// Also provides a CldbRunnable that specifies the user's overrides.
pub struct CldbRunEnv {
    input_file: Option<String>,
    program_lines: Rc<Vec<String>>,
    overrides: Box<dyn CldbRunnable>,
}

impl CldbRunEnv {
    /// Make a new CldbRunEnv given useful information about the program being
    /// run.
    pub fn new(
        input_file: Option<String>,
        program_lines: Rc<Vec<String>>,
        runnable: Box<dyn CldbRunnable>,
    ) -> Self {
        CldbRunEnv {
            input_file,
            program_lines,
            overrides: runnable,
        }
    }

    fn extract_text(&self, l: &Srcloc) -> Option<String> {
        let use_line = if l.line < 1 { None } else { Some(l.line - 1) };
        let use_col = use_line.and(if l.col < 1 { None } else { Some(l.col - 1) });
        let end_col = use_col.map(|c| l.until.as_ref().map(|u| u.col - 1).unwrap_or_else(|| c + 1));
        use_line
            .and_then(|use_line| {
                use_col.and_then(|use_col| end_col.map(|end_col| (use_line, use_col, end_col)))
            })
            .and_then(|coords| {
                let use_line = coords.0;
                let use_col = coords.1;
                let mut end_col = coords.2;

                if use_line >= self.program_lines.len() {
                    None
                } else {
                    let line_text = self.program_lines[use_line].to_string();
                    if use_col >= line_text.len() {
                        None
                    } else if end_col >= line_text.len() {
                        end_col = line_text.len();
                        Some(line_text[use_col..end_col].to_string())
                    } else {
                        Some(line_text[use_col..end_col].to_string())
                    }
                }
            })
    }

    fn whether_is_apply(
        &self,
        s: &SExp,
        collector: &mut BTreeMap<String, String>,
        if_true: &dyn Fn(&mut BTreeMap<String, String>),
        if_false: &dyn Fn(&mut BTreeMap<String, String>),
    ) {
        if let SExp::Integer(_, i) = s {
            if *i == 2_i32.to_bigint().unwrap() {
                if_true(collector);
                return;
            }
        }

        if_false(collector);
    }
}

impl CldbEnvironment for CldbRunEnv {
    fn add_context(
        &self,
        s: &SExp,
        c: &SExp,
        args: Option<Rc<SExp>>,
        context_result: &mut BTreeMap<String, String>,
    ) {
        self.whether_is_apply(
            s,
            context_result,
            &|context_result| match c {
                SExp::Cons(_, a, b) => {
                    context_result.insert("Env".to_string(), a.to_string());
                    context_result.insert("Env-Args".to_string(), b.to_string());
                }
                _ => {
                    context_result.insert("Function-Context".to_string(), c.to_string());
                }
            },
            &|context_result| {
                if let Some(a) = &args {
                    context_result.insert("Arguments".to_string(), a.to_string());
                }
            },
        );
    }

    fn add_function(&self, s: &SExp, context_result: &mut BTreeMap<String, String>) {
        self.whether_is_apply(
            s,
            context_result,
            &|_context_result| {},
            &|context_result| {
                if let Some(name) = self.extract_text(&s.loc()) {
                    if Some(s.loc().file.to_string()) == self.input_file.clone() {
                        context_result.insert("Function".to_string(), name);
                    }
                }
            },
        );
    }

    fn get_override(&self, s: &RunStep) -> Option<Result<RunStep, RunFailure>> {
        self.overrides.replace_step(s)
    }
}

fn hex_to_modern_sexp_inner(
    allocator: &mut Allocator,
    symbol_table: &HashMap<String, String>,
    loc: Srcloc,
    program: NodePtr,
) -> Result<Rc<SExp>, EvalErr> {
    let hash = sha256tree(allocator, program);
    let hash_str = hash.hex();
    let srcloc = symbol_table
        .get(&hash_str)
        .map(|f| Srcloc::start(f))
        .unwrap_or_else(|| loc.clone());

    match allocator.sexp(program) {
        allocator::SExp::Pair(a, b) => Ok(Rc::new(SExp::Cons(
            srcloc.clone(),
            hex_to_modern_sexp_inner(allocator, symbol_table, srcloc.clone(), a)?,
            hex_to_modern_sexp_inner(allocator, symbol_table, srcloc, b)?,
        ))),
        _ => convert_from_clvm_rs(allocator, srcloc, program).map_err(|_| {
            EvalErr::InternalError(NodePtr::NIL, "clvm_rs allocator failed".to_string())
        }),
    }
}

/// A function which, given hex input, produces equivalent SExp.
/// All produced SExp have the location given in loc.
pub fn hex_to_modern_sexp(
    allocator: &mut Allocator,
    symbol_table: &HashMap<String, String>,
    loc: Srcloc,
    input_program: &str,
) -> Result<Rc<SExp>, RunFailure> {
    let input_serialized = Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(
        input_program.to_string(),
    )))
    .map_err(|e| RunFailure::RunErr(loc.clone(), e.to_string()))?;

    let mut stream = Stream::new(Some(input_serialized));
    let sexp = sexp_from_stream(allocator, &mut stream, Box::new(SimpleCreateCLVMObject {}))
        .map(|x| x.1)
        .map_err(|_| RunFailure::RunErr(loc.clone(), "Bad conversion from hex".to_string()))?;

    hex_to_modern_sexp_inner(allocator, symbol_table, loc.clone(), sexp).map_err(|_| {
        RunFailure::RunErr(loc, "Failed to convert from classic to modern".to_string())
    })
}
