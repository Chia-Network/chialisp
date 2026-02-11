use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use crate::compiler::clvm::{convert_from_clvm_rs, run};
use crate::compiler::compiler::{DefaultCompilerOpts, SHA256TREE_PROGRAM_CLVM};
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompilerOpts, CompilerOutput, DefconstData, DefunData, HelperForm,
};
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::frontend::compile_helperform;
use crate::compiler::fuzz::{ExprModifier, FuzzGenerator, FuzzTypeParams, Rule};
use crate::compiler::prims::primquote;
use crate::compiler::sexp::{decode_string, SExp};
use crate::compiler::srcloc::Srcloc;

use crate::tests::compiler::fuzz::{compose_sexp, GenError};
use crate::tests::compiler::modules::{
    hex_to_clvm, perform_compile_of_file, TestModuleCompilerOpts,
};

fn simple_run(
    opts: Rc<dyn CompilerOpts>,
    expr: Rc<SExp>,
    env: Rc<SExp>,
) -> Result<Rc<SExp>, CompileErr> {
    let mut allocator = Allocator::new();
    let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
    Ok(run(
        &mut allocator,
        runner,
        opts.prim_map(),
        expr,
        env,
        None,
        None,
    )?)
}

#[derive(Debug, Clone)]
enum SupportedOperators {
    Plus,
    Minus,
}

impl SupportedOperators {
    fn to_sexp(&self, srcloc: &Srcloc) -> Rc<SExp> {
        match self {
            SupportedOperators::Plus => compose_sexp(srcloc.clone(), "16"),
            SupportedOperators::Minus => compose_sexp(srcloc.clone(), "17"),
        }
    }
}

#[derive(Debug, Clone)]
enum ValueSpecification {
    ConstantValue(Rc<SExp>),
    ConstantRef(Vec<u8>),
    HashOfFunction(Vec<u8>),
    ApplyFunction(Vec<u8>, Rc<ValueSpecification>),
    ClvmBinop(
        SupportedOperators,
        Rc<ValueSpecification>,
        Rc<ValueSpecification>,
    ),
}

impl ValueSpecification {
    fn to_sexp(&self, srcloc: &Srcloc) -> Rc<SExp> {
        match self {
            ValueSpecification::ConstantValue(c) => c.clone(),
            ValueSpecification::ConstantRef(c) => Rc::new(SExp::Atom(srcloc.clone(), c.clone())),
            ValueSpecification::HashOfFunction(f) => compose_sexp(
                srcloc.clone(),
                &format!(
                    "(a (q . {SHA256TREE_PROGRAM_CLVM}) (list {}))",
                    decode_string(&f)
                ),
            ),
            ValueSpecification::ApplyFunction(name, v) => Rc::new(SExp::Cons(
                srcloc.clone(),
                Rc::new(SExp::Atom(srcloc.clone(), name.clone())),
                Rc::new(SExp::Cons(
                    srcloc.clone(),
                    compose_sexp(srcloc.clone(), "(- TIMES 1)"),
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        v.to_sexp(srcloc),
                        Rc::new(SExp::Nil(srcloc.clone())),
                    )),
                )),
            )),
            ValueSpecification::ClvmBinop(op, left, right) => Rc::new(SExp::Cons(
                srcloc.clone(),
                op.to_sexp(srcloc),
                Rc::new(SExp::Cons(
                    srcloc.clone(),
                    left.to_sexp(srcloc),
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        right.to_sexp(srcloc),
                        Rc::new(SExp::Nil(srcloc.clone())),
                    )),
                )),
            )),
        }
    }

    fn interpret(
        &self,
        opts: Rc<dyn CompilerOpts>,
        srcloc: &Srcloc,
        value_map: &BTreeMap<Vec<u8>, Rc<SExp>>,
        hash_map: &BTreeMap<Vec<u8>, Vec<u8>>,
    ) -> Rc<SExp> {
        match self {
            ValueSpecification::ConstantValue(c) => c.clone(),
            ValueSpecification::ConstantRef(c) => {
                if let Some(value) = value_map.get(c) {
                    value.clone()
                } else {
                    todo!();
                }
            }
            ValueSpecification::HashOfFunction(f) => {
                if let Some(h) = hash_map.get(f) {
                    Rc::new(SExp::QuotedString(srcloc.clone(), b'x', h.clone()))
                } else {
                    todo!();
                }
            }
            ValueSpecification::ApplyFunction(name, val) => {
                let to_run = if let Some(fun_body) = value_map.get(name) {
                    fun_body.clone()
                } else {
                    todo!();
                };

                let args = Rc::new(SExp::Cons(
                    srcloc.clone(),
                    compose_sexp(srcloc.clone(), "2"),
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        val.interpret(opts.clone(), srcloc, value_map, hash_map),
                        Rc::new(SExp::Nil(srcloc.clone())),
                    )),
                ));

                simple_run(opts, to_run, args).expect("should succeed")
            }
            ValueSpecification::ClvmBinop(op, left, right) => {
                let operator = op.to_sexp(srcloc);
                let left_val = left.interpret(opts.clone(), srcloc, value_map, hash_map);
                let right_val = right.interpret(opts.clone(), srcloc, value_map, hash_map);
                let nil = Rc::new(SExp::Nil(srcloc.clone()));
                let expr = Rc::new(SExp::Cons(
                    srcloc.clone(),
                    operator,
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        Rc::new(primquote(srcloc.clone(), left_val)),
                        Rc::new(SExp::Cons(
                            srcloc.clone(),
                            Rc::new(primquote(srcloc.clone(), right_val)),
                            nil.clone(),
                        )),
                    )),
                ));
                simple_run(opts, expr, nil).expect("should succeed")
            }
        }
    }
}

// Property test of sorts:
//
// Make a few constants with any of these expression types:
// [ ] let or assign form
// [ ] lambda (and a corresponding constant applying it either directly or via map
//     or filter)?
// [X] a few basic operators
// Constants generated one at a time, depending on each other in tiers.
// Constants generated all at once, depending on any arrangement of the others.
//
// Do the above arrangement but adding 1 or 2 functions that depend on the
// constants and each other.

struct ModuleConstantExpectation {
    constants_and_values: BTreeMap<Vec<u8>, Rc<ValueSpecification>>,
    functions: BTreeSet<Vec<u8>>,
    opts: Rc<dyn CompilerOpts>,
    counter: usize,
    loc: Srcloc,
}
impl ModuleConstantExpectation {
    fn new(opts: Rc<dyn CompilerOpts>) -> Self {
        ModuleConstantExpectation {
            constants_and_values: BTreeMap::new(),
            functions: BTreeSet::new(),
            opts,
            counter: 0,
            loc: Srcloc::start("*constant-test*"),
        }
    }

    fn loc(&self) -> Srcloc {
        self.loc.clone()
    }
}

struct FuzzT {}
impl FuzzTypeParams for FuzzT {
    type Tag = Vec<u8>;
    type Expr = Rc<SExp>;
    type Error = GenError;
    type State = ModuleConstantExpectation;
}

struct TestModuleConstantFuzzTopRule {
    another_constant: bool,
}
impl TestModuleConstantFuzzTopRule {
    fn new(c: bool) -> Self {
        TestModuleConstantFuzzTopRule {
            another_constant: c,
        }
    }
}

impl Rule<FuzzT> for TestModuleConstantFuzzTopRule {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        let heritage_list: Vec<String> = heritage.iter().map(|h| h.to_string()).collect();
        if tag != b"top" {
            return Ok(None);
        }

        eprintln!(
            "T rule check {} {idx} term {terminate} {heritage_list:?}",
            decode_string(tag)
        );

        if self.another_constant && !terminate {
            let start_program = compose_sexp(state.loc(), &format!("( (include *standard-cl-23*) (defconst TIMES 3) ${{{idx}:constant}} . ${{{}:constant-program-tail}})", idx + 1));
            state.constants_and_values.insert(
                b"TIMES".to_vec(),
                Rc::new(ValueSpecification::ConstantValue(compose_sexp(
                    state.loc(),
                    "3",
                ))),
            );
            Ok(Some(start_program.clone()))
        } else {
            let start_program = compose_sexp(state.loc(), &format!("( (include *standard-cl-23*) (defconst A ${{{idx}:constant-body}}) (export A))"));
            Ok(Some(start_program.clone()))
        }
    }
}

struct TestModuleConstantFuzzConstantBodyRule {}
impl TestModuleConstantFuzzConstantBodyRule {
    fn new() -> Self {
        TestModuleConstantFuzzConstantBodyRule {}
    }
}

fn get_constant_id(heritage: &[Rc<SExp>]) -> Option<Vec<u8>> {
    if heritage.len() < 2 {
        return None;
    }

    if let SExp::Cons(_, c, _) = heritage[heritage.len() - 2].borrow() {
        if let SExp::Atom(_, a) = c.borrow() {
            return Some(a.to_vec());
        }
    }

    None
}

impl Rule<FuzzT> for TestModuleConstantFuzzConstantBodyRule {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        let heritage_list: Vec<String> = heritage.iter().map(|h| h.to_string()).collect();
        eprintln!(
            "C rule check {} {idx} term {terminate} {heritage_list:?}",
            decode_string(tag)
        );

        if tag != b"constant-body" {
            return Ok(None);
        }

        let body = compose_sexp(state.loc(), "1");
        let constant_id = if let Some(constant_id) = get_constant_id(heritage) {
            constant_id
        } else {
            return Ok(None);
        };

        state.constants_and_values.insert(
            constant_id.clone(),
            Rc::new(ValueSpecification::ConstantValue(body.clone())),
        );

        Ok(Some(body.clone()))
    }
}

struct TestModuleConstantFuzzApplyOperation {
    op: SupportedOperators,
    other_value: Rc<SExp>,
}
impl Rule<FuzzT> for TestModuleConstantFuzzApplyOperation {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        let heritage_list: Vec<String> = heritage.iter().map(|h| h.to_string()).collect();
        eprintln!("ApplyOperation {} {heritage_list:?}", decode_string(tag));
        if tag != b"constant-body" {
            return Ok(None);
        }

        let my_id = if let Some(constant_id) = get_constant_id(heritage) {
            constant_id
        } else {
            let heritage_list: Vec<String> = heritage.iter().map(|h| h.to_string()).collect();
            eprintln!("bad constant-body in ApplyOperation: {heritage_list:?}");
            return Ok(None);
        };

        // Get the existing constants.
        let constants_with_values =
            if let Ok(Some(c)) = find_all_constants(state.opts.clone(), heritage, false) {
                c
            } else {
                return Ok(None);
            };

        // No constants are finished.
        if constants_with_values.is_empty() {
            return Ok(None);
        }

        // Choose a constant to base it on.
        let choice = state.counter % constants_with_values.len();
        state.counter += 13;
        let chosen: &DefconstData = &constants_with_values[choice];

        let new_value = Rc::new(ValueSpecification::ClvmBinop(
            self.op.clone(),
            Rc::new(ValueSpecification::ConstantValue(self.other_value.clone())),
            Rc::new(ValueSpecification::ConstantRef(chosen.name.clone())),
        ));
        state
            .constants_and_values
            .insert(my_id.clone(), new_value.clone());

        // Add one to any constant that already has a body.
        Ok(Some(new_value.to_sexp(&state.loc())))
    }
}

struct TestModuleConstantFuzzMoreConstants {
    want_more: bool,
}
impl Rule<FuzzT> for TestModuleConstantFuzzMoreConstants {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"constant-program-tail" {
            return Ok(None);
        }

        if !terminate && self.want_more {
            let body = compose_sexp(
                state.loc(),
                &format!(
                    "(${{{}:constant}} . ${{{}:constant-program-tail}})",
                    idx,
                    idx + 1
                ),
            );
            Ok(Some(body))
        } else {
            let body = compose_sexp(state.loc(), &format!("${{{}:exports}}", idx));
            Ok(Some(body))
        }
    }
}

struct TestModuleConstantMoreFunctions {}
impl Rule<FuzzT> for TestModuleConstantMoreFunctions {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"constant-program-tail" || terminate {
            return Ok(None);
        }

        let name = format!("F{idx}");
        let body = compose_sexp(state.loc(), &format!("((defun {name} (TIMES X) (if (not (> () TIMES)) (+ X ${{{}:function-or-arg}}) X)) . ${{{}:constant-program-tail}})", idx+1, idx+2));
        state.functions.insert(name.as_bytes().to_vec());
        Ok(Some(body))
    }
}

struct TestModuleConstantTerminateFunctionBody {}
impl Rule<FuzzT> for TestModuleConstantTerminateFunctionBody {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"function-or-arg" {
            return Ok(None);
        }

        return Ok(Some(compose_sexp(state.loc(), "1")));
    }
}

struct TestModuleConstantTerminateProgramTail {}
impl Rule<FuzzT> for TestModuleConstantTerminateProgramTail {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"constant-program-tail" {
            return Ok(None);
        }

        eprintln!("terminate constant tail");
        Ok(Some(compose_sexp(
            state.loc(),
            &format!("${{{idx}:exports}}"),
        )))
    }
}

struct TestModuleConstantFunctionOrConstant {
    function: bool,
}
impl Rule<FuzzT> for TestModuleConstantFunctionOrConstant {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"function-or-arg" || terminate {
            return Ok(None);
        }

        let my_id = if let Some(constant_id) = get_constant_id(heritage) {
            constant_id
        } else {
            let heritage_list: Vec<String> = heritage.iter().map(|h| h.to_string()).collect();
            eprintln!("bad constant-body in ApplyOperation: {heritage_list:?}");
            return Ok(None);
        };

        let all_constants = if let Ok(Some(all_constants)) =
            find_all_constants(state.opts.clone(), heritage, false)
        {
            all_constants
        } else {
            return Ok(None);
        };

        if all_constants.is_empty() {
            return Ok(None);
        }

        let constant_choice = state.counter % all_constants.len();
        state.counter += 13;
        let constant_chosen: &DefconstData = &all_constants[constant_choice];
        let mut my_value = Rc::new(ValueSpecification::ConstantRef(
            constant_chosen.name.clone(),
        ));
        let mut body = compose_sexp(state.loc(), &decode_string(&constant_chosen.name));

        if self.function {
            let all_functions =
                if let Ok(Some(all_functions)) = find_all_functions(state.opts.clone(), heritage) {
                    all_functions
                } else {
                    return Ok(None);
                };

            // Choose a constant to base it on.
            let choice = state.counter % all_functions.len();
            state.counter += 13;
            let chosen: &DefunData = &all_functions[choice];
            body = Rc::new(SExp::Cons(
                state.loc(),
                Rc::new(SExp::Atom(state.loc(), chosen.name.clone())),
                Rc::new(SExp::Cons(
                    state.loc(),
                    compose_sexp(state.loc(), "(- TIMES 1)"),
                    Rc::new(SExp::Cons(
                        state.loc(),
                        body,
                        Rc::new(SExp::Nil(state.loc())),
                    )),
                )),
            ));
            my_value = Rc::new(ValueSpecification::ApplyFunction(
                chosen.name.clone(),
                my_value,
            ));
        }

        state.constants_and_values.insert(my_id, my_value);

        return Ok(Some(body));
    }
}

fn constant_body_unexpanded(body: Rc<BodyForm>) -> bool {
    if let BodyForm::Value(SExp::Atom(_, name)) = body.borrow() {
        name.starts_with(b"${") && name.ends_with(b"}")
    } else {
        false
    }
}

fn find_all_members<F, T, E>(
    opts: Rc<dyn CompilerOpts>,
    pred: F,
    heritage: &[Rc<SExp>],
) -> Result<Option<Vec<T>>, E>
where
    F: Fn(&HelperForm) -> Result<Option<T>, E>,
{
    // From the toplevel, scan for defconst forms and get their names.
    // Reject any that aren't fully expanded.
    if heritage.is_empty() {
        return Ok(None);
    }

    let mut result = Vec::new();

    let mut program = heritage[0].clone();
    while let SExp::Cons(_, p, tail) = program.clone().borrow() {
        program = tail.clone();

        eprintln!("try to compile: {p}");
        if let Ok(Some(res)) = compile_helperform(opts.clone(), p.clone()) {
            for h in res.new_helpers.iter() {
                if let Some(res) = pred(h)? {
                    result.push(res);
                }
            }
        }
    }

    Ok(Some(result))
}

fn find_all_constants(
    opts: Rc<dyn CompilerOpts>,
    heritage: &[Rc<SExp>],
    abort_on_unexpanded: bool,
) -> Result<Option<Vec<DefconstData>>, ()> {
    find_all_members(
        opts,
        |h| {
            if let HelperForm::Defconstant(dc) = h {
                if constant_body_unexpanded(dc.body.clone()) {
                    if abort_on_unexpanded {
                        return Err(());
                    } else {
                        return Ok(None);
                    }
                }

                Ok(Some(dc.clone()))
            } else {
                Ok(None)
            }
        },
        heritage,
    )
}

fn find_all_functions(
    opts: Rc<dyn CompilerOpts>,
    heritage: &[Rc<SExp>],
) -> Result<Option<Vec<DefunData>>, ()> {
    find_all_members(
        opts,
        |h| {
            if let HelperForm::Defun(false, defun) = h {
                return Ok(Some(*defun.clone()));
            }

            Ok(None)
        },
        heritage,
    )
}

// Only do exports if it's the only replacement left
fn only_export_left(heritage: &[Rc<SExp>]) -> bool {
    if heritage.is_empty() {
        return true;
    }

    let mut waiters = Vec::new();
    heritage[0].find_waiters(&mut waiters);

    waiters.len() == 1
}

struct TestModuleConstantFuzzExports {}
impl Rule<FuzzT> for TestModuleConstantFuzzExports {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"exports" || !only_export_left(heritage) {
            return Ok(None);
        }

        let nil = Rc::new(SExp::Nil(state.loc()));

        let mut result = nil.clone();
        let export_names = |result: &mut Rc<SExp>, names: &[String]| {
            for h in names.iter() {
                if h == "X" {
                    continue;
                }
                *result = Rc::new(SExp::Cons(
                    state.loc(),
                    compose_sexp(state.loc(), &format!("(export {h})")),
                    result.clone(),
                ));
            }
        };

        let have_constants: Vec<String> = state
            .constants_and_values
            .keys()
            .map(|n| decode_string(n))
            .collect();
        export_names(&mut result, &have_constants);

        let have_functions: Vec<String> =
            state.functions.iter().map(|n| decode_string(n)).collect();
        eprintln!("have functions: {have_functions:?}");
        export_names(&mut result, &have_functions);

        Ok(Some(result))
    }
}

struct TestModuleConstantNew {}
impl Rule<FuzzT> for TestModuleConstantNew {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"constant" {
            return Ok(None);
        }

        let name = format!("C{idx}");
        let body = compose_sexp(
            state.loc(),
            &format!("(defconst {name} ${{{}:constant-body}})", idx),
        );
        let heritage_list: Vec<String> = heritage.iter().map(|h| h.to_string()).collect();
        eprintln!("CN {heritage_list:?}");
        Ok(Some(body.clone()))
    }
}

struct TestModuleConstantBasedOnFunctionHash {}
impl Rule<FuzzT> for TestModuleConstantBasedOnFunctionHash {
    fn check(
        &self,
        state: &mut ModuleConstantExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"constant-body" {
            return Ok(None);
        }

        let my_id = if let Some(constant_id) = get_constant_id(heritage) {
            constant_id
        } else {
            eprintln!("bad constant-body in BasedOnFunctionHash");
            return Ok(None);
        };

        let all_functions =
            if let Ok(Some(all_functions)) = find_all_functions(state.opts.clone(), heritage) {
                all_functions
            } else {
                return Ok(None);
            };

        if all_functions.is_empty() {
            return Ok(None);
        }

        // Choose a constant to base it on.
        let choice = state.counter % all_functions.len();
        state.counter += 13;
        let chosen: &DefunData = &all_functions[choice];
        let value = Rc::new(ValueSpecification::HashOfFunction(chosen.name.clone()));
        state.constants_and_values.insert(my_id, value.clone());
        Ok(Some(value.to_sexp(&state.loc())))
    }
}

#[test]
fn test_property_fuzz_stable_constants() {
    let mut rng = ChaCha8Rng::from_seed([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2,
    ]);

    let srcloc = Srcloc::start("*value*");
    let one = Rc::new(SExp::Integer(srcloc.clone(), bi_one()));
    let rules: Vec<Rc<dyn Rule<FuzzT>>> = vec![
        Rc::new(TestModuleConstantFuzzTopRule::new(false)),
        Rc::new(TestModuleConstantFuzzTopRule::new(true)),
        Rc::new(TestModuleConstantFuzzConstantBodyRule::new()),
        Rc::new(TestModuleConstantNew {}),
        Rc::new(TestModuleConstantFuzzMoreConstants { want_more: true }),
        Rc::new(TestModuleConstantFuzzMoreConstants { want_more: false }),
        Rc::new(TestModuleConstantFuzzExports {}),
        Rc::new(TestModuleConstantFuzzApplyOperation {
            op: SupportedOperators::Plus,
            other_value: one.clone(),
        }),
        Rc::new(TestModuleConstantFuzzApplyOperation {
            op: SupportedOperators::Minus,
            other_value: one.clone(),
        }),
        Rc::new(TestModuleConstantMoreFunctions {}),
        Rc::new(TestModuleConstantFunctionOrConstant { function: false }),
        Rc::new(TestModuleConstantFunctionOrConstant { function: true }),
        Rc::new(TestModuleConstantTerminateFunctionBody {}),
        Rc::new(TestModuleConstantTerminateProgramTail {}),
        Rc::new(TestModuleConstantBasedOnFunctionHash {}),
    ];
    let top_node = Rc::new(SExp::Atom(srcloc.clone(), b"${0:top}".to_vec()));

    for _ in 0..30 {
        let mut fuzzgen = FuzzGenerator::new(top_node.clone(), &rules);
        let mut idx = 0;
        let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*test*"));
        let mut mc = ModuleConstantExpectation::new(
            opts.set_dialect(AcceptedDialect {
                stepping: Some(23),
                strict: true,
                int_fix: true,
            })
            .set_optimize(true),
        );
        while fuzzgen
            .expand(&mut mc, idx > 15, &mut rng)
            .expect("should expand")
        {
            idx += 1;
            assert!(idx < 100);
        }

        let forms = if let Some(flist) = fuzzgen.result().proper_list() {
            flist
        } else {
            todo!();
        };

        let program_text = forms
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        eprintln!("program_text\n{program_text}");
        let mut allocator = Allocator::new();
        let runner = Rc::new(DefaultProgramRunner::new());
        let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("test.clsp"))
            .set_search_paths(&["resources/tests/module".to_string()]);
        let source_opts = TestModuleCompilerOpts::new(orig_opts);
        let compiled = perform_compile_of_file(
            &mut allocator,
            runner.clone(),
            source_opts,
            "test.clsp",
            &program_text,
        )
        .expect("should compile");

        // Collect output values from compiled.
        let mut actual_values = BTreeMap::new();
        let mut hashes = BTreeMap::new();
        if let CompilerOutput::Module(x) = &compiled.compiled {
            for component in x.components.iter() {
                eprintln!(
                    "{} ({}) = {}",
                    decode_string(&component.shortname),
                    SExp::QuotedString(mc.loc(), b'x', component.hash.clone()),
                    component.content
                );
                actual_values.insert(component.shortname.clone(), component.content.clone());
                hashes.insert(component.shortname.clone(), component.hash.clone());
            }
        } else {
            assert!(false);
        };

        for cname in mc.constants_and_values.keys().filter(|k| *k != b"X") {
            if let Some(cval) = mc.constants_and_values.get(cname) {
                let desired_value =
                    cval.interpret(mc.opts.clone(), &mc.loc(), &actual_values, &hashes);
                eprintln!("checking for {} = {desired_value}", decode_string(cname));
                let hex_file_name = format!("test_{}.hex", decode_string(cname));
                let hex_data = compiled
                    .source_opts
                    .get_written_file(&hex_file_name)
                    .expect("hex file should exist");
                let compiled_node = hex_to_clvm(&mut allocator, &hex_data);
                let converted_node =
                    convert_from_clvm_rs(&mut allocator, desired_value.loc(), compiled_node)
                        .expect("should convert to sexp objects");
                assert_eq!(*desired_value, *converted_node);
            } else {
                todo!();
            }
        }
    }

    // We've checked all predicted values.
}
