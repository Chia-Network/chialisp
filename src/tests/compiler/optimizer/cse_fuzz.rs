use rand::Rng;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::fuzz::{FuzzTypeParams, Rule};
use crate::compiler::sexp::{AtomValue, NodeSel, SExp, SelectNode, ThisNode};
use crate::compiler::srcloc::Srcloc;

use crate::tests::compiler::fuzz::{
    compose_sexp, simple_seeded_rng, GenError, HasVariableStore, PropertyTest, PropertyTestRun,
    PropertyTestState, SupportedOperators, ValueSpecification,
};

struct TrickyAssignExpectation {
    opts: Rc<dyn CompilerOpts>,
    loc: Srcloc,
    count: usize,
    actions: usize,
    final_var: Option<Rc<ValueSpecification>>,
    var_defs: BTreeMap<Vec<u8>, Rc<ValueSpecification>>,
}

impl TrickyAssignExpectation {
    fn new(opts: Rc<dyn CompilerOpts>) -> Self {
        TrickyAssignExpectation {
            opts,
            loc: Srcloc::start("*constant-test*"),
            count: 0,
            actions: 0,
            final_var: None,
            var_defs: BTreeMap::default(),
        }
    }

    fn loc(&self) -> Srcloc {
        self.loc.clone()
    }

    fn compute(&self) -> Rc<SExp> {
        let val = self.final_var.as_ref().unwrap();
        val.interpret(self.opts.clone(), &self.loc, self)
    }
}

impl HasVariableStore for TrickyAssignExpectation {
    fn get(&self, name: &[u8]) -> Option<Rc<ValueSpecification>> {
        self.var_defs.get(name).cloned()
    }
}

struct FuzzT;
impl FuzzTypeParams for FuzzT {
    type Tag = Vec<u8>;
    type Expr = Rc<SExp>;
    type Error = GenError;
    type State = TrickyAssignExpectation;
}

struct TestTrickyAssignFuzzTopRule {
    defs: usize,
}
impl Rule<FuzzT> for TestTrickyAssignFuzzTopRule {
    fn check(
        &self,
        state: &mut TrickyAssignExpectation,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"top" {
            return Ok(None);
        }

        state.count = self.defs;
        let start_program = compose_sexp(state.loc(), &format!("(mod (X) (include *standard-cl-23*) (defun F (X) (assign . ${{{}:assign-test-form}})) (F X))", idx));
        Ok(Some(start_program.clone()))
    }
}

struct TestTrickyAssignFuzzTestFormRule;
impl Rule<FuzzT> for TestTrickyAssignFuzzTestFormRule {
    fn check(
        &self,
        state: &mut TrickyAssignExpectation,
        tag: &Vec<u8>,
        idx: usize,
        _terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"assign-test-form" {
            return Ok(None);
        }

        if state.count > 0 {
            let varname = format!("v{}", state.count);
            state.count -= 1;
            Ok(Some(compose_sexp(
                state.loc(),
                &format!(
                    "({varname} ${{{idx}:vardef}} . ${{{}:assign-test-form}})",
                    idx + 1
                ),
            )))
        } else {
            Ok(Some(compose_sexp(
                state.loc(),
                &format!("${{{idx}:final-expr}}"),
            )))
        }
    }
}

fn find_var_name_in_heritage(heritage: &[Rc<SExp>]) -> Vec<u8> {
    let NodeSel::Cons((_varloc, varname), _) = NodeSel::Cons(AtomValue::Here(()), ThisNode)
        .select_nodes(heritage[heritage.len() - 2].clone())
        .unwrap();
    varname.clone()
}

struct TestTrickyAssignVarDefConstantRule {
    value: Rc<SExp>,
}
impl Rule<FuzzT> for TestTrickyAssignVarDefConstantRule {
    fn check(
        &self,
        state: &mut TrickyAssignExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"vardef" || heritage.len() < 3 {
            return Ok(None);
        }

        state.actions += 13;
        let my_name = find_var_name_in_heritage(heritage);

        let spec = Rc::new(ValueSpecification::ConstantValue(self.value.clone()));
        state.var_defs.insert(my_name.clone(), spec);
        Ok(Some(self.value.clone()))
    }
}

struct TestTrickyAssignVarDefBinopRule {
    op: SupportedOperators,
    other: Rc<SExp>,
}
impl Rule<FuzzT> for TestTrickyAssignVarDefBinopRule {
    fn check(
        &self,
        state: &mut TrickyAssignExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"vardef" || state.var_defs.is_empty() || heritage.len() < 3 {
            return Ok(None);
        }

        state.actions += 13;
        let my_name = find_var_name_in_heritage(heritage);

        // We'll choose one other value and compose with the existing value.
        let to_skip = state.actions % state.var_defs.len();
        let (k, _) = state.var_defs.iter().skip(to_skip).next().unwrap();
        let my_value = Rc::new(ValueSpecification::VarRef(k.to_vec()));

        let spec = Rc::new(ValueSpecification::ClvmBinop(
            self.op.clone(),
            Rc::new(ValueSpecification::ConstantValue(self.other.clone())),
            my_value,
        ));
        state.var_defs.insert(my_name.clone(), spec.clone());
        Ok(Some(Rc::new(spec.to_sexp(&state.loc))))
    }
}

struct TestTrickyAssignFinalExpr;
impl Rule<FuzzT> for TestTrickyAssignFinalExpr {
    fn check(
        &self,
        state: &mut TrickyAssignExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"final-expr" || state.var_defs.is_empty() {
            return Ok(None);
        }

        let to_skip = state.actions % state.var_defs.len();
        let (k, _) = state.var_defs.iter().skip(to_skip).next().unwrap();
        state.final_var = Some(Rc::new(ValueSpecification::VarRef(k.to_vec())));
        Ok(Some(Rc::new(SExp::Cons(
            state.loc.clone(),
            Rc::new(SExp::Atom(state.loc.clone(), k.to_vec())),
            Rc::new(SExp::Nil(state.loc.clone())),
        ))))
    }
}

struct TestTrickyAssignFinalBinopRule {
    op: SupportedOperators,
    other: Rc<SExp>,
}
impl Rule<FuzzT> for TestTrickyAssignFinalBinopRule {
    fn check(
        &self,
        state: &mut TrickyAssignExpectation,
        tag: &Vec<u8>,
        _idx: usize,
        _terminate: bool,
        _heritage: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, GenError> {
        if tag != b"final-expr" || state.var_defs.is_empty() {
            return Ok(None);
        }

        let to_skip = state.actions % state.var_defs.len();
        let (k, _) = state.var_defs.iter().skip(to_skip).next().unwrap();
        let result_expr = Rc::new(ValueSpecification::ClvmBinop(
            self.op.clone(),
            Rc::new(ValueSpecification::ConstantValue(self.other.clone())),
            Rc::new(ValueSpecification::VarRef(k.to_vec())),
        ));
        state.final_var = Some(result_expr.clone());
        Ok(Some(Rc::new(SExp::Cons(
            state.loc.clone(),
            Rc::new(result_expr.to_sexp(&state.loc)),
            Rc::new(SExp::Nil(state.loc.clone())),
        ))))
    }
}

impl PropertyTestState<FuzzT> for TrickyAssignExpectation {
    fn new_state<R: Rng>(_r: &mut R) -> Self {
        let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*test*"));
        TrickyAssignExpectation::new(
            opts.set_dialect(AcceptedDialect {
                stepping: Some(23),
                strict: true,
                int_fix: false,
            })
            .set_optimize(true),
        )
    }
}
impl PropertyTestRun for TrickyAssignExpectation {
    fn run_args(&self) -> String {
        "(3)".to_string()
    }
    fn check(&self, run_result: Rc<SExp>) {
        let want_result = self.compute();
        eprintln!("run_result {run_result} have {want_result}");
        assert_eq!(run_result, want_result);
    }
}

#[test]
fn test_property_fuzz_cse_binding() {
    let srcloc = Srcloc::start("*value*");
    let mut rng = simple_seeded_rng(0x02020202);
    let test = PropertyTest {
        run_times: 500,
        run_cutoff: 300,
        run_expansion: 20,

        top_node: compose_sexp(srcloc.clone(), "${0:top}"),
        rules: vec![
            Rc::new(TestTrickyAssignFuzzTopRule { defs: 1 }),
            Rc::new(TestTrickyAssignFuzzTopRule { defs: 2 }),
            Rc::new(TestTrickyAssignFuzzTopRule { defs: 3 }),
            Rc::new(TestTrickyAssignFuzzTopRule { defs: 4 }),
            Rc::new(TestTrickyAssignFuzzTopRule { defs: 5 }),
            Rc::new(TestTrickyAssignFuzzTestFormRule),
            Rc::new(TestTrickyAssignVarDefConstantRule {
                value: compose_sexp(srcloc.clone(), "1"),
            }),
            Rc::new(TestTrickyAssignVarDefBinopRule {
                op: SupportedOperators::Times,
                other: compose_sexp(srcloc.clone(), "2"),
            }),
            Rc::new(TestTrickyAssignFinalExpr),
            Rc::new(TestTrickyAssignFinalBinopRule {
                op: SupportedOperators::Times,
                other: compose_sexp(srcloc, "2"),
            }),
        ],
    };

    test.run(&mut rng);
}
