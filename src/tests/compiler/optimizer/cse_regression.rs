use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::compiler::{DefaultCompilerOpts, FUZZ_TEST_PRE_CSE_MERGE_FIX_FLAG};
use crate::compiler::comptypes::{BodyForm, CompileForm, CompilerOpts, DefunData, HelperForm};
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::BasicCompileContext;

use crate::tests::compiler::fuzz::{compose_sexp, simple_seeded_rng};
use crate::tests::compiler::fuzz_assign::{
    create_complex_assign_expression, create_variable_set, GeneratedExpr,
};

// Produce a program that provides a valid regression test for the cse merge fix.
fn produce_valid_cse_regression_merge_test<R: Rng>(
    srcloc: &Srcloc,
    rng: &mut R,
) -> Option<CompileForm> {
    // Strategy:
    let vars = create_variable_set(srcloc.clone(), 7);

    // Generate a definition graph including assign forms with fresh variables.
    let structure_graph = create_complex_assign_expression(rng, &vars);

    // Generate fresh argument variables.
    let args: Vec<Vec<u8>> = vec![b"a1".to_vec()];

    // Get the generated variable graph.
    eprintln!("vars\n{structure_graph:?}");

    // For each variable in the graph, generate some candidate expressions to
    // define it.
    let candidate_definitions: BTreeMap<Vec<u8>, GeneratedExpr> = structure_graph
        .bindings
        .keys()
        .map(|k| {
            (
                k.clone(),
                structure_graph.generate_expression(&srcloc, 10, rng, &args, k),
            )
        })
        .collect();

    let body = structure_graph.create_assign_form(srcloc, &candidate_definitions);
    let args = compose_sexp(srcloc.clone(), "(a1)");
    let function = HelperForm::Defun(
        false,
        Box::new(DefunData {
            loc: srcloc.clone(),
            nl: srcloc.clone(),
            kw: None,
            name: b"defined-fun".to_vec(),
            orig_args: args.clone(),
            args: args.clone(),
            synthetic: None,
            body: Rc::new(body),
        }),
    );

    Some(CompileForm {
        loc: srcloc.clone(),
        args: args,
        helpers: vec![function],
        include_forms: vec![],
        exp: Rc::new(BodyForm::Call(
            srcloc.clone(),
            vec![
                Rc::new(BodyForm::Value(SExp::Atom(
                    srcloc.clone(),
                    b"defined-fun".to_vec(),
                ))),
                Rc::new(BodyForm::Value(SExp::Atom(srcloc.clone(), b"a1".to_vec()))),
            ],
            None,
        )),
    })
}

#[test]
fn test_cse_merge_regression() {
    let mut rng = simple_seeded_rng(1);
    let srcloc = Srcloc::start("*test*");

    let produce_program = |rng: &mut ChaCha8Rng| loop {
        if let Some(res) = produce_valid_cse_regression_merge_test(&srcloc, rng) {
            return res;
        }
    };

    for _ in 0..20 {
        let test_program = produce_program(&mut rng);
        let program_sexp = Rc::new(SExp::Cons(
            srcloc.clone(),
            Rc::new(SExp::Atom(srcloc.clone(), b"mod".to_vec())),
            test_program.to_sexp(),
        ));

        eprintln!("test_program {program_sexp}");
        let dialect = AcceptedDialect {
            stepping: Some(23),
            strict: true,
            int_fix: false,
        };
        let new_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("test.clsp"))
            .set_dialect(dialect.clone())
            .set_optimize(true)
            .set_frontend_opt(false);
        let old_flags: Rc<HashSet<usize>> = Rc::new(
            (&[FUZZ_TEST_PRE_CSE_MERGE_FIX_FLAG])
                .iter()
                .copied()
                .collect(),
        );
        let old_opts: Rc<dyn CompilerOpts> =
            Rc::new(DefaultCompilerOpts::new("merge-fix-regression-test.clsp"))
                .set_dialect(dialect.clone())
                .set_optimize(true)
                .set_frontend_opt(false)
                .set_diag_flags(old_flags);
        let runner = Rc::new(DefaultProgramRunner::new());
        let opt_err_loc = Srcloc::start(&new_opts.filename());
        let new_compiled = {
            let mut context = BasicCompileContext::new(
                Allocator::new(),
                runner.clone(),
                HashMap::new(),
                get_optimizer(&opt_err_loc, new_opts.clone()).unwrap(),
                Vec::new(),
            );
            new_opts
                .compile_program(&mut context, program_sexp.clone())
                .expect("should compile (new)")
        };

        let old_compiled = {
            let mut context = BasicCompileContext::new(
                Allocator::new(),
                runner,
                HashMap::new(),
                get_optimizer(&opt_err_loc, new_opts.clone()).unwrap(),
                Vec::new(),
            );
            old_opts
                .compile_program(&mut context, program_sexp)
                .expect("should compile (old)")
        };

        assert_eq!(new_compiled.to_sexp(), old_compiled.to_sexp());
    }
}
