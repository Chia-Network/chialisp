use std::collections::HashMap;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;

use crate::compiler::clvm::run;
use crate::compiler::codegen::codegen;
use crate::compiler::compiler::{do_desugar, DefaultCompilerOpts};
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompilerOpts, DefconstData, Export, FrontendOutput,
    HelperForm,
};
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::above22::Strategy23;
use crate::compiler::optimize::depgraph::{DepgraphOptions, FunctionDependencyGraph};
use crate::compiler::rename::rename_args_compileform;
use crate::compiler::sexp::{decode_string, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::{BasicCompileContext, Funcache, FunctionEntry};
use crate::tests::compiler::modules::TestModuleCompilerOpts;

fn get_expected_outcome(
    opts: Rc<dyn CompilerOpts>,
    run_arg: Rc<SExp>,
) -> Result<Rc<SExp>, CompileErr> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let ff_program_base = parse_sexp(
        Srcloc::start("*run-ff*"),
        "(sha256 (+ 2 (q . 33)) 5 (concat 2 5))".bytes(),
    )
    .unwrap()[0]
        .clone();
    let gg_program_base = parse_sexp(
        Srcloc::start("*run-gg*"),
        "(sha256 (sha256 2 (q . 99)) (sha256 (q . 1) 5 (q . 99)))".bytes(),
    )
    .unwrap()[0]
        .clone();
    let ff_base_outcome = run(
        &mut allocator,
        runner.clone(),
        opts.prim_map(),
        ff_program_base,
        run_arg.clone(),
        None,
        None,
    )?;
    let gg_base_outcome = run(
        &mut allocator,
        runner.clone(),
        opts.prim_map(),
        gg_program_base,
        run_arg.clone(),
        None,
        None,
    )?;
    Ok(Rc::new(SExp::Cons(
        run_arg.loc(),
        ff_base_outcome.clone(),
        gg_base_outcome.clone(),
    )))
}

fn extract_helper_gen(map: &HashMap<Vec<u8>, FunctionEntry>, name: &str) -> Option<Rc<SExp>> {
    map.values()
        .filter(|v| decode_string(&v.name) == name)
        .next()
        .map(|v| v.code.clone())
}

fn flip_inline(h: &HelperForm, toward: bool) -> Option<HelperForm> {
    if let HelperForm::Defun(_inl, defdata) = h {
        return Some(HelperForm::Defun(toward, defdata.clone()));
    }
    None
}

fn match_name_prefix(prefix: &str, h: &HelperForm) -> bool {
    decode_string(h.name()).starts_with(prefix)
}

fn untable_constant(h: &HelperForm) -> Option<HelperForm> {
    if let HelperForm::Defconstant(defc) = h {
        return Some(HelperForm::Defconstant(DefconstData {
            tabled: false,
            ..defc.clone()
        }));
    }
    None
}

fn change_constant_value(h: &HelperForm, body: Rc<BodyForm>) -> Option<HelperForm> {
    if let HelperForm::Defconstant(defc) = h {
        return Some(HelperForm::Defconstant(DefconstData {
            body,
            ..defc.clone()
        }));
    }
    None
}

fn transform_helper<F: Fn(&HelperForm) -> Option<HelperForm>>(
    compileform: &mut CompileForm,
    transform: F,
) {
    for h in compileform.helpers.iter_mut() {
        if let Some(res) = transform(&h) {
            eprintln!("{} became {}", h.to_sexp(), res.to_sexp());
            *h = res;
        }
    }
}

fn diffmap<K: Eq + std::hash::Hash, V>(m1: &mut HashMap<K, V>, m2: &HashMap<K, V>) {
    for key in m2.keys() {
        m1.remove(key);
    }
}

#[test]
fn test_codegen_function_cache() {
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(&"*test*".to_string()));
    let orig_opts = orig_opts
        .set_search_paths(&["resources/tests/module".to_string(), ".".to_string()])
        .set_stdenv(false)
        .set_dialect(AcceptedDialect {
            stepping: Some(25),
            strict: true,
            int_fix: true,
            extra_numeric_constants: false,
        });
    let fs_opts = TestModuleCompilerOpts::new(orig_opts);
    let opts: Rc<dyn CompilerOpts> = Rc::new(fs_opts.clone());
    let (_, content) = opts
        .read_new_file(
            opts.filename(),
            "resources/tests/module/cache-test-1.clsp".to_string(),
        )
        .expect("ok");
    let parsed = parse_sexp(Srcloc::start(&opts.filename()), content.iter().cloned()).expect("ok");
    let program = frontend(opts.clone(), &parsed).expect("ok");
    let (mut compileform, exports) = if let FrontendOutput::Module(compileform, exports) = program {
        (compileform, exports)
    } else {
        panic!();
    };

    // Module style programs produce exports that are handled separately.  They're turned into full
    // programs in this way:
    (compileform.args, compileform.exp) = if let Export::MainProgram(ep) = &exports[0] {
        (ep.args.clone(), ep.expr.clone())
    } else {
        panic!();
    };
    compileform = rename_args_compileform(&compileform).unwrap();

    //
    // Now we've got a compileform with some exports and some intermediate functions.
    // check-1 depends on FF which depends on F, which depends on C
    // check-2 depends on GG which depends on G and H, which depend on CC
    // We should be able to change FF without changing the output of and G or H without
    // chaning the output of GG unless the environment changes.
    // We should be able to change C without changing the output of GG and CC without
    // changing the output of FF.
    //
    // conversely, we should always pick up a change in C or F in FF.
    // and CC, G, H in GG.
    //
    let runner = Rc::new(DefaultProgramRunner::new());
    let mut context = BasicCompileContext::new(
        Allocator::new(),
        runner.clone(),
        HashMap::new(),
        Box::new(Strategy23 {}),
    );

    let mut desugared = do_desugar(opts.clone(), &compileform).unwrap();

    // This enables caching.
    context.funcache = Some(Funcache::default());

    // Set the constant CC to not be tabled so we can test inlined constant changes.
    transform_helper(&mut desugared, |h| {
        if match_name_prefix("CC", h) {
            return untable_constant(h);
        }
        None
    });

    eprintln!("compileform {}", desugared.to_sexp());

    let dependency_graph = FunctionDependencyGraph::new_with_options(
        &desugared,
        DepgraphOptions {
            with_constants: true,
        },
    );

    let base_generated = Rc::new(
        codegen(
            &mut context,
            opts.clone(),
            Some(&dependency_graph),
            &desugared,
        )
        .unwrap(),
    );
    eprintln!("generated code {base_generated}");

    let original_map = context.funcache.as_ref().unwrap().function_outputs.clone();
    let mut allocator = Allocator::new();
    let run_arg = parse_sexp(Srcloc::start("*args*"), b"(3 7)".iter().cloned()).unwrap()[0].clone();
    eprintln!("use args {}", run_arg);
    let expected_outcome = get_expected_outcome(opts.clone(), run_arg.clone()).unwrap();
    let run_outcome_base = run(
        &mut allocator,
        runner.clone(),
        opts.prim_map(),
        base_generated.clone(),
        run_arg.clone(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(run_outcome_base, expected_outcome);

    // Ensure the cache is being used.  We should have entries for FF, GG.
    let want_set = vec!["FF", "GG", "F", "G", "H"];
    let names: Vec<_> = context
        .funcache
        .as_ref()
        .map(|fc| {
            fc.function_outputs
                .values()
                .map(|e| decode_string(&e.name))
                .collect()
        })
        .unwrap_or_default();
    // maps names to their original hashes.
    let original_name_map: HashMap<_, _> = context
        .funcache
        .as_ref()
        .map(|fc| {
            fc.function_outputs
                .iter()
                .map(|(k, v)| (decode_string(&v.name), k.to_vec()))
                .collect()
        })
        .unwrap_or_default();

    // A let binding will have been desugared from FF.
    // Show all the expected names are present in the cache along with the desugared assignment.
    assert!(!names.is_empty());
    assert_eq!(names.len(), want_set.len() + 1);
    assert_eq!(names.len(), original_name_map.len());

    for name in want_set.iter() {
        assert!(names.iter().any(|n| n == name));
    }

    let mut flipped_letbinding_program = desugared.clone();

    transform_helper(&mut flipped_letbinding_program, |h| {
        if match_name_prefix("letbinding_$_", h) {
            return flip_inline(h, true);
        }
        None
    });

    // If the code wasn't regenerated in response to this, we'd have the same program.
    let generated_flipped_let_binding = Rc::new(
        codegen(
            &mut context,
            opts.clone(),
            Some(&dependency_graph),
            &flipped_letbinding_program,
        )
        .unwrap(),
    );
    let mut flipped_map = context.funcache.as_ref().unwrap().function_outputs.clone();
    diffmap(&mut flipped_map, &original_map);

    assert_ne!(
        base_generated.to_string(),
        generated_flipped_let_binding.to_string()
    );

    // Extract the generated FF and verify that it's different.
    let old_ff = extract_helper_gen(&original_map, "FF").unwrap();
    let new_ff = extract_helper_gen(&flipped_map, "FF").unwrap();

    let old_h = extract_helper_gen(&original_map, "H").unwrap();

    // GG will have been different here because of the environment shift.
    // We'll keep a representation of it to know for sure.
    let old_gg = extract_helper_gen(&original_map, "GG").unwrap();
    let new_gg = extract_helper_gen(&flipped_map, "GG").unwrap();

    // They aren't the same function code.
    assert_ne!(old_ff, new_ff);

    // Verify the same run outcome.
    let run_outcome_flipped = run(
        &mut allocator,
        runner.clone(),
        opts.prim_map(),
        generated_flipped_let_binding.clone(),
        run_arg.clone(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(expected_outcome, run_outcome_flipped);

    // Swap dummy and the let binding.  That should leave the environment in the same shape.
    // If we've done things correctly, GG should be the same function in generated_flipped_let
    // as in generated_double_flip.
    let mut fake_flip_program = flipped_letbinding_program.clone();
    transform_helper(&mut fake_flip_program, |h| {
        if match_name_prefix("fake-letbinding", h) {
            return flip_inline(h, false);
        }
        None
    });

    let generated_double_flip = Rc::new(
        codegen(
            &mut context,
            opts.clone(),
            Some(&dependency_graph),
            &fake_flip_program,
        )
        .unwrap(),
    );

    // We'll zap the cache and do it again.  The version we get for GG should be the same as in
    // the original because we transformed the environment back to the same shape observed in the
    // original while involving only functions that GG doesn't use.
    context.funcache = Some(Funcache::default());
    let generated_double_flip_clean_cache = Rc::new(
        codegen(
            &mut context,
            opts.clone(),
            Some(&dependency_graph),
            &fake_flip_program,
        )
        .unwrap(),
    );

    // They're the same program, so we can use the clean cache below to check generated bodies.
    assert_eq!(generated_double_flip, generated_double_flip_clean_cache);

    let fake_flip_map_clean_cache = context.funcache.as_ref().unwrap().function_outputs.clone();

    let swap_ff = extract_helper_gen(&fake_flip_map_clean_cache, "FF").unwrap();
    let swap_gg = extract_helper_gen(&fake_flip_map_clean_cache, "GG").unwrap();

    // FF should not be the same
    assert_ne!(swap_ff, old_ff);
    // GG should be the same, so we removed it by filtering.
    assert_eq!(swap_gg, old_gg);
    assert_ne!(swap_gg, new_gg);

    // Modify a constant affecting H.
    let mut constant_diff_program = desugared.clone();
    transform_helper(&mut constant_diff_program, |h| {
        if match_name_prefix("CC", h) {
            return change_constant_value(
                h,
                Rc::new(BodyForm::Quoted(SExp::Atom(h.loc(), vec![37]))),
            );
        }
        None
    });

    let generated_diff_c = Rc::new(
        codegen(
            &mut context,
            opts.clone(),
            Some(&dependency_graph),
            &constant_diff_program,
        )
        .unwrap(),
    );

    // Clobber cache and generate again so we can see exactly what was generated.
    context.funcache = Some(Funcache::default());
    let generated_diff_c_clean = Rc::new(
        codegen(
            &mut context,
            opts.clone(),
            Some(&dependency_graph),
            &constant_diff_program,
        )
        .unwrap(),
    );
    let generated_diff_c_map_clean = context.funcache.as_ref().unwrap().function_outputs.clone();
    assert_eq!(generated_diff_c, generated_diff_c_clean);

    // // Now we have a different value of GG, but same value of FF.
    let diffcc_ff = extract_helper_gen(&generated_diff_c_map_clean, "FF").unwrap();
    let diffcc_gg = extract_helper_gen(&generated_diff_c_map_clean, "GG").unwrap();
    let diffcc_h = extract_helper_gen(&generated_diff_c_map_clean, "H").unwrap();
    eprintln!("new H {diffcc_h}");

    assert_eq!(diffcc_ff, old_ff);
    assert_eq!(diffcc_gg, old_gg);
    assert_ne!(diffcc_h, old_h);
}
