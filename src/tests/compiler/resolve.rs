use std::collections::HashMap;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::run;
use crate::compiler::codegen::codegen;
use crate::compiler::compiler::{do_desugar, DefaultCompilerOpts};
use crate::compiler::comptypes::{CompileErr, CompilerOpts};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::resolve::resolve_namespaces;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::BasicCompileContext;

fn compile_program_get_result(
    test_program: &str,
    test_input: &str,
) -> Result<Rc<SExp>, CompileErr> {
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*resolve-test*"));
    let loc = Srcloc::start("*resolve-test*");
    let parsed = parse_sexp(loc.clone(), test_program.bytes())?;
    let processed = frontend(opts.clone(), &parsed)?;
    let resolved = resolve_namespaces(opts.clone(), &processed.compileform())?;
    let desugared = do_desugar(&resolved)?;
    let mut context = BasicCompileContext {
        allocator: Allocator::new(),
        runner: Rc::new(DefaultProgramRunner::new()),
        symbols: HashMap::new(),
        optimizer: get_optimizer(&loc, opts.clone())?,
    };
    let generated = codegen(&mut context, opts.clone(), &desugared)?;
    run(
        &mut context.allocator,
        context.runner.clone(),
        opts.prim_map(),
        Rc::new(generated),
        parse_sexp(loc.clone(), test_input.bytes())?[0].clone(),
        None,
        None,
    )
    .map_err(|e| e.into())
}

#[test]
fn test_compile_module_with_resolver() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (+ Z Z1.Q)) (defun G (Z) (- Z Z1.Q))) (import X hiding G) (namespace Y (defun G (Z) (* Z 2))) (import Y exposing (G as GG)) (F (GG (+ A (@ 5)))))";
    let outcome = compile_program_get_result(test_program, "(3)").unwrap();
    assert_eq!(outcome.to_string(), "13");
}

#[test]
fn test_helper_not_found() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (import Z exposing Q1) Q1)";
    let outcome = compile_program_get_result(test_program, "(3)");
    assert!(outcome.is_err());
}

#[test]
fn test_resolve_with_let() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (let ((ZZ (+ Z Z1.Q))) ZZ))) (import X hiding G) (F 3))";
    let outcome = compile_program_get_result(test_program, "(3)").unwrap();
    assert_eq!(outcome.to_string(), "4");
}

#[test]
fn test_resolve_with_let_star() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (let* ((ZZ (+ Z Z1.Q))) ZZ))) (import X hiding G) (F 3))";
    let outcome = compile_program_get_result(test_program, "(3)").unwrap();
    assert_eq!(outcome.to_string(), "4");
}

#[test]
fn test_resolve_with_assign() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (assign ZZ (+ Z Z1.Q) ZZ))) (import X hiding G) (F 3))";
    let outcome = compile_program_get_result(test_program, "(3)").unwrap();
    assert_eq!(outcome.to_string(), "4");
}
