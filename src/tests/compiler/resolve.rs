use std::collections::HashMap;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::run;
use crate::compiler::codegen::codegen;
use crate::compiler::compiler::{do_desugar, DefaultCompilerOpts};
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompilerOpts, DefunData, HelperForm, LetFormKind,
};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::resolve::resolve_namespaces;
use crate::compiler::sexp::{decode_string, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::BasicCompileContext;

fn do_program_module_resolution(
    opts: Rc<dyn CompilerOpts>,
    loc: Srcloc,
    test_program: &str,
) -> Result<CompileForm, CompileErr> {
    let parsed = parse_sexp(loc.clone(), test_program.bytes())?;
    let processed = frontend(opts.clone(), &parsed)?;
    resolve_namespaces(opts.clone(), &processed.compileform())
}

fn resolve_modules(test_program: &str) -> Result<CompileForm, CompileErr> {
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*resolve-test*"));
    let loc = Srcloc::start("*resolve-test*");
    do_program_module_resolution(opts, loc, test_program)
}

fn compile_program_get_result(
    test_program: &str,
    test_input: &str,
) -> Result<Rc<SExp>, CompileErr> {
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*resolve-test*"));
    let loc = Srcloc::start("*resolve-test*");
    let resolved = do_program_module_resolution(opts.clone(), loc.clone(), test_program)?;
    let desugared = do_desugar(opts.clone(), &resolved)?;
    let mut context = BasicCompileContext::new(
        Allocator::new(),
        Rc::new(DefaultProgramRunner::new()),
        HashMap::new(),
        get_optimizer(&loc, opts.clone())?,
    );
    let generated = codegen(&mut context, opts.clone(), None, &desugared)?;
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
fn test_compile_module_with_resolver_renaming() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (+ Z Z1.Q)) (defun G (Z) (- Z Z1.Q))) (import X exposing (F as FF)) (namespace Y (defun G (Z) (* Z 2))) (import Y exposing (G as GG)) (FF (GG (+ A (@ 5)))))";
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

fn find_helper<F, R>(cf: &CompileForm, name: &str, f: F) -> Option<R>
where
    F: FnOnce(&DefunData) -> R,
{
    for h in cf.helpers.iter() {
        if decode_string(h.name()) == name {
            if let HelperForm::Defun(_, dd) = &h {
                return Some(f(dd));
            }
        }
    }

    None
}

#[test]
fn test_module_resolve_preserves_parallel_let() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (let ((ZZ (+ Z Z1.Q))) ZZ))) (import X hiding G) (F 3))";
    let cf = resolve_modules(test_program).unwrap();
    let outcome = compile_program_get_result(test_program, "(3)").unwrap();
    assert_eq!(outcome.to_string(), "4");
    let funbody = find_helper(&cf, "X.F", |f| f.body.clone()).unwrap();
    assert!(matches!(&*funbody, BodyForm::Let(LetFormKind::Parallel, _)));
}

#[test]
fn test_resolve_with_assign() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (assign ZZ (+ Z Z1.Q) ZZ))) (import X hiding G) (F 3))";
    let outcome = compile_program_get_result(test_program, "(3)").unwrap();
    assert_eq!(outcome.to_string(), "4");
}

#[test]
fn test_resolve_error_on_bad_as_keyword() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z 1337 Z1) (defun F (Z) (assign ZZ (+ Z Z1.Q) ZZ))) (import X hiding G) (F 3))";
    assert!(compile_program_get_result(test_program, "(3)").is_err());
}

#[test]
fn test_resolve_error_on_bad_as_keyword_rename() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (assign ZZ (+ Z Z1.Q) ZZ))) (import X exposing (F 1337 FF)) (FF 3))";
    assert!(compile_program_get_result(test_program, "(3)").is_err());
}

#[test]
fn test_resolve_error_on_bad_as_keyword_rename_target() {
    let test_program =
        "(mod (A) (include *standard-cl-24*) (namespace Z (defconst Q 1)) (namespace X (import qualified Z as Z1) (defun F (Z) (assign ZZ (+ Z Z1.Q) ZZ))) (import X exposing (F as 1337)) (FF 3))";
    assert!(compile_program_get_result(test_program, "(3)").is_err());
}
