use std::collections::HashMap;
use std::rc::Rc;

use clvm_rs::allocator::Allocator;

use crate::compiler::compiler::compile_file;
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{CompileErr, CompilerOpts};
use crate::compiler::evaluate::{Evaluator, EVAL_STACK_LIMIT};
use crate::compiler::frontend::{from_clvm, frontend};
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::util::ErrInto;

use crate::tests::compiler::compiler::squash_name_differences;

fn shrink_expr_from_string(s: String) -> Result<String, CompileErr> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts = Rc::new(DefaultCompilerOpts::new(&"*program*".to_string()));
    let loc = Srcloc::start(&"*program*".to_string());
    let result = parse_sexp(loc.clone(), s.bytes())
        .err_into()
        .and_then(|parsed_program| {
            return frontend(opts.clone(), &parsed_program);
        })
        .and_then(|program| {
            let e = Evaluator::new(opts.clone(), runner, program.compileform().helpers.clone());
            return e.shrink_bodyform(
                &mut allocator,
                program.compileform().args.clone(),
                Rc::new(HashMap::new()),
                program.compileform().exp.clone(),
                false,
                Some(EVAL_STACK_LIMIT),
            );
        })?;

    let result_sexp =
        squash_name_differences(result.to_sexp()).map_err(|e| CompileErr(loc.clone(), e))?;
    Ok(result_sexp.to_string())
}

#[test]
fn test_basic_shrink_arithmetic() {
    assert_eq!(
        shrink_expr_from_string("(+ 3 (- 10 7))".to_string()).unwrap(),
        "(q . 6)".to_string()
    );
}

#[test]
fn test_basic_expand_macro() {
    assert_eq!(
        shrink_expr_from_string("(if 3 1 0)".to_string()).unwrap(),
        "(q . 1)"
    );
}

#[test]
fn test_basic_expand_macro_2() {
    assert_eq!(
        shrink_expr_from_string("(list 1 2 3)".to_string()).unwrap(),
        "(q 1 2 3)"
    );
}

#[test]
fn test_basic_expand_macro_3() {
    assert_eq!(
        shrink_expr_from_string("(mod (A) (list 1 A))".to_string()).unwrap(),
        "(c (q . 1) (c A (q)))"
    );
}

fn convert_clvm_to_chialisp(s: String) -> Result<Rc<SExp>, CompileErr> {
    let loc = Srcloc::start(&"*program*".to_string());
    parse_sexp(loc.clone(), s.bytes())
        .err_into()
        .map(|parsed_program| from_clvm(parsed_program[0].clone()))
}

#[test]
fn test_simple_conversion_from_clvm_to_chialisp() {
    let converted = convert_clvm_to_chialisp("(+ (q . 3) 2)".to_string()).unwrap();
    assert_eq!(converted.to_string(), "(+ (q . 3) (@ 2))");
}

#[test]
fn test_basic_expand_macro_4() {
    assert_eq!(
        shrink_expr_from_string(
            "(mod () (defun torp (S A B) (if S (+ A B) (* A B))) (torp 0 2 3))".to_string()
        )
        .unwrap(),
        "(q . 6)"
    );
}

#[test]
fn test_expand_with_recursive_1() {
    assert_eq!(
        shrink_expr_from_string("(mod () (defun factorial (input_$_11) (if (= input_$_11 1) 1 (* (factorial (- input_$_11 1)) input_$_11))) (factorial 3))".to_string()).unwrap(),
        "(q . 6)"
    );
}

fn compile_with_fe_opt(s: String) -> Result<String, CompileErr> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let mut opts: Rc<dyn CompilerOpts> =
        Rc::new(DefaultCompilerOpts::new(&"*program*".to_string()));
    opts = opts.set_frontend_opt(true);
    compile_file(
        &mut allocator,
        runner,
        opts,
        &s,
        &mut HashMap::new(),
        &mut Vec::new(),
    )
    .map(|r| r.to_sexp().to_string())
}

#[test]
fn test_simple_fe_opt_compile_1() {
    assert_eq!(
        compile_with_fe_opt("(mod () 99)".to_string()).unwrap(),
        "(2 (1 1 . 99) (4 (1) 1))".to_string()
    );
}

#[test]
fn test_lambda_eval_1() {
    assert_eq!(
        shrink_expr_from_string("(lambda (X) (+ X 1))".to_string()).unwrap(),
        "(lambda (X_$_A) (+ X_$_A 1))".to_string()
    );
}

#[test]
fn test_assign_simple_form_0() {
    assert_eq!(
        shrink_expr_from_string("(assign A (* Z 3) X 3 Y 4 Z (+ X Y) A)".to_string()).unwrap(),
        "(q . 21)"
    );
}

#[test]
fn test_lambda_eval_2() {
    assert_eq!(
        shrink_expr_from_string("(a (lambda (X) (+ X 1)) (list 3))".to_string()).unwrap(),
        "(q . 4)".to_string()
    );
}

#[test]
fn test_assign_simple_form_1() {
    assert_eq!(
        shrink_expr_from_string("(assign A (* Z 3) Z 2 A)".to_string()).unwrap(),
        "(q . 6)"
    );
}

#[test]
fn test_lambda_eval_3() {
    assert_eq!(
        shrink_expr_from_string(
            "(let ((L 10)) (a (lambda ((& L) X) (+ X L)) (list 3)))".to_string()
        )
        .unwrap(),
        "(q . 13)".to_string()
    );
}

#[test]
fn test_lambda_eval_4() {
    assert_eq!(
        shrink_expr_from_string(
            "(a (let ((L 10)) (lambda ((& L) X) (+ X L))) (list 3))".to_string()
        )
        .unwrap(),
        "(q . 13)".to_string()
    );
}

#[test]
fn test_assign_simple_form_2() {
    assert_eq!(
        shrink_expr_from_string("(assign Z 2 A (* Z 3) A)".to_string()).unwrap(),
        "(q . 6)"
    );
}
