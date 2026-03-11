use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use clvm_rs::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::classic::clvm_tools::binutils::disassemble;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::run;
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::{CompileErr, CompilerOpts};
use crate::compiler::dialect::{AcceptedDialect, KNOWN_DIALECTS};
use crate::compiler::frontend::{collect_used_names_sexp, frontend};
use crate::compiler::rename::rename_in_cons;
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::{decode_string, enlist, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;

use crate::tests::classic::run::do_basic_brun;

const TEST_TIMEOUT: usize = 1000000;

fn compile_string(content: &String) -> Result<String, CompileErr> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts = Rc::new(DefaultCompilerOpts::new(&"*test*".to_string()));

    compile_file(&mut allocator, runner, opts, &content, &mut HashMap::new()).map(|x| x.to_string())
}

fn run_string_maybe_opt(
    content: &String,
    args: &String,
    fe_opt: bool,
    strict: bool,
) -> Result<Rc<SExp>, CompileErr> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let mut opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(&"*test*".to_string()));
    let srcloc = Srcloc::start(&"*test*".to_string());
    opts = opts
        .set_frontend_opt(fe_opt && !strict)
        .set_search_paths(&vec!["resources/tests".to_string()]);

    if strict {
        opts = opts.set_dialect(AcceptedDialect {
            stepping: Some(21),
            strict: true,
            int_fix: false,
        });
    }

    let sexp_args = parse_sexp(srcloc.clone(), args.bytes())?[0].clone();

    compile_file(
        &mut allocator,
        runner.clone(),
        opts,
        &content,
        &mut HashMap::new(),
    )
    .and_then(|x| {
        run(
            &mut allocator,
            runner,
            Rc::new(HashMap::new()),
            Rc::new(x),
            sexp_args,
            None,
            Some(TEST_TIMEOUT),
        )
        .map_err(|e| match e {
            RunFailure::RunErr(l, s) => CompileErr(l, s),
            RunFailure::RunExn(l, s) => CompileErr(l, s.to_string()),
        })
    })
}

pub fn run_string(content: &String, args: &String) -> Result<Rc<SExp>, CompileErr> {
    run_string_maybe_opt(content, args, false, false)
}

pub fn run_string_strict(content: &String, args: &String) -> Result<Rc<SExp>, CompileErr> {
    run_string_maybe_opt(content, args, false, true)
}

// Given some renaming that leaves behind gensym style names with _$_<n> in them,
// order them and use a locally predictable renaming scheme to give them a final
// test checkable value.
pub fn squash_name_differences(in_sexp: Rc<SExp>) -> Result<Rc<SExp>, String> {
    let found_names_set: BTreeSet<_> = collect_used_names_sexp(in_sexp.clone())
        .into_iter()
        .filter(|n| n.contains(&b'$'))
        .collect();
    let mut found_names_progression = b'A';
    let mut replacement_map = HashMap::new();
    for found_name in found_names_set.iter() {
        if let Some(located_dollar_part) = found_name.iter().position(|x| *x == b'$') {
            let mut new_name: Vec<u8> = found_name
                .iter()
                .take(located_dollar_part + 2)
                .copied()
                .collect();
            new_name.push(found_names_progression);
            found_names_progression += 1;
            replacement_map.insert(found_name.clone(), new_name);
        } else {
            return Err(decode_string(&found_name));
        }
    }
    if replacement_map.len() != found_names_set.len() {
        return Err(format!(
            "mismatched lengths {replacement_map:?} vs {found_names_set:?}"
        ));
    }
    Ok(rename_in_cons(&replacement_map, in_sexp, false))
}

/* // Upcoming support for extra optimization (WIP)
fn run_string_opt(content: &String, args: &String) -> Result<Rc<SExp>, CompileErr> {
    run_string_maybe_opt(content, args, true)
}
*/

#[test]
fn compile_test_1() {
    let result = compile_string(
        &"(mod () (defmacro testmacro (A) (qq (+ 1 (unquote A)))) (testmacro 3))".to_string(),
    )
    .unwrap();
    assert_eq!(result, "(2 (1 16 (1 . 1) (1 . 3)) (4 (1) 1))".to_string());
}

#[test]
fn compile_test_2() {
    let result =
        compile_string(
            &"(mod () (defmacro if (A B C) (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @))) (if () (+ 1 3) (+ 5 8)))".to_string()
        ).unwrap();
    assert_eq!(result, "(2 (1 2 (3 (1) (1 2 (1 16 (1 . 1) (1 . 3)) 1) (1 2 (1 16 (1 . 5) (1 . 8)) 1)) 1) (4 (1) 1))".to_string());
}

#[test]
fn compile_test_3() {
    let result = compile_string(&"(mod (A) (include *standard-cl-21*) A)".to_string()).unwrap();
    assert_eq!(result, "(2 (1 . 5) (4 (1) 1))".to_string());
}

#[test]
fn compile_test_4() {
    let result =
        compile_string(
            &"(mod () (defmacro if (A B C) (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @))) (if () (+ 1 3) (+ 5 8)))".to_string()
        ).unwrap();
    assert_eq!(result, "(2 (1 2 (3 (1) (1 2 (1 16 (1 . 1) (1 . 3)) 1) (1 2 (1 16 (1 . 5) (1 . 8)) 1)) 1) (4 (1) 1))".to_string());
}

#[test]
fn compile_test_5() {
    let result =
        compile_string(
            &"(mod (X) (include *standard-cl-21*) (defmacro testmacro (x) (qq (+ 1 (unquote x)))) (if X (testmacro 3) (testmacro 4)))".to_string()
        ).unwrap();
    assert_eq!(
        result,
        "(2 (1 2 (3 5 (1 2 (1 16 (1 . 1) (1 . 3)) 1) (1 2 (1 16 (1 . 1) (1 . 4)) 1)) 1) (4 (1) 1))"
            .to_string()
    );
}

#[test]
fn compile_test_6() {
    let result = compile_string(&"(mod () (list 1 2 3))".to_string()).unwrap();
    assert_eq!(
        result,
        "(2 (1 4 (1 . 1) (4 (1 . 2) (4 (1 . 3) (1)))) (4 (1) 1))".to_string()
    );
}

// odd numbered list
#[test]
fn compile_test_8() {
    let result =
        compile_string(&"(mod (S) (c S (q . #(2000 3000 4000 5000 6000 7000 8000))))".to_string())
            .unwrap();
    assert_eq!(
        result,
        "(2 (1 4 5 (1 (2000 3000 . 4000) (5000 . 6000) 7000 . 8000)) (4 (1) 1))".to_string()
    );
}

// even numbered list
#[test]
fn compile_test_9() {
    let result = compile_string(&"(mod (S) (c S (q . #(a b c d))))".to_string()).unwrap();
    assert_eq!(
        result,
        "(2 (1 4 5 (1 (a . b) c . d)) (4 (1) 1))".to_string()
    );
}

// word
#[test]
fn compile_test_10() {
    let result = compile_string(&"(mod (S) (c S #fake))".to_string()).unwrap();
    assert_eq!(result, "(2 (1 4 5 (1 . fake)) (4 (1) 1))".to_string());
}

// op letter
#[test]
fn compile_test_11() {
    let result = compile_string(&"(mod (S) (c S #a))".to_string()).unwrap();
    assert_eq!(result, "(2 (1 4 5 (1 . 2)) (4 (1) 1))".to_string());
}

// length 1 list
#[test]
fn compile_test_12() {
    let result = compile_string(&"(mod (S) (c S (q . #(100))))".to_string()).unwrap();
    assert_eq!(result, "(2 (1 4 5 (1 . 100)) (4 (1) 1))".to_string());
}

// length 0 list
#[test]
fn compile_test_13() {
    let result = compile_string(&"(mod (S) (c S (q . #())))".to_string()).unwrap();
    assert_eq!(result, "(2 (1 4 5 (1)) (4 (1) 1))".to_string());
}

// length 2 list
#[test]
fn compile_test_14() {
    let result = compile_string(&"(mod (S) (c S (q . #(a b))))".to_string()).unwrap();
    assert_eq!(result, "(2 (1 4 5 (1 a . b)) (4 (1) 1))".to_string());
}

// use structured list in solution
#[test]
fn compile_test_15() {
    let result = run_string_maybe_opt(
        &"(mod #(a b c) (- (+ a c) b))".to_string(),
        &"(100 20 . 10)".to_string(),
        true,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "90".to_string());
}

// use structured list in solution
#[test]
fn compile_test_16() {
    let result = run_string_maybe_opt(
        &"(mod #(a b c) (- (+ a c) b))".to_string(),
        &"#(100 20 10)".to_string(),
        true,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "90".to_string());
}

fn run_test_1_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &"(mod () (defun f (a b) (+ (* a a) b)) (f 3 1))".to_string(),
        &"()".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "10".to_string());
}

#[test]
fn compile_test_7() {
    let result =
        compile_string(&"(mod (X) (include *standard-cl-21*) (/ X 10))".to_string()).unwrap();
    assert_eq!(result, "(2 (1 5 (20 5 (1 . 10))) (4 (1) 1))".to_string());
}

#[test]
fn run_test_1() {
    run_test_1_maybe_opt(false);
}

#[test]
fn run_test_1_opt() {
    run_test_1_maybe_opt(true);
}

fn run_test_2_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &"(mod (c) (defun f (a b) (+ (* a a) b)) (f 3 c))".to_string(),
        &"(4)".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "13".to_string());
}

#[test]
fn run_test_2() {
    run_test_2_maybe_opt(false);
}

#[test]
fn run_test_2_opt() {
    run_test_2_maybe_opt(true);
}

fn run_test_3_maybe_opt(opt: bool) {
    let result =
        run_string_maybe_opt(
            &"(mod (arg_one) (defun factorial (input) (if (= input 1) 1 (* (factorial (- input 1)) input))) (factorial arg_one))".to_string(),
            &"(5)".to_string(),
            opt,
            false,
        ).unwrap();
    assert_eq!(result.to_string(), "120".to_string());
}

#[test]
fn run_test_3() {
    run_test_3_maybe_opt(false);
}

#[test]
fn run_test_3_opt() {
    run_test_3_maybe_opt(true);
}

fn run_test_4_maybe_opt(opt: bool) {
    let result =
        run_string_maybe_opt(
            &"(mod () (defun makelist (a) (if a (c (q . 4) (c (f a) (c (makelist (r a)) (q . ())))) (q . ()))) (makelist (q . (1 2 3))))".to_string(),
            &"()".to_string(),
            opt,
            false,
        ).unwrap();
    assert_eq!(result.to_string(), "(4 1 (4 2 (4 3 ())))".to_string());
}

#[test]
fn run_test_4() {
    run_test_4_maybe_opt(false);
}

#[test]
fn run_test_4_opt() {
    run_test_4_maybe_opt(true);
}

fn run_test_5_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &"(mod (a) (list 1 2))".to_string(),
        &"()".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "(1 2)".to_string());
}

#[test]
fn run_test_5() {
    run_test_5_maybe_opt(false);
}

#[test]
fn run_test_5_opt() {
    run_test_5_maybe_opt(true);
}

fn run_test_6_maybe_opt(opt: bool) {
    let result =
        run_string_maybe_opt(
            &"(mod args (defmacro square (input) (qq (* (unquote input) (unquote input)))) (defun sqre_list (my_list) (if my_list (c (square (f my_list)) (sqre_list (r my_list))) my_list)) (sqre_list args))".to_string(),
            &"(10 9 8 7)".to_string(),
            opt,
            false,
        ).unwrap();
    assert_eq!(result.to_string(), "(100 81 64 49)".to_string());
}

#[test]
fn run_test_6() {
    run_test_6_maybe_opt(false);
}

#[test]
fn run_test_6_opt() {
    run_test_6_maybe_opt(true);
}

fn run_test_7_maybe_opt(opt: bool) {
    let result =
        run_string_maybe_opt(
            &"(mod (PASSWORD_HASH password new_puzhash amount) (defconstant CREATE_COIN 51) (defun check_password (PASSWORD_HASH password new_puzhash amount) (if (= (sha256 password) PASSWORD_HASH) (list (list CREATE_COIN new_puzhash amount)) (x))) (check_password PASSWORD_HASH password new_puzhash amount))".to_string(),
            &"(0x2ac6aecf15ac3042db34af4863da46111da7e1bf238fc13da1094f7edc8972a1 \"sha256ftw\" 0x12345678 1000000000)".to_string(),
            opt,
            false,
        ).unwrap();
    assert_eq!(
        result.to_string(),
        "((51 0x12345678 1000000000))".to_string()
    );
}

#[test]
fn run_test_7() {
    run_test_7_maybe_opt(false);
}

#[test]
fn run_test_7_opt() {
    run_test_7_maybe_opt(true);
}

fn run_test_8_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &"(mod (a b) (let ((x (+ a 1)) (y (+ b 1))) (+ x y)))".to_string(),
        &"(5 8)".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "15".to_string());
}

#[test]
fn run_test_8() {
    run_test_8_maybe_opt(false);
}

#[test]
fn run_test_8_opt() {
    run_test_8_maybe_opt(true);
}

#[test]
fn run_inlines() {
    let result = run_string(
        &"(mod (a) (defun-inline F (x) (+ x 1)) (defun-inline G (x) (* x 2)) (G (F a)))"
            .to_string(),
        &"(13)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "28".to_string());
}

#[test]
fn run_inlines_2() {
    let result = run_string(
        &"(mod (a b) (defun-inline F (x y) (+ x y)) (defun-inline G (x y) (- y x)) (defun-inline H (x y) (* x y)) (H (F a b) (G a b)))"
            .to_string(),
        &"(103 107)".to_string(),
    )
        .unwrap();
    // H (103 + 107) (108 - 104)
    // 210 * 4
    // 840
    assert_eq!(result.to_string(), "840".to_string());
}

#[test]
fn run_test_at_form() {
    let result = run_string(
        &"(mod (a b) (- (@ 11) (@ 5)))".to_string(),
        &"(51 107)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "56".to_string());
}

#[test]
fn run_test_intermediate_let_1() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun-inline letbinding_$_264 args args)
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}
        .to_string(),
        &"(100)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "((100) 101)".to_string());
}

#[test]
fn run_test_intermediate_let_1_1() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun-inline letbinding_$_265 args args)
                (defun letbinding_$_264 ((a) x) (letbinding_$_265 (c (c a ()) (c x ())) (+ x 1)))
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}
        .to_string(),
        &"(100)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "(((100) 101) 102)".to_string());
}

#[test]
fn run_test_intermediate_let_1_2() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun letbinding_$_265 args args)
                (defun letbinding_$_264 ((a) x) (letbinding_$_265 (c (c a ()) (c x ())) (+ x 1)))
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}
        .to_string(),
        &"(100)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "(((100) 101) 102)".to_string());
}

#[test]
fn run_test_intermediate_let_1_3() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun-inline letbinding_$_265 args args)
                (defun letbinding_$_264 ((a) x) (letbinding_$_265 (c (c a ()) (c x ())) (+ x 1)))
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}
        .to_string(),
        &"(100)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "(((100) 101) 102)".to_string());
}

#[test]
fn run_test_intermediate_let_1_4() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun letbinding_$_265 args args)
                (defun-inline letbinding_$_264 ((a) x) (letbinding_$_265 (c (c a ()) (c x ())) (+ x 1)))
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}.to_string(),
        &"(100)".to_string(),
    ).unwrap();
    assert_eq!(result.to_string(), "(((100) 101) 102)".to_string());
}

#[test]
fn run_test_intermediate_let_2() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun-inline letbinding_$_265 args args)
                (defun-inline letbinding_$_264 ((a) x) (letbinding_$_265 (c (c a ()) (c x ())) (+ x 1)))
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}.to_string(),
        &"(100)".to_string(),
    ).unwrap();
    assert_eq!(result.to_string(), "(((100) 101) 102)".to_string());
}

#[test]
fn run_test_intermediate_let_final() {
    let result = run_string(
        &indoc! {"
            (mod (a)
                (defun-inline letbinding_$_265 (((a) x_$_263) y) (+ x_$_263 y))
                (defun-inline letbinding_$_264 ((a) x) (letbinding_$_265 (c (c a ()) (c x ())) (+ x 1)))
                (letbinding_$_264 (r @) (+ a 1))
                )
        "}.to_string(),
        &"(100)".to_string(),
    ).unwrap();
    assert_eq!(result.to_string(), "203".to_string());
}

#[test]
fn run_test_let_star_2_deep() {
    let result = run_string(
        &"(mod (a) (let* ((x (+ a 1)) (y (+ x 1))) (+ x y)))".to_string(),
        &"(100)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "203".to_string());
}

#[test]
fn run_test_let_star_3_deep() {
    let result = run_string(
        &"(mod (a) (let* ((x (+ a 1)) (y (+ x 1)) (z (* a y))) (+ x y z)))".to_string(),
        &"(100)".to_string(),
    )
    .unwrap();
    assert_eq!(result.to_string(), "10403".to_string());
}

#[test]
fn run_test_normal_with_macro_call() {
    let result = run_string(
        &"(mod (a) (defun test-value (a n) (if (- a n) 9999 1111)) (c (test-value a 3) (test-value a 2)))".to_string(),
        &"(3)".to_string()
    ).unwrap();
    assert_eq!(result.to_string(), "(1111 . 9999)".to_string());
}

#[test]
fn run_test_inline_with_macro_call() {
    let result = run_string(
        &"(mod (X) (defun-inline test-value (a n) (if (- a n) 9999 1111)) (c (test-value X 3) (test-value X 2)))".to_string(),
        &"(3)".to_string()
    ).unwrap();
    assert_eq!(result.to_string(), "(1111 . 9999)".to_string());
}

/*
 * - TODO: Ensure that a compileform name inlined and shadowed in a macro doesn't
 *   disrupt macro execution.
 *   ... i'll have to think about how to handle this.
 * /
#[test]
fn run_test_inline_with_macro_call_tricky_naming() {
    let result = run_string(
        &"(mod (a) (defun-inline test-value (a n) (if (- a n) 9999 1111)) (c (test-value a 3) (test-value a 2)))".to_string(),
        &"(3)".to_string()
    ).unwrap();
    assert_eq!(result.to_string(), "(1111 . 9999)".to_string());
}
 */

fn run_test_9_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &"(mod (a) (defun f (i) (let ((x (not i)) (y (* i 2))) (+ x y))) (f a))".to_string(),
        &"(0)".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "1".to_string());
}

#[test]
fn run_test_9() {
    run_test_9_maybe_opt(false);
}

#[test]
fn run_test_9_opt() {
    run_test_9_maybe_opt(true);
}

fn run_test_10_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &"(mod (a) (defun f (i) (let ((x (not i)) (y (* i 2))) (+ x y))) (f a))".to_string(),
        &"(3)".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "6".to_string());
}

#[test]
fn run_test_10() {
    run_test_10_maybe_opt(false);
}

#[test]
fn run_test_10_opt() {
    run_test_10_maybe_opt(true);
}

#[test]
fn test_defconstant() {
    let result =
        run_string(&indoc!{"
            (mod (password new_puzhash amount)
              (include *standard-cl-21*) ;; Specify chialisp-21 compilation.
              (defconstant CREATE_COIN 51)
              (defun check-password (password)
                (let ((password-hash (sha256 password))
                      (real-hash 0x2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824))
                  (= password-hash real-hash)
                  )
                )

              (if (check-password password)
                (list (list CREATE_COIN new_puzhash amount))
                (x)
                )
              )
        "}.to_string(),
                   &"(hello 0x5f5767744f91c1c326d927a63d9b34fa7035c10e3eb838c44e3afe127c1b7675 2)".to_string(),
        ).unwrap();

    assert_eq!(
        result.to_string(),
        "((51 0x5f5767744f91c1c326d927a63d9b34fa7035c10e3eb838c44e3afe127c1b7675 2))".to_string()
    );
}

#[test]
fn inline_compile_test() {
    let result = compile_string(
        &indoc! {"
        (mod (A)
          (include *standard-cl-21*)

          (defun-inline f (x) (+ x 1))
          (f A)
          )
        "}
        .to_string(),
    )
    .unwrap();
    assert_eq!(result, "(2 (1 16 5 (1 . 1)) (4 (1) 1))".to_string());
}

#[test]
fn cant_redefine_defconstant() {
    let result = compile_string(
        &"(mod (X) (include *standard-cl-21*) (defconstant A 3) (defconstant A 4) A)".to_string(),
    );
    assert!(result.is_err());
}

#[test]
fn cant_redefine_defun_with_defun() {
    let inline_if_not_zero = |i, c| {
        if i & c == 0 {
            ""
        } else {
            "-inline"
        }
    };
    for i in 0..4 {
        let result = compile_string(&format!(
            indoc! {"
            (mod (A)
              (include *standard-cl-21*)

              (defun{} f (x) (+ x 1))
              (defun{} f (y) (- y 1))

              (f A)
              )
            "},
            inline_if_not_zero(i, 2),
            inline_if_not_zero(i, 1)
        ));
        assert!(result.is_err());
    }
}

#[test]
fn test_at_destructure_1() {
    let result = run_string(
        &indoc! {"
            (mod (A (@ Z (B C)) D)
              (include *standard-cl-21*) ;; Specify chialisp-21 compilation.
              A
              )
        "}
        .to_string(),
        &"(1 (2 3) 4)".to_string(),
    )
    .unwrap();

    assert_eq!(result.to_string(), "1".to_string());
}

#[test]
fn test_at_destructure_2() {
    let result = run_string(
        &indoc! {"
            (mod (A (@ Z (B C)) D)
              (include *standard-cl-21*) ;; Specify chialisp-21 compilation.
              Z
              )
        "}
        .to_string(),
        &"(1 (2 3) 4)".to_string(),
    )
    .unwrap();

    assert_eq!(result.to_string(), "(2 3)".to_string());
}

#[test]
fn test_at_destructure_3() {
    let result = run_string(
        &indoc! {"
            (mod (A (@ Z (B C)) D)
              (include *standard-cl-21*) ;; Specify chialisp-21 compilation.
              B
              )
        "}
        .to_string(),
        &"(1 (2 3) 4)".to_string(),
    )
    .unwrap();

    assert_eq!(result.to_string(), "2".to_string());
}

#[test]
fn test_at_destructure_4() {
    let result = run_string(
        &indoc! {"
            (mod (A (@ Z (B C)) D)
              (include *standard-cl-21*) ;; Specify chialisp-21 compilation.
              C
              )
        "}
        .to_string(),
        &"(1 (2 3) 4)".to_string(),
    )
    .unwrap();

    assert_eq!(result.to_string(), "3".to_string());
}

#[test]
fn test_at_destructure_5() {
    let result = run_string(
        &indoc! {"
            (mod (A (@ Z (B C)) D)
              (include *standard-cl-21*) ;; Specify chialisp-21 compilation.
              D
              )
        "}
        .to_string(),
        &"(1 (2 3) 4)".to_string(),
    )
    .unwrap();

    assert_eq!(result.to_string(), "4".to_string());
}

fn test_collatz_maybe_opt(opt: bool) {
    let result = run_string_maybe_opt(
        &indoc! {"
            (mod (A)
             (include *standard-cl-22*)
             (defun-inline odd (X) (logand X 1))
             (defun collatz (N X)
              (if (= X 1)
               N
               (let ((incN (+ N 1)))
                (if (odd X)
                 (collatz incN (+ 1 (* 3 X)))
                 (collatz incN (/ X 2))
                )
               )
              )
             )
             (collatz 0 A)
            )
        "}
        .to_string(),
        &"(4)".to_string(),
        opt,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "2");
}

#[test]
fn test_collatz() {
    test_collatz_maybe_opt(false);
}

#[test]
fn fancy_nested_let_bindings_should_work() {
    let result = run_string_maybe_opt(
        &indoc! {"
    (mod X
     (include *standard-cl-21*)
     (defun do-something (solutions se)
        (if solutions
             (let (
                    (R (f solutions))
                    (S (- se 99))
                  )
                (if (= (f solutions) 99)
                    S
                    (let ((something-else (+ se R)))
                        (do-something (r solutions) something-else)
                    )
                )
             )
             100
        )
    )

    (do-something X 1)
    )
        "}
        .to_string(),
        &"(1 2 3 100 99)".to_string(),
        false,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "8");
}

#[test]
fn let_as_argument() {
    let result = run_string_maybe_opt(
        &indoc! {"
    (mod (X)
     (include *standard-cl-21*)
     (defun twice (x) (* x 2))
     (defun plus (x y) (+ x y))
     (plus (let ((t (twice X))) t) 3))
        "}
        .to_string(),
        &"(5)".to_string(),
        false,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "13");
}

#[test]
fn recursive_let_complicated_arguments() {
    let result = run_string_maybe_opt(
        &indoc! {"
    (mod (X Y)
     (include *standard-cl-21*)
     (defun G (A (@ pt (X Y))) (+ A X Y))
     (defun F (@ pt (X Y)) (let ((X1 (+ X 1)) (Y1 (+ Y 3))) (G X1 (let ((p (list X1 Y1))) p))))
     (F X Y)
     )
        "}
        .to_string(),
        &"(7 13)".to_string(),
        false,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "32");
}

#[test]
fn test_let_structure_access_1() {
    let result = run_string_maybe_opt(
        &indoc! {"
    (mod (X Y)
      (include *standard-cl-21*)
      (let ((a 1)
            (b 2))
        (let ((A (+ a 1))
               (XX (+ X 1))
               (C (+ b b)))
         (if Y
            (let ((D (+ C 1))
                  (E (+ XX Y)))
              (c D E)
              )
            (let ((D XX)
                  (E (+ XX Y)))
              (c D E)
              )
          )
         )
       )
      )
        "}
        .to_string(),
        &"(7 13)".to_string(),
        false,
        false,
    )
    .unwrap();
    // a = 1
    // b = 2
    // A = 2
    // XX = X + 1
    // C = 2 + 2
    // if Y
    //   D = 5
    //   E = X + Y + 1
    // else
    //   D = X + 1
    //   E = X + Y + 1
    assert_eq!(result.to_string(), "(5 . 21)");
}

#[test]
fn test_let_structure_access_2() {
    let result = run_string_maybe_opt(
        &indoc! {"
    (mod (X Y)
      (include *standard-cl-21*)
      (let ((a 1)
            (b 2))
        (let* ((A (+ a 1))
               (XX (+ X 1))
               (C (+ A b)))
         (if Y
            (let ((D (+ C 1))
                  (E (+ XX Y)))
              (c D E)
              )
            (let ((D XX)
                  (E (+ XX Y)))
              (c D E)
              )
          )
         )
       )
      )
        "}
        .to_string(),
        &"(7 13)".to_string(),
        false,
        false,
    )
    .unwrap();
    // a = 1
    // b = 2
    // A = 2
    // XX = X + 1
    // C = 2 + 2
    // if Y
    //   D = 5
    //   E = X + Y + 1
    // else
    //   D = X + 1
    //   E = X + Y + 1
    assert_eq!(result.to_string(), "(5 . 21)");
}

#[test]
fn test_let_inline_1() {
    let result = run_string_maybe_opt(
        &indoc! {"
    (mod (G)
      (include *standard-cl-21*)
      (defun-inline F (X) (let ((Y (* X 2))) (+ Y 1)))
      (F G)
      )
        "}
        .to_string(),
        &"(5)".to_string(),
        false,
        false,
    )
    .unwrap();
    assert_eq!(result.to_string(), "11");
}

#[test]
fn read_of_hex_constant_in_modern_chialisp() {
    let result = run_string(
        &indoc! {"(mod () (include *standard-cl-21*) (sha256 (q . 1) (q . 0xf22ada22a0ed015000ea157013ee62dc6ce337a649ec01054fc62ed6caac7eaf)))"}
        .to_string(),
        &"()".to_string(),
    )
        .unwrap();

    assert_eq!(
        result.to_string(),
        "36364122602940516403027890844760998025315693007634105146514094828976803085567".to_string()
    );
}

#[test]
fn hash_handling_test_2() {
    let result = run_string(
        &indoc! {"(mod () (include *standard-cl-21*) (sha256 (q . 1) (q . 1)))"}.to_string(),
        &"()".to_string(),
    )
    .unwrap();

    assert_eq!(
        result.to_string(),
        "-44412188149083219915772186748035909266791016930429887947443501395007119841358"
    );
}

#[test]
fn hash_handling_test_3() {
    let result = run_string(
        &indoc! {"(mod () (include *standard-cl-21*) (sha256 1 0))"}.to_string(),
        &"()".to_string(),
    )
    .unwrap();

    assert_eq!(
        result.to_string(),
        "34356466678672179216206944866734405838331831190171667647615530531663699592602"
    );
}

#[test]
fn sebastian_hash_test_1() {
    let result = run_string(
        &indoc! {"
(mod (MOD_HASH TOKEN_A_AMOUNT TOKEN_B_AMOUNT K token_a_delta token_b_delta)
    (include \"condition_codes.clvm\")
    (include \"curry-and-treehash.clinc\")
    (include *standard-cl-21*)

    (defun sha256tree1 (TREE)
        (if (l TREE)
            (sha256 2 (sha256tree1 (f TREE)) (sha256tree1 (r TREE)))
            (sha256 1 TREE)
        )
    )
    (defun return-new-coin (mod_hash new_token_a_amount new_token_b_amount K)
        (list (list mod_hash (sha256tree1 mod_hash))
            (list
                CREATE_COIN
               (puzzle-hash-of-curried-function
                    mod_hash (sha256tree1 mod_hash)
                )
                1
            )
        )
    )

    (let (
         (new_token_a_amount (+ TOKEN_A_AMOUNT token_a_delta))
         (new_token_b_amount (+ TOKEN_B_AMOUNT token_b_delta))
        )
        (if (all (= K (* new_token_a_amount new_token_b_amount)) (> new_token_a_amount 0) (> new_token_b_amount 0) )
            (return-new-coin MOD_HASH new_token_a_amount new_token_b_amount K)
            (x new_token_a_amount new_token_b_amount)
        )
    ))"}
        .to_string(),
        &"(0xf22ada22a0ed015000ea157013ee62dc6ce337a649ec01054fc62ed6caac7eaf 10000 200 2000000 -2000 50)".to_string()
    )
        .unwrap();

    assert_eq!(result.to_string(), "((0xf22ada22a0ed015000ea157013ee62dc6ce337a649ec01054fc62ed6caac7eaf 36364122602940516403027890844760998025315693007634105146514094828976803085567) (51 -54330418644829767769717664495663952010367061369724157609947295940464695774007 1))");
}

#[test]
fn sebastian_hash_test_2() {
    let result = run_string(
        &indoc! {"
(mod (MOD_HASH TOKEN_A_AMOUNT TOKEN_B_AMOUNT K token_a_delta token_b_delta)
    (include \"condition_codes.clvm\")
    (include \"curry-and-treehash.clinc\")

    (defun sha256tree1 (TREE)
        (if (l TREE)
            (sha256 2 (sha256tree1 (f TREE)) (sha256tree1 (r TREE)))
            (sha256 1 TREE)
        )
    )
    (defun return-new-coin (mod_hash new_token_a_amount new_token_b_amount K)
        (list (list mod_hash (sha256tree1 mod_hash))
            (list
                CREATE_COIN
               (puzzle-hash-of-curried-function
                    mod_hash (sha256tree1 mod_hash)
                )
                1
            )
        )
    )

        (if (all (= K (* (+ TOKEN_A_AMOUNT token_a_delta) (+ TOKEN_B_AMOUNT token_b_delta))) (> (+ TOKEN_A_AMOUNT token_a_delta) 0) (> (+ TOKEN_B_AMOUNT token_b_delta) 0) )
            (return-new-coin MOD_HASH (+ TOKEN_A_AMOUNT token_a_delta) (+ TOKEN_B_AMOUNT token_b_delta) K)
            (x (+ TOKEN_A_AMOUNT token_a_delta) (+ TOKEN_B_AMOUNT token_b_delta))
        )
    )"}
        .to_string(),
        &"(0xf22ada22a0ed015000ea157013ee62dc6ce337a649ec01054fc62ed6caac7eaf 10000 200 2000000 -2000 50)".to_string()
    )
        .unwrap();

    assert_eq!(result.to_string(), "((0xf22ada22a0ed015000ea157013ee62dc6ce337a649ec01054fc62ed6caac7eaf 36364122602940516403027890844760998025315693007634105146514094828976803085567) (51 -54330418644829767769717664495663952010367061369724157609947295940464695774007 1))");
}

#[test]
fn test_modern_inline_at_capture() {
    let result = run_string(
        &indoc! {"
(mod (Z)
  (include *standard-cl-21*)
  (defun-inline check_lineage_proof
    (
      THIS_MOD_HASH
      TYPES
      (@ lineage_proof (foo bar))
      conditions
    )
    (if TYPES
        (x lineage_proof)
        conditions
    )
  )

  (check_lineage_proof 1 2 Z 4)
  )"}
        .to_string(),
        &"(99)".to_string(),
    )
    .unwrap_err();

    assert_eq!(result.1, "clvm raise in (8 5) (() 99)");
}

#[test]
fn test_modern_inline_recurse() {
    // Test for a crash doing a recursive inline call.
    run_string(
        &"(mod () (include *standard-cl-21*) (defun-inline FOO (X) (FOO (+ X 1))) (FOO 3))"
            .to_string(),
        &"()".to_string(),
    )
    .unwrap_err();
}

#[test]
fn test_modern_inline_recurse_deep() {
    // Test for a crash doing a recursive inline call.
    run_string(
        &"(mod () (include *standard-cl-21*) (defun-inline BAR (X) (FOO (- X 1))) (defun-inline FOO (X) (BAR (+ X 1))) (FOO 3))".to_string(),
        &"()".to_string()
    ).unwrap_err();
}

#[test]
fn test_modern_mod_form() {
    let result = run_string(
        &indoc! {"
(mod () (include *standard-cl-21*) (a (mod (X) (+ 1 (* X 2))) (list 3)))
"}
        .to_string(),
        &"()".to_string(),
    )
    .unwrap();

    assert_eq!(result.to_string(), "7");
}

#[test]
fn test_fuzz_seed_3956111146_1_alt_test_1() {
    let res = run_string(
        &"(mod () (include *standard-cl-21*) (q (r . lbvepvnoc) dbhk))".to_string(),
        &"()".to_string(),
    )
    .unwrap();
    assert_eq!(res.to_string(), "((r . lbvepvnoc) dbhk)");
}

#[test]
fn test_fuzz_seed_3956111146_1() {
    let res = run_string(
        &"(mod () (include *standard-cl-21*) (q (r . lbvepvnoc) . dbhk))".to_string(),
        &"()".to_string(),
    )
    .unwrap();
    assert_eq!(res.to_string(), "((r . lbvepvnoc) . dbhk)");
}

#[test]
fn arg_destructure_test_1() {
    let prog = indoc! {"
(mod
  (
      SINGLETON_MOD_HASH
      LAUNCHER_HASH
      launcher_id
      . delegated_puzzle_hash
  )

  (include *standard-cl-21*)

  delegated_puzzle_hash
)"}
    .to_string();
    let res = run_string(&prog, &"(1 2 3 . 4)".to_string()).unwrap();
    assert_eq!(res.to_string(), "4");
}

#[test]
fn test_defconstant_tree() {
    let prog = indoc! {"
(mod ()
  (include *standard-cl-21*)
  (include test-defconstant-tree.clib)
  constant-tree
  )"}
    .to_string();
    let res = run_string(&prog, &"()".to_string()).unwrap();
    assert_eq!(res.to_string(), "((0x4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a . 0x9dcf97a184f32623d11a73124ceb99a5709b083721e878a16d78f596718ba7b2) 0x02a12871fee210fb8619291eaea194581cbd2531e4b23759d225f6806923f63222 . 0x02a8d5dd63fba471ebcb1f3e8f7c1e1879b7152a6e7298a91ce119a63400ade7c5)");
}

#[test]
fn test_assign_form_0() {
    let prog = indoc! {"
(mod (X)
  (assign
    X1 (+ X 1) ;; X1 is 14 if X is 13.
    X1
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "14");
}

#[test]
fn test_assign_form_1() {
    let prog = indoc! {"
(mod (X)
  (assign
    ;; A lot of forms in order.
    X1 (+ X 1) ;; 14
    X2 (+ X1 1) ;; 15
    X3 (+ X2 1) ;; 16
    Y0 (+ X3 1) ;; 17
    X4 (+ X3 1) ;; 17
    X5 (+ Y0 1) ;; 18
    Y1 (+ X5 Y0) ;; 35
    Y1
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_assign_form_cplx_1() {
    let prog = indoc! {"
(mod (X)
  (defun-inline tup (X Y) (c X Y))
  (assign
    (X1 . X2) (tup (+ X 1) (+ X 2)) ;; 14
    X3 (+ X1 1) ;; 15
    X4 (+ X2 1) ;; 16
    (Y0 . X5) (tup (+ X4 1) (+ X4 1)) ;; 17
    X6 (+ Y0 1) ;; 18
    Y1 (+ X6 Y0) ;; 35
    Y1
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_assign_form_in_let_binding() {
    let prog = indoc! {"
(mod (X)
  (defun-inline tup (X Y) (c X Y))
  (let
    ((FOO
      (assign
        (X1 . X2) (tup (+ X 1) (+ X 2)) ;; 14
        X3 (+ X1 1) ;; 15
        X4 (+ X2 1) ;; 16
        (Y0 . X5) (tup (+ X4 1) (+ X4 1)) ;; 17
        X6 (+ Y0 1) ;; 18
        Y1 (+ X6 Y0) ;; 35
        Y1
        )))
    FOO
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_assign_form_in_function_argument() {
    let prog = indoc! {"
(mod (X)
  (defun-inline tup (X Y) (c X Y))
  (defun F (A B) (+ A B))
  (F
    (assign
      (X1 . X2) (tup (+ X 1) (+ X 2)) ;; 14
      X3 (+ X1 1) ;; 15
      X4 (+ X2 1) ;; 16
      (Y0 . X5) (tup (+ X4 1) (+ X4 1)) ;; 17
      X6 (+ Y0 1) ;; 18
      Y1 (+ X6 Y0) ;; 35
      Y1
      )
    101
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "136");
}

#[test]
fn test_assign_form_in_inline_argument() {
    let prog = indoc! {"
(mod (X)
  (defun-inline tup (X Y) (c X Y))
  (defun-inline F (A B) (+ A B))
  (F
    (assign
      (X1 . X2) (tup (+ X 1) (+ X 2)) ;; 14
      X3 (+ X1 1) ;; 15
      X4 (+ X2 1) ;; 16
      (Y0 . X5) (tup (+ X4 1) (+ X4 1)) ;; 17
      X6 (+ Y0 1) ;; 18
      Y1 (+ X6 Y0) ;; 35
      Y1
      )
    101
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "136");
}

#[test]
fn test_assign_in_if() {
    let prog = indoc! {"
(mod (X)
  (defun-inline tup (X Y) (c X Y))
  (defun-inline F (A B) (+ A B))
  (if X
    (assign
      (X1 . X2) (tup (+ X 1) (+ X 2)) ;; 14
      X3 (+ X1 1) ;; 15
      X4 (+ X3 1) ;; 16
      (Y0 . X5) (tup (+ X4 1) (+ X4 1)) ;; 17
      X6 (+ Y0 1) ;; 18
      Y1 (+ X6 Y0) ;; 35
      Y1
      )
      101
      )
    )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_assign_fun_cplx_2() {
    let prog = indoc! {"
(mod (X)
  (defun-inline tup (X Y) (c X Y))
  (defun-inline F (A B) (+ A B))
  (if X
    (let*
      ((Z
        (assign
          (X1 . X2) (tup (+ X 1) (+ X 2)) ;; 14
          X3 (+ X1 1) ;; 15
          X4 (+ X2 1) ;; 16
          (Y0 . X5) (tup (+ X4 1) (+ X4 1)) ;; 17
          X6 (+ Y0 1) ;; 18
          Y1 (+ X6 Y0) ;; 35
          Y1
          ))
        (Q (assign R (+ 3 2) (* R Z)))
        )
      Q
      )
      101
      )
    )"}
    .to_string();
    let res = run_string(&prog, &"(13)".to_string()).unwrap();
    assert_eq!(res.to_string(), "175");
}

#[test]
fn test_assign_simple_with_reodering() {
    let prog = indoc! {"
(mod (A) ;; 11
  (include *standard-cl-21*)
  (defun tup (a b) (c a b))
  (assign
    ;; This exercises reordering in assign.
    ;; Each set os grouped with a tier that will be taken together.
    ;; The tiers are numbered in order, but given out of order.

    ;; Tier 1
    (X0 . X1) (tup (+ A 1) (- A 1)) ;; 12 10

    ;; Tier 4
    finish (+ x2_gtr_x3 (- X3 x2_minus_x3)) ;; 1 + (70 - 50)

    ;; Tier 3
    x2_gtr_x3 (> X2 X3) ;; 1
    x2_minus_x3 (- X2 X3) ;; 50

    ;; Tier 2
    X2 (* X0 10) ;; 120
    X3 (* X1 7) ;; 70

    finish
    ))"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).unwrap();
    assert_eq!(res.to_string(), "21");
}

#[test]
fn test_assign_detect_multiple_definition() {
    let prog = indoc! {"
(mod (A) ;; 11
  (include *standard-cl-21*)
  (defun tup (a b) (c a b))
  (assign
    ;; Tier 1
    (X0 . X1) (tup (+ A 1) (- A 1)) ;; 12 10

    ;; Tier 4
    finish (+ x2_gtr_x3 (- X3 x2_minus_x3)) ;; 1 + (70 - 50)

    ;; Tier 3
    x2_gtr_x3 (> X2 X3) ;; 1
    x2_minus_x3 (- X2 X3) ;; 50

    ;; Tier 2
    X2 (* X0 10) ;; 120
    X2 (* X1 7) ;; 70

    finish
    ))"}
    .to_string();
    if let Err(CompileErr(l, e)) = run_string(&prog, &"(11)".to_string()) {
        assert_eq!(l.line, 17);
        assert!(e.starts_with("Duplicate"));
    } else {
        assert!(false);
    }
}

#[test]
fn test_assign_dont_detect_unrelated_inlines_as_recursive() {
    let prog = indoc! {"
(mod (A) ;; 11
  (include *standard-cl-22*)
  (defun-inline <= (A B) (not (> A B)))
  (let
    ((foo (<= 2 A))
     (bar (<= 1 A)))

    (let
      ((baz (<= foo bar)))

      (let
        ((yorgle (<= baz bar)))

        (<= yorgle foo)
        )
      )
    )
  )"}
    .to_string();
    let res = run_string(&prog, &"(2)".to_string()).expect("should compile");
    assert_eq!(res.to_string(), "1");
}

#[test]
fn test_inline_out_of_bounds_diagnostic() {
    let prog = indoc! {"
(mod ()
  (include *standard-cl-21*)
  (defun-inline FOO (X Y) (+ X Y))
  (FOO 3)
  )"}
    .to_string();
    let res = run_string(&prog, &"()".to_string());
    if let Err(CompileErr(l, e)) = res {
        assert_eq!(l.line, 4);
        assert!(e.starts_with("Lookup"));
    } else {
        assert!(false);
    }
}

#[test]
fn test_lambda_without_capture_from_function() {
    let prog = indoc! {"
(mod (A B)
  (include *standard-cl-21*)
  (defun FOO () (lambda (X Y) (+ X Y)))
  (a (FOO) (list A B))
  )"}
    .to_string();
    let res = run_string(&prog, &"(3 4)".to_string()).unwrap();
    assert_eq!(res.to_string(), "7");
}

#[test]
fn test_lambda_without_capture() {
    let prog = indoc! {"
(mod (A B)
  (include *standard-cl-21*)
  (a (lambda (X Y) (+ X Y)) (list A B))
  )"}
    .to_string();
    let res = run_string(&prog, &"(3 4)".to_string()).unwrap();
    assert_eq!(res.to_string(), "7");
}

#[test]
fn test_lambda_with_capture_from_function() {
    let prog = indoc! {"
(mod (A B)
  (include *standard-cl-21*)
  (defun FOO (Z) (lambda ((& Z) X) (- X Z)))
  (a (FOO A) (list B))
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 19)".to_string()).unwrap();
    assert_eq!(res.to_string(), "14");
}

#[test]
fn test_lambda_with_capture() {
    let prog = indoc! {"
(mod (A B)
  (include *standard-cl-21*)
  (a (lambda ((& A) Y) (- Y A)) (list B))
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 19)".to_string()).unwrap();
    assert_eq!(res.to_string(), "14");
}

#[test]
fn test_lambda_in_let_0() {
    let prog = indoc! {"
(mod (A)
  (include *standard-cl-21*)
  (defun FOO (Z)
    (let ((Q (* 2 Z)))
      (lambda ((& Q)) (- 100 Q))
      )
    )
  (a (FOO A) ())
  )"}
    .to_string();
    let res = run_string(&prog, &"(5)".to_string()).unwrap();
    assert_eq!(res.to_string(), "90");
}

#[test]
fn test_lambda_in_let_1() {
    let prog = indoc! {"
(mod (A B)
  (include *standard-cl-21*)
  (defun FOO (Z)
    (let ((Q (* 2 Z)))
      (lambda ((& Q) X) (- X Q))
      )
    )
  (a (FOO A) (list B))
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 19)".to_string()).unwrap();
    assert_eq!(res.to_string(), "9");
}

#[test]
fn test_lambda_in_map() {
    let prog = indoc! {"
(mod (add-number L)

  (include *standard-cl-21*)

  (defun map (F L)
    (if L
      (c (a F (list (f L))) (map F (r L)))
      ()
      )
    )

  (map
    (lambda ((& add-number) number) (+ add-number number))
    L
    )
  )
"}
    .to_string();
    let res = run_string(&prog, &"(5 (1 2 3 4))".to_string()).unwrap();
    assert_eq!(res.to_string(), "(6 7 8 9)");
}

#[test]
fn test_lambda_in_map_with_let_surrounding() {
    let prog = indoc! {"
(mod (add-number L)

  (include *standard-cl-21*)

  (defun map (F L)
    (if L
      (c (a F (list (f L))) (map F (r L)))
      ()
      )
    )

  (map
    (let ((A (* add-number 2)))
      (lambda ((& A) number) (+ A number))
      )
    L
    )
  )
"}
    .to_string();
    let res = run_string(&prog, &"(5 (1 2 3 4))".to_string()).unwrap();
    assert_eq!(res.to_string(), "(11 12 13 14)");
}

#[test]
fn test_map_with_lambda_function_from_env_and_bindings() {
    let prog = indoc! {"
    (mod (add-number L)

     (include *standard-cl-21*)

     (defun map (F L)
      (if L
       (c (a F (list (f L))) (map F (r L)))
       ()
      )
     )

     (defun add-twice (X Y) (+ (* 2 X) Y))

     (map
      (lambda ((& add-number) number) (add-twice add-number number))
      L
     )
    )"}
    .to_string();
    let res = run_string(&prog, &"(5 (1 2 3 4))".to_string()).unwrap();
    assert_eq!(res.to_string(), "(11 12 13 14)");
}

#[test]
fn test_map_with_lambda_function_from_env_no_bindings() {
    let prog = indoc! {"
    (mod (L)

     (include *standard-cl-21*)

     (defun map (F L)
      (if L
       (c (a F (list (f L))) (map F (r L)))
       ()
      )
     )

     (defun sum-list (L)
       (if L
         (+ (f L) (sum-list (r L)))
         ()
         )
       )

     (map
      (lambda (lst) (sum-list lst))
      L
     )
    )"}
    .to_string();
    let res = run_string(&prog, &"(((5 10 15) (2 4 8) (3 6 9)))".to_string()).unwrap();
    assert_eq!(res.to_string(), "(30 14 18)");
}

#[test]
fn test_lambda_using_let() {
    let prog = indoc! {"
    (mod (P L)

     (include *standard-cl-21*)

     (defun map (F L)
      (if L
       (c (a F (list (f L))) (map F (r L)))
       ()
      )
     )

     (map
      (lambda ((& P) item) (let ((composed (c P item))) composed))
      L
     )
    )"}
    .to_string();
    let res = run_string(&prog, &"(1 (10 20 30))".to_string()).unwrap();
    assert_eq!(res.to_string(), "((1 . 10) (1 . 20) (1 . 30))");
}

#[test]
fn test_lambda_using_macro() {
    let prog = indoc! {"
    (mod (P L)

     (include *standard-cl-21*)

     (defun map (F L)
      (if L
       (c (a F (list (f L))) (map F (r L)))
       ()
      )
     )

     (map
      (lambda ((& P) item) (list P item))
      L
     )
    )"}
    .to_string();
    let res = run_string(&prog, &"(1 (10 20 30))".to_string()).unwrap();
    assert_eq!(res.to_string(), "((1 10) (1 20) (1 30))");
}

#[test]
fn test_lambda_reduce() {
    let prog = indoc! {"
    (mod (LST)
     (include *standard-cl-21*)
     (defun reduce (fun lst init)
      (if lst
       (reduce fun (r lst) (a fun (list (f lst) init)))
       init
       )
      )

     (let
      ((capture 100))
      (reduce (lambda ((& capture) (X Y) ACC) (+ (* X Y) ACC capture)) LST 0)
      )
     )
    "}
    .to_string();
    let res = run_string(&prog, &"(((2 3) (4 9)))".to_string()).unwrap();
    assert_eq!(res.to_string(), "242");
}

#[test]
fn test_lambda_as_let_binding() {
    let prog = indoc! {"
    (mod (P L)
      (defun map (F L)
        (if L (c (a F (list (f L))) (map F (r L))) ())
        )
      (defun x2 (N) (* 2 N))
      (defun x3p1 (N) (+ 1 (* 3 N)))
      (let* ((H (lambda (N) (x2 N)))
             (G (lambda (N) (x3p1 N)))
             (F (if P G H)))
        (map F L)
        )
      )
    "}
    .to_string();
    let res0 = run_string(&prog, &"(0 (1 2 3))".to_string()).unwrap();
    assert_eq!(res0.to_string(), "(2 4 6)");
    let res1 = run_string(&prog, &"(1 (1 2 3))".to_string()).unwrap();
    assert_eq!(res1.to_string(), "(4 7 10)");
}

#[test]
fn test_lambda_mixed_let_binding() {
    let prog = indoc! {"
    (mod (P L)
      (defun map (F L)
        (if L (c (a F (list (f L))) (map F (r L))) ())
        )
      (defun x2 (N) (* 2 N))
      (defun x3p1 (N) (+ 1 (* 3 N)))
      (let* ((G (lambda (N) (x3p1 N)))
             (F (if P G (lambda (N) (x2 N)))))
        (map F L)
        )
      )
    "}
    .to_string();
    let res0 = run_string(&prog, &"(0 (1 2 3))".to_string()).unwrap();
    assert_eq!(res0.to_string(), "(2 4 6)");
    let res1 = run_string(&prog, &"(1 (1 2 3))".to_string()).unwrap();
    assert_eq!(res1.to_string(), "(4 7 10)");
}

#[test]
fn test_lambda_hof_1() {
    let prog = indoc! {"
    (mod (P)
      (a (a (lambda ((& P) X) (lambda ((& P X)) (+ P X))) (list 3)) ())
      )
    "}
    .to_string();
    let res = run_string(&prog, &"(1)".to_string()).unwrap();
    assert_eq!(res.to_string(), "4");
}

#[test]
fn test_lambda_as_argument_to_macro() {
    let prog = indoc! {"
    (mod (P)
      (defun map-f (A L)
        (if L (c (a (f L) A) (map-f A (r L))) ())
        )
      (let ((Fs (list (lambda (X) (- X 1)) (lambda (X) (+ X 1)) (lambda (X) (* 2 X))))
            (args (list P)))
        (map-f args Fs)
        )
      )
    "}
    .to_string();
    let res = run_string(&prog, &"(10)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(9 11 20)");
}

#[test]
fn test_lambda_as_argument_to_macro_with_inner_let() {
    let prog = indoc! {"
    (mod (P)
      (defun map-f (A L)
        (if L (c (a (f L) A) (map-f A (r L))) ())
        )
      (let ((Fs (list (lambda (X) (let ((N (* X 3))) N)) (lambda (X) (+ X 1)) (lambda (X) (* 2 X))))
            (args (list P)))
        (map-f args Fs)
        )
      )
    "}
    .to_string();
    let res = run_string(&prog, &"(10)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(30 11 20)");
}

#[test]
fn test_treat_function_name_as_value() {
    let prog = indoc! {"
(mod (X)
 (include *standard-cl-21*)
 (defun G (X) (* 2 X))
 (defun F (X) (G (+ 1 X)))
 (a F (list X))
)
    "}
    .to_string();
    let res = run_string(&prog, &"(99)".to_string()).expect("should compile");
    assert_eq!(res.to_string(), "200");
}

#[test]
fn test_treat_function_name_as_value_filter() {
    let prog = indoc! {"
    (mod L
     (include *standard-cl-21*)
     (defun greater-than-3 (X) (> X 3))
     (defun filter (F L) (let ((rest (filter F (r L)))) (if L (if (a F (list (f L))) (c (f L) rest) rest) ())))
     (filter greater-than-3 L)
    )
    "}
    .to_string();
    let res = run_string(&prog, &"(1 2 3 4 5)".to_string()).expect("should compile");
    assert_eq!(res.to_string(), "(4 5)");
}

#[test]
fn test_inline_in_assign_not_actually_recursive() {
    let prog = indoc! {"
(mod (POINT)
  (include *standard-cl-21*)
  (defun-inline no-op (V) V)
  (let ((TU 100)) (let ((TI1 (no-op TU)) (TU2 (no-op TU))) 9999))
  )"}
    .to_string();
    let res = run_string(&prog, &"()".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "9999");
}

#[test]
fn test_simple_rest_call_0() {
    let prog = indoc! {"
(mod X
  (include *standard-cl-21*)

  (defun F Xs
    (if Xs
      (+ (f Xs) (F &rest (r Xs)))
      ()
      )
    )

  (F &rest X)
  )"}
    .to_string();
    let res = run_string(&prog, &"(13 99 144)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "256");
}

#[test]
fn test_simple_rest_call_inline() {
    let prog = indoc! {"
(mod X
  (include *standard-cl-21*)

  (defun sum (Xs)
    (if Xs
      (+ (f Xs) (sum (r Xs)))
      ()
      )
    )

  (defun-inline F (A1 . A2) (* A1 (sum A2)))

  (F 3 &rest X)
  )"}
    .to_string();
    let res = run_string(&prog, &"(13 99 144)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "768");
}

#[test]
fn test_simple_rest_lambda() {
    let prog = indoc! {"
(mod (Z X)
  (include *standard-cl-21*)

  (defun silly-lambda-consumer (Q F) (a F (list Q)))

  (silly-lambda-consumer &rest (list X (lambda ((& Z) X) (* Z X))))
  )"}
    .to_string();
    let res = run_string(&prog, &"(13 51)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "663");
}

#[test]
fn test_lambda_in_lambda() {
    let prog = indoc! {"
(mod (Z X)
  (include *standard-cl-21*)

  (defun silly-lambda-consumer (Q F) (a F (list Q)))

  (a (silly-lambda-consumer X (lambda ((& Z) X) (lambda ((& Z X)) (* Z X)))) ())
  )"}
    .to_string();
    let res = run_string(&prog, &"(13 51)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "663");
}

#[test]
fn test_let_in_rest_0() {
    let prog = indoc! {"
(mod (Z X)
  (include *standard-cl-21*)

  (defun F (X) (+ X 3))

  (F &rest (list (let ((Q (* X Z))) (+ Q 99))))
  )"}
    .to_string();
    let res = run_string(&prog, &"(3 2)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "108");
}

#[test]
fn test_let_in_rest_1() {
    let prog = indoc! {"
(mod (Z X)
  (include *standard-cl-21*)

  (defun F (X) (+ X 3))

  (F &rest (let ((Q (* X Z))) (list (+ Q 99))))
  )"}
    .to_string();
    let res = run_string(&prog, &"(3 2)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "108");
}

#[test]
fn test_lambda_override_name_arg_let_capture() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden)))
      (lambda ((& overridden) z) (+ overridden z))
      )
    )

  (a (F X) (list 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "50");
}

#[test]
fn test_lambda_override_name_arg_let_with_let_in_lambda_1() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden)))
      (lambda ((& overridden) z)
        (let
          ((z (+ 123 z)))
          (+ overridden z)
          )
        )
      )
    )

  (a (F X) (list 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "173");
}

#[test]
fn test_lambda_override_name_arg_let_with_let_in_lambda_2() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden)))
      (lambda ((& overridden) z)
        (let
          ((overridden (+ 123 overridden)))
          (+ overridden z)
          )
        )
      )
    )

  (a (F X) (list 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "173");
}

#[test]
fn test_lambda_override_name_arg_assign_with_assign_in_lambda_1() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (Z)
    (assign overridden (* 3 Z)
      (lambda ((& overridden) z)
        (let
          ((z (+ 123 z)))
          (+ overridden z)
          )
        )
      )
    )

  (a (F X) (list 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "173");
}

#[test]
fn test_lambda_override_name_arg_let_with_assign_in_lambda_1() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden))) ;; overridden = 33
      (lambda ((& overridden) z) ;; overridden = 33
        (assign overridden (+ 123 z) ;; overridden = 17 + 123 = 140
          (+ overridden z) ;; 157
          )
        )
      )
    )

  (a (F X) (list 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "157");
}

#[test]
fn test_lambda_override_name_arg_let_with_let_in_lambda_3() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden))) ;; overridden = 33
      (lambda ((& overridden) z) ;; overridden = 33
        (let ((overridden (+ 123 z))) ;; overridden = 17 + 123 = 140
          (+ overridden z) ;; 157
          )
        )
      )
    )

  (a (F X) (list 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "157");
}

#[test]
fn test_lambda_override_name_arg_let_with_assign_twice_in_lambda() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden))) ;; overridden = 33
      (lambda ((& overridden) y z) ;; overridden = 33
        (assign
          overridden (+ 123 z) ;; overridden = 123 + 17 = 140
          y (+ 191 z overridden) ;; y = 191 + 17 + 140 = 348
          (+ overridden z y) ;; 505
          )
        )
      )
    )

  (a (F X) (list 13 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "505");
}

#[test]
fn test_lambda_override_name_arg_let_with_let_twice_in_lambda() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden))) ;; overridden = 33
      (lambda ((& overridden) y z) ;; overridden = 33
        (let
          ((overridden (+ 123 z))) ;; overridden = 123 + 17 = 140
          (let ((y (+ 191 z overridden))) ;; y = 191 + 17 + 140 = 348
            (+ overridden z y) ;; 505
            )
          )
        )
      )
    )

  (a (F X) (list 13 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "505");
}

#[test]
fn test_lambda_override_name_arg_let_with_let_star_twice_in_lambda() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden))) ;; overridden = 33
      (lambda ((& overridden) y z) ;; overridden = 33
        (let*
          ((overridden (+ 123 z)) ;; overridden = 123 + 17 = 140
           (y (+ 191 z overridden))) ;; y = 191 + 17 + 140 = 348
          (+ overridden z y) ;; 505
          )
        )
      )
    )

  (a (F X) (list 13 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "505");
}

#[test]
fn test_lambda_let_override_in_binding() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F (overridden)
    (let ((overridden (* 3 overridden))) ;; overridden = 33
      (lambda ((& overridden) y z) ;; overridden = 33
        (let
          ((y (+ 191 z (let ((overridden (+ 123 z))) overridden)))) ;; overridden = 123 + 17 = 140, y = 191 + 17 + 140 = 348
          (+ overridden z y) ;; 33 + 17 + 348 = 398
          )
        )
      )
    )

  (a (F X) (list 13 17))
  )"}
    .to_string();
    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "398");
}

#[test]
fn test_rename_in_compileform_run() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F overridden
    (let
      ((overridden (* 3 (f overridden))) ;; overridden = 33
       (y (f (r overridden))) ;; y = 13
       (z (f (r (r overridden))))) ;; z = 17
      (+ overridden z y) ;; 33 + 13 + 17 = 63
      )
    )

  (F X 13 17)
  )"}
    .to_string();

    let res = run_string(&prog, &"(11)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "63");
}

#[test]
fn test_rename_in_compileform_simple() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)

  (defun F overridden
    (let
      ((overridden (* 3 (f overridden))) ;; overridden = 33
       (y (f (r overridden))) ;; y = 11
       (z (f (r (r overridden))))) ;; z = 17
      (+ overridden z y) ;; 33 11 17
      )
    )

  (F X 13 17)
  )"}
    .to_string();
    // Note: renames use gensym so they're unique but not spot predictable.
    //
    // We'll rename them in detection order to a specific set of names and should
    // get for F:
    //
    let desired_outcome = "(defun F overridden_$_A (let ((overridden_$_B (* 3 (f overridden_$_A))) (y_$_C (f (r overridden_$_A))) (z_$_D (f (r (r overridden_$_A))))) (+ overridden_$_B z_$_D y_$_C)))";
    let parsed = parse_sexp(Srcloc::start("*test*"), prog.bytes()).expect("should parse");
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(&"*test*".to_string()));
    let compiled = frontend(opts, &parsed).expect("should compile");
    let helper_f: Vec<_> = compiled
        .helpers
        .iter()
        .filter(|f| f.name() == b"F")
        .collect();
    let renamed_helperform = squash_name_differences(helper_f[0].to_sexp()).expect("should rename");
    assert_eq!(renamed_helperform.to_string(), desired_outcome);
}

#[test]
fn test_handle_explicit_empty_atom() {
    let filename = "*empty-atom-test*";
    let srcloc = Srcloc::start(filename);
    let opts = Rc::new(DefaultCompilerOpts::new(filename)).set_dialect(AcceptedDialect {
        stepping: Some(21),
        strict: true,
        int_fix: false,
    });

    let atom = |s: &str| Rc::new(SExp::Atom(srcloc.clone(), s.as_bytes().to_vec()));

    let sublist = |l: &[Rc<SExp>]| Rc::new(enlist(srcloc.clone(), l));

    let nil = Rc::new(SExp::Nil(srcloc.clone()));

    let program = sublist(&[
        atom("mod"),
        nil.clone(),
        sublist(&[atom("include"), atom("*strict-cl-21*")]),
        sublist(&[
            atom("+"),
            atom(""),
            Rc::new(SExp::Integer(srcloc.clone(), bi_one())),
        ]),
    ]);
    let mut allocator = Allocator::new();
    let mut symbols = HashMap::new();
    let runner = Rc::new(DefaultProgramRunner::new());

    let compiled = opts
        .compile_program(&mut allocator, runner.clone(), program, &mut symbols)
        .expect("should compile");
    let outcome = run(
        &mut allocator,
        runner,
        opts.prim_map(),
        Rc::new(compiled),
        nil,
        None,
        None,
    )
    .expect("should run");
    assert_eq!(outcome.to_string(), "1");
}

#[test]
fn test_almost_empty_lambda_gives_error() {
    let prog = indoc! {"
(mod (X)
  (include *standard-cl-21*)
  (lambda)
  )"}
    .to_string();
    let res = run_string(&prog, &"()".to_string());
    assert!(res.is_err());
    assert!(format!("{res:?}").contains("Must provide at least arguments and body to lambda"));
}

#[test]
fn test_exhaustive_chars() {
    // Verify that we can create a program that gives the expected output using
    // every byte value in the first, mid and last position of a value.
    let mut substitute = vec![b'x', b'x', b'x'];

    let srcloc = Srcloc::start("*extest*");
    let make_test_program = |sub: Rc<SExp>| {
        // (mod () (include *standard-cl-23.1*) (q . <sub>))
        Rc::new(SExp::Cons(
            srcloc.clone(),
            Rc::new(SExp::Atom(srcloc.clone(), b"mod".to_vec())),
            Rc::new(SExp::Cons(
                srcloc.clone(),
                Rc::new(SExp::Nil(srcloc.clone())),
                Rc::new(SExp::Cons(
                    srcloc.clone(),
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        Rc::new(SExp::Atom(srcloc.clone(), b"include".to_vec())),
                        Rc::new(SExp::Cons(
                            srcloc.clone(),
                            Rc::new(SExp::Atom(srcloc.clone(), b"*standard-cl-23.1*".to_vec())),
                            Rc::new(SExp::Nil(srcloc.clone())),
                        )),
                    )),
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        Rc::new(SExp::Cons(
                            srcloc.clone(),
                            Rc::new(SExp::Integer(srcloc.clone(), bi_one())),
                            sub,
                        )),
                        Rc::new(SExp::Nil(srcloc.clone())),
                    )),
                )),
            )),
        ))
    };

    let runner = Rc::new(DefaultProgramRunner::new());

    for i in 0..=2 {
        for j in 0..=255 {
            substitute[i] = j;

            let sub_qe = Rc::new(SExp::QuotedString(srcloc.clone(), b'"', substitute.clone()));

            let mut allocator = Allocator::new();
            let mut opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*extest*"));
            let dialect = KNOWN_DIALECTS["*standard-cl-23.1*"].accepted.clone();
            opts = opts.set_dialect(dialect);

            let compiled = opts
                .compile_program(
                    &mut allocator,
                    runner.clone(),
                    make_test_program(sub_qe),
                    &mut HashMap::new(),
                )
                .expect("should compile");

            let compiled_output = compiled.to_string();
            let result = do_basic_brun(&vec!["brun".to_string(), compiled_output])
                .trim()
                .to_string();

            let classic_atom = allocator.new_atom(&substitute).expect("should work");
            let disassembled = disassemble(&mut allocator, classic_atom, None);
            assert_eq!(result, disassembled);

            substitute[i] = b'x';
        }
    }
}

#[test]
fn test_odd_hex_works() {
    let res = run_string(
        &"(mod () (include *standard-cl-23*) (+ 1 0xf))".to_string(),
        &"()".to_string(),
    )
    .expect("should work");
    assert_eq!(res.to_string(), "16");
}
