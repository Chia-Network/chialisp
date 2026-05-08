use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm_tools::cmds::launch_tool;

use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::frontend::frontend;
use crate::compiler::sexp::parse_sexp;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::usecheck::check_parameters_used_compileform;

fn check_argument_use(input_program: String) -> Vec<String> {
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(&"*test*".to_string()));
    let pre_forms =
        parse_sexp(Srcloc::start(&opts.filename()), input_program.bytes()).expect("should parse");
    let g = frontend(opts.clone(), &pre_forms).expect("should pass frontend");
    let set = check_parameters_used_compileform(opts, Rc::new(g.compileform().clone()))
        .expect("should be able to determine unused vars");
    let mut result = Vec::new();
    for x in set.iter() {
        result.push(Bytes::new(Some(BytesFromType::Raw(x.clone()))).decode());
    }
    result.sort();
    result
}

fn do_basic_run(args: &Vec<String>) -> String {
    let mut s = Stream::new(None);
    launch_tool(&mut s, args, &"run".to_string(), 2);
    return s.get_value().decode();
}

fn empty_vec() -> Vec<String> {
    vec![]
}

#[test]
fn check_unused_base_case_0() {
    assert_eq!(check_argument_use("(mod () (x))".to_string()), empty_vec());
}

#[test]
fn check_unused_base_case_1() {
    assert_eq!(check_argument_use("(mod (_) (x))".to_string()), empty_vec());
}

#[test]
fn check_unused_base_case_2() {
    assert_eq!(check_argument_use("(mod (A) (x))".to_string()), empty_vec());
}

#[test]
fn check_unused_base_case_3() {
    assert_eq!(
        check_argument_use("(mod (a) (x))".to_string()),
        vec!["a".to_string()]
    );
}

#[test]
fn check_unused_base_case_4() {
    assert_eq!(
        check_argument_use("(mod (a) (x a))".to_string()),
        empty_vec()
    );
}

#[test]
fn check_unused_fun_1() {
    assert_eq!(
        check_argument_use("(mod (a) (defun F (g) (+ g 1)) (F a))".to_string()),
        empty_vec()
    );
}

#[test]
fn check_unused_fun_2() {
    assert_eq!(
        check_argument_use("(mod (a) (defun F (g) (+ 2 3)) (F a))".to_string()),
        vec!["a".to_string()]
    );
}

#[test]
fn check_unused_fun_if_1() {
    assert_eq!(
        check_argument_use("(mod (a) (defun F (g h) (if g (+ h 1) ())) (F () a))".to_string()),
        vec!["a".to_string()]
    );
}

#[test]
fn check_unused_fun_if_2() {
    assert_eq!(
        check_argument_use("(mod (a) (defun F (g h) (if g (+ h 1) ())) (F 1 a))".to_string()),
        empty_vec()
    );
}

#[test]
fn check_unused_fun_rec_1() {
    assert_eq!(
        check_argument_use("(mod (a) (defun F (X) (if X (F (- X 1)) ())) (F a))".to_string()),
        empty_vec()
    );
}

#[test]
fn check_unused_fun_rec_2() {
    assert_eq!(
        check_argument_use("(mod (a) (defun F (R) (if R (F (- R 1)) R)) (F a))".to_string()),
        empty_vec()
    );
}

#[test]
fn check_unused_fun_rec_3() {
    assert_eq!(
        check_argument_use(
            "(mod (a b) (defun F (X Y) (if X (F (- X 1) Y) X)) (F a b))".to_string()
        ),
        vec!["b".to_string()]
    );
}

#[test]
fn check_unused_fun_rec_4() {
    assert_eq!(
        check_argument_use(
            "(mod (a b) (defun G (s t) t) (defun F (a b) (if b (F a (- b 1)) (G a b))) (F a b))"
                .to_string()
        ),
        vec!["a".to_string()]
    );
}

#[test]
fn check_unused_fun_let_1() {
    assert_eq!(
        check_argument_use(
            "(mod (a) (include *standard-cl-21*) (defun F (x) (let ((s (sha256 x))) 3)) (F a))"
                .to_string()
        ),
        vec!["a".to_string()]
    );
}

#[test]
fn verify_use_check_with_singleton_top_layer_fails_when_we_comment_out_all_uses_of_lineage_proof() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/usecheck-fail".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "--check-unused-args".to_string(),
        "resources/tests/singleton_top_layer.clvm".to_string(),
    ]);
    assert_eq!(res, "unused arguments detected at the mod level (lower case arguments are considered uncurried by convention)\n - lineage_proof\n");
}

#[test]
fn verify_use_check_with_singleton_top_layer_works() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "--check-unused-args".to_string(),
        "resources/tests/singleton_top_layer.clvm".to_string(),
    ]);
    assert!(res.len() > 0 && res.as_bytes()[0] == b'(');
}
