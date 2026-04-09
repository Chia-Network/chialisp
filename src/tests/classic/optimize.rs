use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, SExp};

use crate::classic::clvm_tools::binutils::{assemble_from_ir, disassemble};
use crate::classic::clvm_tools::ir::reader::read_ir;
use crate::classic::clvm_tools::stages::stage_2::operators::run_program_for_search_paths;
use crate::classic::clvm_tools::stages::stage_2::optimize::{
    children_optimizer, cons_q_a_optimizer, constant_optimizer, optimize_sexp, seems_constant,
    sub_args,
};

fn test_cons_q_a(src: String) -> String {
    let mut allocator = Allocator::new();
    let memo = RefCell::new(HashMap::new());
    let input_ir = read_ir(&src, 0).unwrap();
    let assembled = assemble_from_ir(&mut allocator, Rc::new(input_ir)).unwrap();
    let runner = run_program_for_search_paths("*test*", &vec![".".to_string()], false, 0);
    let optimized = cons_q_a_optimizer(&mut allocator, &memo, assembled, runner.clone()).unwrap();
    disassemble(&mut allocator, optimized, Some(0))
}

fn test_children_optimizer(src: String) -> String {
    let mut allocator = Allocator::new();
    let memo = RefCell::new(HashMap::new());
    let input_ir = read_ir(&src, 0).unwrap();
    let assembled = assemble_from_ir(&mut allocator, Rc::new(input_ir)).unwrap();
    let runner = run_program_for_search_paths("*test*", &vec![".".to_string()], false, 0);
    let optimized = children_optimizer(&mut allocator, &memo, assembled, runner.clone()).unwrap();
    disassemble(&mut allocator, optimized, Some(0))
}

fn test_constant_optimizer(src: String) -> String {
    let mut allocator = Allocator::new();
    let memo = RefCell::new(HashMap::new());
    let input_ir = read_ir(&src, 0).unwrap();
    let assembled = assemble_from_ir(&mut allocator, Rc::new(input_ir)).unwrap();
    let runner = run_program_for_search_paths("*test*", &vec![".".to_string()], false, 0);
    let optimized =
        constant_optimizer(&mut allocator, &memo, assembled, 0, runner.clone()).unwrap();
    disassemble(&mut allocator, optimized, Some(0))
}

fn test_optimizer(src: String) -> String {
    let mut allocator = Allocator::new();
    let input_ir = read_ir(&src, 0).unwrap();
    let assembled = assemble_from_ir(&mut allocator, Rc::new(input_ir)).unwrap();
    let runner = run_program_for_search_paths("*test*", &vec![".".to_string()], false, 0);
    let optimized = optimize_sexp(&mut allocator, assembled, runner.clone()).unwrap();
    disassemble(&mut allocator, optimized, Some(0))
}

fn test_sub_args(src: String) -> String {
    let mut allocator = Allocator::new();
    let input_ir = read_ir(&src, 0).unwrap();
    let assembled = assemble_from_ir(&mut allocator, Rc::new(input_ir)).unwrap();
    match allocator.sexp(assembled) {
        SExp::Pair(a, b) => {
            let optimized = sub_args(&mut allocator, a, b).unwrap();
            disassemble(&mut allocator, optimized, Some(0))
        }
        _ => {
            panic!("assembled a list got an atom");
        }
    }
}

#[test]
fn cons_q_a_simple() {
    assert_eq!(
        test_cons_q_a("(a (q 1 . \"opt\") 1)".to_string()),
        "(q . \"opt\")".to_string()
    );
}

#[test]
fn cons_q_a_optimizer_example() {
    let src = "(a (q \"opt\" (q 2 (\"opt\" (\"com\" (q 88 65 66) (q () () ((q . \"list\") (a (q (q . 97) (q (q . 97) (q . 2) (c (q . 2) (c (q . 3) (q)))) (c (q (q . 97) (i (q . 5) (q (q . 99) 4 (c (q . 9) (c (a (q . 2) (c (q . 2) (c (q . 13) (q)))) (q)))) (q (q . 1))) (q . 1)) (q . 1))) (q . 1))) ((q . \"defmacro\") (c \"list\" (c (f (q . 1)) (c (c \"mod\" (c (f (r (q . 1))) (c (f (r (r (q . 1)))) (q)))) (q)))))) (q (65 5) (66 11) (88 2)))) (c (\"opt\" (\"com\" (q 16 65 66) (q () () ((q . \"list\") (a (q (q . 97) (q (q . 97) (q . 2) (c (q . 2) (c (q . 3) (q)))) (c (q (q . 97) (i (q . 5) (q (q . 99) 4 (c (q . 9) (c (a (q . 2) (c (q . 2) (c (q . 13) (q)))) (q)))) (q (q . 1))) (q . 1)) (q . 1))) (q . 1))) ((q . \"defmacro\") (c \"list\" (c (f (q . 1)) (c (c \"mod\" (c (f (r (q . 1))) (c (f (r (r (q . 1)))) (q)))) (q)))))) (q (65 5) (66 11) (88 2)))) 1))) 1)".to_string();
    let optimized = test_cons_q_a(src);
    assert_eq!(optimized, "(\"opt\" (q 2 (\"opt\" (\"com\" (q 88 65 66) (q () () ((q . \"list\") (a (q (q . 97) (q (q . 97) (q . 2) (c (q . 2) (c (q . 3) (q)))) (c (q (q . 97) (i (q . 5) (q (q . 99) 4 (c (q . 9) (c (a (q . 2) (c (q . 2) (c (q . 13) (q)))) (q)))) (q (q . 1))) (q . 1)) (q . 1))) (q . 1))) ((q . \"defmacro\") (c \"list\" (c (f (q . 1)) (c (c \"mod\" (c (f (r (q . 1))) (c (f (r (r (q . 1)))) (q)))) (q)))))) (q (65 5) (66 11) (88 2)))) (c (\"opt\" (\"com\" (q 16 65 66) (q () () ((q . \"list\") (a (q (q . 97) (q (q . 97) (q . 2) (c (q . 2) (c (q . 3) (q)))) (c (q (q . 97) (i (q . 5) (q (q . 99) 4 (c (q . 9) (c (a (q . 2) (c (q . 2) (c (q . 13) (q)))) (q)))) (q (q . 1))) (q . 1)) (q . 1))) (q . 1))) ((q . \"defmacro\") (c \"list\" (c (f (q . 1)) (c (c \"mod\" (c (f (r (q . 1))) (c (f (r (r (q . 1)))) (q)))) (q)))))) (q (65 5) (66 11) (88 2)))) 1)))".to_string());
}

#[test]
fn children_optimizer_example() {
    let src = "(c (a (q 1 . 1) 1) (a (q . 2) 1))".to_string();
    assert_eq!(test_children_optimizer(src), "(c (q . 1) 2)");
}

#[test]
fn constant_optimizer_example() {
    let src = "(c (q . 29041) (c (c (q . \"unquote\") (c (c (a (q 1 . \"macros\") (q . 1)) (a (q 1) (q . 1))) (q))) (q)))".to_string();
    assert_eq!(
        test_constant_optimizer(src),
        "(q 29041 (\"unquote\" (\"macros\")))".to_string()
    );
}

#[test]
fn test_sub_args_1() {
    let src = "(5 . 1)".to_string();
    assert_eq!(test_sub_args(src), "(f (r 1))".to_string());
}

#[test]
fn test_path_optimizer_3() {
    let src = "(sha256 (r 1))".to_string();
    assert_eq!(test_optimizer(src), "(sha256 3)");
}

#[test]
fn test_path_optimizer_5() {
    let src = "(sha256 (f (r 1)))".to_string();
    assert_eq!(test_optimizer(src), "(sha256 5)");
}

#[test]
fn children_optimizer_test_2() {
    let src = "(c (a (q 1) 1) (a (q 1) 1))".to_string();
    assert_eq!(test_children_optimizer(src), "(c () ())");
}

#[test]
fn test_optimizer_q_empty_list() {
    let src = "(a (q 1) 1)".to_string();
    assert_eq!(test_optimizer(src), "()");
}

#[test]
fn seems_constant_quote_test() {
    let mut allocator = Allocator::new();
    let src = "(q . 15)".to_string();
    let input_ir = read_ir(&src, 0).unwrap();
    let assembled = assemble_from_ir(&mut allocator, Rc::new(input_ir)).unwrap();
    assert_eq!(seems_constant(&mut allocator, assembled), true);
}

#[test]
fn test_optimize_1() {
    let src = "(\"opt\" (\"com\" (q 29041 (\"opt\" (\"com\" (q \"unquote\" \"BODY\") (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) (q (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))))) (q (\"BODY\" 2))))".to_string();
    assert_eq!(
        test_optimizer(src),
        "(q 4 (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))".to_string()
    );
}
