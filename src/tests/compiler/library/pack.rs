//! Regression coverage for inline identities, &rest, and macro-generated constructors.
use super::evaluate;
use crate::compiler::sexp::SExp;
use std::rc::Rc;

fn compile(body: &str) -> Rc<SExp> {
    compile_with_library(body, "std.pack")
}

fn compile_with_library(body: &str, library: &str) -> Rc<SExp> {
    let source = format!(
        "(include *standard-cl-26*) (import {library})
         (export (A B C D atom optional nil) {body})"
    );
    super::compile(&source, "pack.clsp").unwrap_or_else(|error| panic!("{source}\n{error:?}"))
}

fn input(values: &[i32; 7]) -> String {
    format!(
        "({})",
        values
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    )
}

#[test]
fn test_pack_matches_explicit_constructors_without_extra_cost() {
    for (template, explicit) in [
        ("A", "A"),
        ("()", "()"),
        ("(A . B)", "(c A B)"),
        ("(D . (A . B))", "(c D (c A B))"),
        ("((A . B) . (C . D))", "(c (c A B) (c C D))"),
        ("(A B C D)", "(c A (c B (c C (c D ()))))"),
        ("(A (B . C) D)", "(c A (c (c B C) (c D ())))"),
        ("(A B . C)", "(c A (c B C))"),
        ("#(A B C)", "(c A (c B C))"),
        ("#(A B C D)", "(c (c A B) (c C D))"),
        ("(atom A)", "(c atom (c A ()))"),
        ("(optional #(A B C))", "(c optional (c (c A (c B C)) ()))"),
        ("(nil . (atom . optional))", "(c nil (c atom optional))"),
        ("(A . nil)", "(c A nil)"),
        (
            "(A B C D atom optional nil)",
            "(c A (c B (c C (c D (c atom (c optional (c nil ())))))))",
        ),
    ] {
        let packed = compile(&format!("(pack {template})"));
        let baseline = compile(explicit);
        for values in [[11, 22, 33, 44, 55, 66, 77], [0, 0, 0, 0, 0, 0, 0]] {
            let (cost, result) = evaluate(packed.clone(), &input(&values));
            let (baseline_cost, expected) = evaluate(baseline.clone(), &input(&values));
            assert_eq!(result, expected, "template {template}, values {values:?}");
            assert!(
                cost <= baseline_cost,
                "{template}: pack cost {cost} exceeds explicit constructor cost {baseline_cost}"
            );
        }
    }
}

#[test]
fn test_make_uses_pack_construction_after_stripping_annotations() {
    for (template, explicit) in [
        ("((atom A) . (optional #(B C D)))", "(c A (c B (c C D)))"),
        ("((atom A) B C . nil)", "(c A (c B (c C ())))"),
        ("(optional (A B . C))", "(c A (c B C))"),
    ] {
        let made = compile_with_library(&format!("(make {template})"), "std.shapes");
        let baseline = compile(explicit);
        for values in [[11, 22, 33, 44, 55, 66, 77], [0, 0, 0, 0, 0, 0, 0]] {
            let (cost, result) = evaluate(made.clone(), &input(&values));
            let (baseline_cost, expected) = evaluate(baseline.clone(), &input(&values));
            assert_eq!(result, expected, "template {template}");
            assert!(
                cost <= baseline_cost,
                "{template}: make cost {cost} exceeds explicit constructor cost {baseline_cost}"
            );
        }
    }
}
