use crate::tests::compiler::compiler::run_string;
use crate::tests::compiler::repl::test_repl_outcome;

// All tests needed for rest calls:
//
// Call from non-inline to inline
// 01 - Too many arguments, no tail position argument to absorb.
// 02 - Too many arguments, tail position argument to absorb, &rest.
// 03 - Too many arguments, tail position argument to absorb, no &rest.
// 04 - Exact number of arguments, no tail position argument.
// 05 - Exact number of arguments, tail position argument, &rest.
// 06 - Exact number of arguments, tail position argument, no &rest.
// 07 - Too few arguments, &rest, tail.
// 08 - Too few arguments, &rest, no tail.
// 09 - Too few arguments, no &rest, tail. (error)
// 10 - Too few arguments, &rest, no tail.
// Call from inline to inline
// 11 - Too many arguments, no tail position argument to absorb.
// 12 - Too many arguments, tail position argument to absorb, &rest.
// 13 - Too many arguments, tail position argument to absorb, no &rest.
// 14 - Exact number of arguments, no tail position argument.
// 15 - Exact number of arguments, tail position argument, &rest.
// 16 - Exact number of arguments, tail position argument, no &rest.
// 17 - Too few arguments, &rest, tail.
// 18 - Too few arguments, &rest, no tail.
// 19 - Too few arguments, no &rest, tail. (error)
// 20 - Too few arguments, &rest, no tail.
// Call from non-inline to non-inline
// 21 - Too many arguments, no tail position argument to absorb.
// 22 - Too many arguments, tail position argument to absorb, &rest.
// 23 - Too many arguments, tail position argument to absorb, no &rest.
// 24 - Exact number of arguments, no tail position argument.
// 25 - Exact number of arguments, tail position argument, &rest.
// 26 - Exact number of arguments, tail position argument, no &rest.
// 27 - Too few arguments, &rest, tail.
// 28 - Too few arguments, &rest, no tail.
// 29 - Too few arguments, no &rest, tail. (not error)
// 30 - Too few arguments, &rest, no tail.
// Call from inline to non-inline
// 31 - Too many arguments, no tail position argument to absorb.
// 32 - Too many arguments, tail position argument to absorb, &rest.
// 33 - Too many arguments, tail position argument to absorb, no &rest.
// 34 - Exact number of arguments, no tail position argument.
// 35 - Exact number of arguments, tail position argument, &rest.
// 36 - Exact number of arguments, tail position argument, no &rest.
// 37 - Too few arguments, &rest, tail.
// 38 - Too few arguments, &rest, no tail.
// 39 - Too few arguments, no &rest, tail. (not error)
// 40 - Too few arguments, &rest, no tail.
//
#[test]
fn test_simple_inline_toomany_args_01() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B) (+ A B))

  (F X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_simple_inline_toomany_args_improper_tail_02() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun sum (Xs)
    (if Xs
      (+ (f Xs) (sum (r Xs)))
      ()
      )
    )

  (defun-inline F (A B . C) (* A B (sum C)))

  (F X Y 99 101 &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (301 313))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "28490");
}

#[test]
fn test_simple_inline_toomany_args_improper_no_tail_03() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun sum (Xs)
    (if Xs
      (+ (f Xs) (sum (r Xs)))
      ()
      )
    )

  (defun-inline F (A B . C) (* A B (sum C)))

  (F X Y 99 101)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "7000");
}

#[test]
fn test_simple_inline_exact_no_tails_04() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline F (A B) (* A B))

  (F X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_simple_inline_exact_improper_tail_05() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B . C) (* A B C))

  (F X Y &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "315");
}

#[test]
fn test_simple_inline_exact_improper_no_tail_06() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline F (A B . C) (+ A B C))

  (F X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_simple_inline_exact_toofew_improper_tail_07() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B C . D) (list A B C (f D)))

  (F X Y &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101 103)");
}

#[test]
fn test_simple_inline_exact_toofew_tail_08() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B C) (list A B C))

  (F X Y &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_simple_inline_exact_toofew_improper_no_tail_09() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline F (A B C . D) (list A B C (f D)))

  (F X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string());
    if let Err(e) = res {
        assert!(e.1.contains("Lookup for argument 3"));
    } else {
        assert!(false);
    }
}

#[test]
fn test_simple_inline_exact_toofew_tail_10() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B C) (list A B C))

  (F X Y &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_inline_inline_toomany_args_11() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B) (+ A B))

  (defun-inline G (X Y Z) (F X Y Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_inline_inline_toomany_args_improper_tail_12() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline return-list (Xs) Xs)

  (defun-inline F (A B . C) (list A B (return-list C)))

  (F X Y 99 101 &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (301 313))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 (c e 301 313))");
}

#[test]
fn test_simple_inline_toomany_args_improper_no_tail_13() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline return-list (Xs) Xs)

  (defun-inline F (A B . C) (list A B (return-list C)))

  (F X Y 99 101)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 (c e))");
}

#[test]
fn test_inline_inline_exact_no_tails_14() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline F (A B) (* A B))

  (defun-inline G (A B) (F A B))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_inline_inline_exact_improper_tail_15() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B . C) (* A B C))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "315");
}

#[test]
fn test_inline_inline_exact_improper_no_tail_16() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline F (A B . C) (+ A B C))

  (defun-inline G (X Y) (F X Y))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_simple_inline_exact_toofew_improper_tail_17() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B C . D) (list A B C (f D)))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101 103)");
}

#[test]
fn test_inline_inline_exact_toofew_tail_18() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B C) (list A B C))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_inline_inline_exact_toofew_improper_no_tail_19() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun-inline F (A B C . D) (list A B C (f D)))

  (defun-inline G (X Y) (F X Y))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string());
    if let Err(e) = res {
        assert!(e.1.contains("Lookup for argument 3"));
    } else {
        assert!(false);
    }
}

#[test]
fn test_simple_inline_exact_toofew_tail_20() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun-inline F (A B C) (list A B C))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_ni_ni_toomany_args_21() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B) (+ A B))

  (defun G (X Y Z) (F X Y Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_ni_ni_toomany_args_improper_tail_22() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun return-list (Xs) Xs)

  (defun F (A B . C) (list A B (return-list C)))

  (F X Y 99 101 &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (301 313))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 (c e 301 313))");
}

#[test]
fn test_simple_inline_toomany_args_improper_no_tail_23() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun return-list (Xs) Xs)

  (defun F (A B . C) (list A B (return-list C)))

  (F X Y 99 101)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 (c e))");
}

#[test]
fn test_ni_ni_exact_no_tails_24() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B) (* A B))

  (defun G (A B) (F A B))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_ni_ni_exact_improper_tail_25() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B . C) (* A B C))

  (defun G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "315");
}

#[test]
fn test_ni_ni_rest_call_25() {
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
fn test_ni_ni_exact_improper_no_tail_26() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B . C) (+ A B C))

  (defun G (X Y) (F X Y))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_simple_inline_exact_toofew_improper_tail_27() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B C . D) (list A B C (f D)))

  (defun G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101 103)");
}

#[test]
fn test_ni_ni_exact_toofew_tail_28() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_ni_ni_exact_toofew_improper_no_tail_29() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B C . D) (list A B C (f D)))

  (defun G (X Y) (F X Y))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string());
    if let Err(e) = res {
        assert!(e.1.contains("bad path"));
    } else {
        assert!(false);
    }
}

#[test]
fn test_ni_ni_exact_toofew_tail_30() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_inline_ni_toomany_args_31() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B) (+ A B))

  (defun-inline G (X Y Z) (F X Y Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_inline_ni_toomany_args_improper_tail_32() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun return-list (Xs) Xs)

  (defun F (A B . C) (list A B (return-list C)))

  (F X Y 99 101 &rest Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (301 313))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 (c e 301 313))");
}

#[test]
fn test_simple_inline_toomany_args_improper_no_tail_33() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun return-list (Xs) Xs)

  (defun F (A B . C) (list A B (return-list C)))

  (F X Y 99 101)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 (c e))");
}

#[test]
fn test_inline_ni_exact_no_tails_34() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B) (* A B))

  (defun-inline G (A B) (F A B))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "35");
}

#[test]
fn test_inline_ni_exact_improper_tail_35() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B . C) (* A B C))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 9)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "315");
}

#[test]
fn test_simple_rest_call_inline_35() {
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
fn test_inline_ni_exact_improper_no_tail_36() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B . C) (+ A B C))

  (defun-inline G (X Y) (F X Y))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "12");
}

#[test]
fn test_simple_inline_exact_toofew_improper_tail_37() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B C . D) (list A B C (f D)))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101 103)");
}

#[test]
fn test_inline_ni_exact_toofew_tail_38() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_inline_ni_exact_toofew_improper_no_tail_39() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B C . D) (list A B C (f D)))

  (defun-inline G (X Y) (F X Y))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string());
    if let Err(e) = res {
        assert!(e.1.contains("bad path"));
    } else {
        assert!(false);
    }
}

#[test]
fn test_inline_ni_exact_toofew_tail_40() {
    let prog = indoc! {"
(mod (X Y Z)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun-inline G (X Y Z) (F X Y &rest Z))

  (G X Y Z)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7 (101 103))".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 101)");
}

#[test]
fn test_repl_01() {
    assert_eq!(
        test_repl_outcome(vec!["(defun-inline F (A B) (+ A B))", "(F 5 7 9)"])
            .unwrap()
            .unwrap(),
        "(q . 12)"
    );
}

const REPL_TEST_SUM: &str = indoc! {"
(defun sum (Xs)
    (if Xs
      (+ (f Xs) (sum (r Xs)))
      ()
      )
    )"};

#[test]
fn test_repl_02() {
    assert_eq!(
        test_repl_outcome(vec![
            REPL_TEST_SUM,
            "(defun-inline F (A B . C) (* A B (sum C)))",
            "(F 5 7 99 101 &rest (list 301 313))"
        ])
        .unwrap()
        .unwrap(),
        "(q . 28490)"
    );
}

#[test]
fn test_repl_03() {
    assert_eq!(
        test_repl_outcome(vec![
            REPL_TEST_SUM,
            "(defun-inline F (A B . C) (* A B (sum C)))",
            "(F 5 7 99 101)"
        ])
        .unwrap()
        .unwrap(),
        "(q . 7000)"
    );
}

#[test]
fn test_repl_04() {
    assert_eq!(
        test_repl_outcome(vec!["(defun F (A B) (* A B))", "(F 5 7)"])
            .unwrap()
            .unwrap(),
        "(q . 35)"
    );
}

#[test]
fn test_repl_05() {
    assert_eq!(
        test_repl_outcome(vec!["(defun F (A B . C) (* A B C))", "(F 5 7 &rest 9)"])
            .unwrap()
            .unwrap(),
        "(q . 315)"
    );
}

#[test]
fn test_repl_06() {
    assert_eq!(
        test_repl_outcome(vec!["(defun F (A B . C) (+ A B C))", "(F 5 7)"])
            .unwrap()
            .unwrap(),
        "(q . 12)"
    );
}

#[test]
fn test_repl_07() {
    assert_eq!(
        test_repl_outcome(vec![
            "(defun F (A B C . D) (list A B C (f D)))",
            "(F 5 7 &rest (list 101 103))"
        ])
        .unwrap()
        .unwrap(),
        "(q 5 7 101 103)"
    );
}

#[test]
fn test_repl_08() {
    assert_eq!(
        test_repl_outcome(vec![
            "(defun F (A B C) (list A B C))",
            "(F 5 7 &rest (list 101 103))"
        ])
        .unwrap()
        .unwrap(),
        "(q 5 7 101)"
    );
}

#[test]
fn test_repl_09() {
    assert!(
        test_repl_outcome(vec!["(defun F (A B C . D) (list A B C (f D)))", "(F 5 7)"]).is_err()
    );
}

#[test]
fn test_repl_10() {
    assert_eq!(
        test_repl_outcome(vec![
            "(defun F (A B C) (list A B C))",
            "(F 5 7 &rest (list 101 103))"
        ])
        .unwrap()
        .unwrap(),
        "(q 5 7 101)"
    );
}

#[test]
fn test_compiler_tail_let_inline() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun-inline G (X Y) (F X &rest (let ((Q (+ Y 1))) (list Y Q))))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 8)");
}

#[test]
fn test_compiler_tail_let_ni() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun G (X Y) (F X &rest (let ((Q (+ Y 1))) (list Y Q))))

  (G X Y)
  )"}
    .to_string();
    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 8)");
}

#[test]
fn test_compiler_tail_assign_ni() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun G (X Y) (F X &rest (assign Q (+ Y 1) (list Y Q))))

  (G X Y)
  )"}
    .to_string();

    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 8)");
}

#[test]
fn test_compiler_tail_assign_inline() {
    let prog = indoc! {"
(mod (X Y)
  (include *standard-cl-21*)

  (defun F (A B C) (list A B C))

  (defun-inline G (X Y) (F X &rest (assign Q (+ Y 1) (list Y Q))))

  (G X Y)
  )"}
    .to_string();

    let res = run_string(&prog, &"(5 7)".to_string()).expect("should compile and run");
    assert_eq!(res.to_string(), "(5 7 8)");
}

#[test]
fn test_repl_tail_let() {
    assert_eq!(
        test_repl_outcome(vec![
            "(defun F (A B C D) (list A B C D))",
            "(F 5 7 &rest (let ((Q (list 101 103))) (c 99 Q)))"
        ])
        .unwrap()
        .unwrap(),
        "(q 5 7 99 101)"
    );
}
