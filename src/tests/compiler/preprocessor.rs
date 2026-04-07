use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::preprocessor::preprocess;
use crate::compiler::sexp::parse_sexp;
use crate::compiler::srcloc::Srcloc;
use std::rc::Rc;

use crate::tests::compiler::compiler::run_string_strict;

#[test]
fn test_defmac_basic_0() {
    let prog = indoc! {"
    (mod (X)
      (defmac double-arg (A) (list (string->symbol (string-append (symbol->string A) \"1\")) (string->symbol (string-append (symbol->string A) \"2\"))))
      (defun strange (double-arg X) (+ X1 X2))
      (strange X (* 2 X))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "9");
}

#[test]
fn test_defmac_basic_shared_constant() {
    let prog = indoc! {"
    (mod (X)
      (defconstant twostring \"2\")
      (defmac double-arg (A) (list (string->symbol (string-append (symbol->string A) \"1\")) (string->symbol (string-append (symbol->string A) twostring))))
      (defun strange (double-arg X) (+ X1 X2))
      (c twostring (strange X (* 2 X)))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(\"2\" . 9)");
}

#[test]
fn test_defmac_basic_shared_constant_not_string_with_string_operator() {
    let prog = indoc! {"
    (mod (X)
      (defconstant twostring 2)
      (defmac double-arg (A) (list (string->symbol (string-append (symbol->string A) \"1\")) (string->symbol (string-append (symbol->string A) twostring))))
      (defun strange (double-arg X) (+ X1 X2))
      (c twostring (strange X (* 2 X)))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string());
    assert!(res.is_err());
}

#[test]
fn test_defmac_basic_shared_constant_not_string_with_string_operator_fun() {
    let prog = indoc! {"
    (mod (X)
      (defconstant twostring \"2\")
      (defun make-arg-list (A) (list (string->symbol (string-append (symbol->string A) \"1\")) (string->symbol (string-append (symbol->string A) twostring))))
      (defmac double-arg (A) (make-arg-list A))
      (defun strange (double-arg X) (+ X1 X2))
      (c twostring (strange X (* 2 X)))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(\"2\" . 9)");
}

#[test]
fn test_defmac_basic_test_is_string_pos() {
    let prog = indoc! {"
    (mod (X)
      (defmac classify (S)
        (if (string? S)
          (qq (c 1 (unquote S)))
          (qq (c 2 (unquote S)))
          )
        )
      (c X (classify \"test\"))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 1 . \"test\")");
}

#[test]
fn test_defmac_basic_test_is_string_neg() {
    let prog = indoc! {"
    (mod (X)
      (defmac classify (S)
        (if (string? S)
          (qq (c 1 (unquote S)))
          (qq (c 2 (unquote S)))
          )
        )
      (c X (classify 99))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 2 . 99)");
}

#[test]
fn test_defmac_basic_test_is_symbol_pos() {
    let prog = indoc! {"
    (mod (X)
      (defmac classify (S)
        (if (symbol? S)
          (qq (c 1 (unquote (symbol->string S))))
          (qq (c 2 (unquote S)))
          )
        )
      (c X (classify test))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 1 . \"test\")");
}

#[test]
fn test_defmac_basic_test_is_symbol_neg() {
    let prog = indoc! {"
    (mod (X)
      (defmac classify (S)
        (if (symbol? S)
          (qq (c 1 (unquote S)))
          (qq (c 2 (unquote S)))
          )
        )
      (c X (classify \"test\"))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 2 . \"test\")");
}

#[test]
fn test_defmac_basic_test_is_number_pos() {
    let prog = indoc! {"
    (mod (X)
      (defmac classify (S)
        (if (number? S)
          (qq (c 1 (unquote S)))
          (qq (c 2 (unquote S)))
          )
        )
      (c X (classify 7))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 1 . 7)");
}

#[test]
fn test_defmac_basic_test_is_number_neg() {
    let prog = indoc! {"
    (mod (X)
      (defmac classify (S)
        (if (number? S)
          (qq (c 1 (unquote S)))
          (qq (c 2 (unquote S)))
          )
        )
      (c X (classify \"test\"))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 2 . \"test\")");
}

#[test]
fn test_defmac_extension_from_function() {
    let prog = indoc! {"
    (mod (X)
      (defun FX (X) (symbol->string X))
      (defmac F (X) (FX X))
      (c 3 (F X))
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(3 . \"X\")");
}

#[test]
fn test_defmac_if_extension() {
    let prog = indoc! {"
    (mod (X)
      (defun FX (X) (if X (number->string 1) 2))
      (defmac F (X) (c 1 (FX X)))
      (F X)
      )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(9)".to_string()).unwrap();
    assert_eq!(res.to_string(), "\"1\"");
}

#[test]
fn test_defmac_create_match_form() {
    let prog = indoc! {"
    ;; Make a simple list match ala ocaml/elm/haskell.
    ;;
    ;; The real version will be more elaborate.  This is a test case and a demo.
    (mod X
        (include *standard-cl-21*)
        (defun list-nth (L N)
          (if N
            (list-nth (r L) (- N 1))
            (f L)
            )
          )

        (defun list-len (L N)
          (if L
            (list-len (r L) (+ N 1))
            N
            )
          )

        (defun funcall (name args) (c (string->symbol name) args))
        (defun quoted (X) (c 1 X))
        (defun equals (A B) (funcall \"=\" (list A B)))
        (defun emit-list-nth (L N) (funcall \"list-nth\" (list L N)))
        (defun emit-list-len (L N) (funcall \"list-len\" (list L N)))
        (defun emit-if (C T E) (funcall \"if\" (list C T E)))
        (defun emit-let (bindings body) (funcall \"let\" (list bindings body)))

        ;; Determine the size of each list as well as the constants and bindings
        ;; Return either a list of (number-of-elts matches bindings) or symbol.
        (defun build-matches-and-bindings (n pattern matches bindings)
            (if (not pattern)
                (list n matches bindings)
                (if (l pattern)
                    (if (symbol? (f pattern))
                        (build-matches-and-bindings (+ n 1) (r pattern) matches (c (c n (f pattern)) bindings))
                        (build-matches-and-bindings (+ n 1) (r pattern) (c (c n (f pattern)) matches) bindings)
                        )
                    pattern
                    )
                )
            )

        ;; Emit code that matches each match list for length and constants.
        (defun write-match-code (expr matches)
            (if (not matches)
                (quoted 1)
                (if (l (f matches))
                    (let*
                        (
                         (offset (f (f matches)))
                         (desire (r (f matches)))
                         (this-match (equals (quoted desire) (emit-list-nth expr (quoted offset))))
                         )
                      (if (not (r matches))
                          (list this-match)
                          (c this-match (write-match-code expr (r matches)))
                          )
                      )
                    (quoted 1)
                    )
                )
            )

        ;; Generate let bindings for the bindings indicated in the match.
        (defun let-bindings (expr bindings)
            (if (not bindings)
                ()
                (let
                    ((n (f (f bindings)))
                     (binding (r (f bindings)))
                     )
                  (c (list binding (emit-list-nth expr n)) (let-bindings expr (r bindings)))
                  )
                )
            )

        ;; Generate if expressions that match the indicates matches and return
        ;; the indicated code with bindings installed.
        (defun match-if (expr clauses)
            (if (not clauses)
                (list 8)
                (let
                    ((extracted-clause-data (build-matches-and-bindings 0 (f (f clauses)) () ()))
                     (code (f (r (f clauses))))
                     )
                  (if (l extracted-clause-data)
                      (let
                          (
                           (number-of-elts (f extracted-clause-data))
                           (matches (list-nth extracted-clause-data 1))
                           (bindings (list-nth extracted-clause-data 2))
                           )
                        (emit-if
                          (emit-if
                            (equals (emit-list-len expr 0) (quoted number-of-elts))
                            ;; then
                            (c (string->symbol \"logand\") (write-match-code expr matches))
                            ;; else
                            ()
                            )
                          ;; then
                          (emit-let (let-bindings expr bindings) code)
                          ;; else
                          (match-if expr (r clauses))
                          )
                        )
                      (emit-let (list (list extracted-clause-data expr)) code)
                      )
                     )
                )
            )

        ;; match as below.
        (defmac match (expr . matches) (match-if expr matches))

        (match X
          ((16 x y) (c 1 (+ x y)))
          ((3 () b c) c)
          ((3 1 b c) b)
          (x x)
          )
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(3 () 1001 1002)".to_string()).unwrap();
    assert_eq!(res.to_string(), "1002");
}

#[test]
fn test_defmac_stringq() {
    let prog = indoc! {"
      (mod ()
         (defmac is-string (X) (string? X))
         (list (is-string X) (is-string \"X\") (is-string 3))
         )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"()".to_string()).unwrap();
    assert_eq!(res.to_string(), "(() 1 ())");
}

#[test]
fn test_defmac_numberq() {
    let prog = indoc! {"
      (mod ()
         (defmac is-number (X) (number? X))
         (list (is-number X) (is-number \"X\") (is-number 3))
         )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"()".to_string()).unwrap();
    assert_eq!(res.to_string(), "(() () 1)");
}

#[test]
fn test_defmac_symbolq() {
    let prog = indoc! {"
      (mod ()
         (defmac is-symbol (X) (symbol? X))
         (list (is-symbol X) (is-symbol \"X\") (is-symbol 3))
         )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"()".to_string()).unwrap();
    assert_eq!(res.to_string(), "(1 () ())");
}

#[test]
fn test_defmac_string_to_symbol() {
    let prog = indoc! {"
      (mod ()
         (defmac is-symbol (X) (symbol? X))
         (list (is-symbol X) (is-symbol \"X\") (is-symbol 3))
         )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"()".to_string()).unwrap();
    assert_eq!(res.to_string(), "(1 () ())");
}

#[test]
fn test_defmac_string_to_symbol_converts() {
    let prog = indoc! {"
      (mod (X)
        (defmac let_pi (code) (qq (let (((unquote (string->symbol \"pi\")) 31415)) (unquote code))))
        (let_pi (+ pi X))
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(5)".to_string()).unwrap();
    assert_eq!(res.to_string(), "31420");
}

#[test]
fn test_defmac_string_needs_conversion() {
    let prog = indoc! {"
      (mod (X)
        (defmac let_pi (code) (qq (let ((\"pi\" 31415)) (unquote code))))
        (let_pi (+ pi X))
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(5)".to_string());
    eprintln!("res {res:?}");
    assert!(res.is_err());
}

#[test]
fn test_defmac_string_substr_0() {
    let prog = indoc! {"
      (mod (X)
        (defmac first-letter-of (Q)
          (let ((first-character (substring (symbol->string Q) 0 1)))
            (qq (c (unquote first-character) (unquote (string->symbol first-character))))
            )
          )
        (first-letter-of Xanadu)
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(5999)".to_string()).unwrap();
    assert_eq!(res.to_string(), "(\"X\" . 5999)");
}

#[test]
fn test_defmac_string_substr_bad() {
    let prog = indoc! {"
      (mod (test_variable_name)
        (defmac bind-tail-of-symbol (N Q CODE)
          (let*
            ((stringified (symbol->string Q))
             (slen (string-length stringified))
             (suffix (string->symbol (substring stringified N slen))))
            (qq (let (((unquote suffix) (r (unquote Q)))) (unquote CODE)))
            )
          )
        (bind-tail-of-symbol 100 test_variable_name (c 9999 variable_name))
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"((87 89 91))".to_string());
    assert!(res.is_err());
}

#[test]
fn test_defmac_string_to_number_0() {
    let prog = indoc! {"
      (mod (X_7)
        (defmac add-n-to (X)
          (let*
            ((stringified (symbol->string X))
             (slen (string-length stringified))
             (number-part (substring stringified (- slen 1) slen))
             (numeric-value (string->number number-part)))
            (qq (+ (unquote numeric-value) (unquote X)))
            )
          )
        (add-n-to X_7)
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(31)".to_string()).unwrap();
    assert_eq!(res.to_string(), "38");
}

#[test]
fn test_defmac_string_to_number_bad() {
    let prog = indoc! {"
      (mod (X_A)
        (defmac add-n-to (X)
          (let*
            ((stringified (symbol->string X))
             (slen (string-length stringified))
             (number-part (substring stringified (- slen 1) slen))
             (numeric-value (string->number number-part)))
            (qq (+ (unquote numeric-value) (unquote X)))
            )
          )
        (add-n-to X_A)
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(31)".to_string());
    assert!(res.is_err());
}

#[test]
fn test_defmac_number_to_string() {
    let prog = indoc! {"
      (mod (Q)
        (defmac with-my-length (X)
          (let*
            ((stringified (symbol->string X))
             (slen (string-length stringified)))
            (string->symbol (string-append stringified \"-\" (number->string slen)))
            )
          )
        (defun F (Xanadu-6) (+ (with-my-length Xanadu) 99))
        (F Q)
        )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"(37)".to_string()).unwrap();
    assert_eq!(res.to_string(), "136");
}

#[test]
fn test_preprocess_basic_list() {
    let file = "*test*";
    let input_form_set = indoc! {"(
         (include *strict-cl-21*)
         (list 1 2 3)
    )"};
    let parsed_forms =
        parse_sexp(Srcloc::start(file), input_form_set.bytes()).expect("should parse");
    let opts: Rc<dyn CompilerOpts> =
        Rc::new(DefaultCompilerOpts::new(file)).set_dialect(AcceptedDialect {
            stepping: Some(21),
            strict: true,
            int_fix: false,
        });
    let mut includes = Vec::new();
    let pp = preprocess(opts.clone(), &mut includes, parsed_forms[0].clone())
        .expect("should preprocess");
    assert_eq!(pp[pp.len() - 1].to_string(), "(4 1 (4 2 (4 3 ())))");
}

#[test]
fn test_preprocess_expansion_makes_numeric_operators() {
    let prog = indoc! {"
      (mod ()
         (include *strict-cl-21*)
         (defmac G () (com (4 \"test\" ())))
         (G)
         )
    "}
    .to_string();
    let res = run_string_strict(&prog, &"()".to_string()).unwrap();
    assert_eq!(res.to_string(), "(\"test\")");
}

#[test]
fn test_preprocessor_tours_includes_properly() {
    let prog = indoc! {"
      ( ;; Note: preprocessing is run in the list of the body forms.
        (include *strict-cl-21*)
        (include condition_codes.clvm)
        (include curry-and-treehash.clinc)
        ()
      )
    "}
    .to_string();
    let pname = "*test*";
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(pname))
        .set_search_paths(&["resources/tests".to_string()])
        .set_dialect(AcceptedDialect {
            stepping: Some(21),
            strict: true,
            int_fix: false,
        });
    let parsed = parse_sexp(Srcloc::start(pname), prog.bytes()).expect("should parse");
    let mut includes = Vec::new();
    let res = preprocess(opts, &mut includes, parsed[0].clone()).expect("should preprocess");
    let expected_lines = &[
        "(defmac __chia__primitive__if (A B C) (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @)))",
        "(defun __chia__if (ARGS) (a (i (r (r (r ARGS))) (com (qq (a (i (unquote (f ARGS)) (com (unquote (f (r ARGS)))) (com (unquote (__chia__if (r (r ARGS)))))) @))) (com (qq (a (i (unquote (f ARGS)) (com (unquote (f (r ARGS)))) (com (unquote (f (r (r ARGS)))))) @)))) @))",
        "(defmac if ARGS (__chia__if ARGS))",
        "(defun __chia__compile-list (args) (a (i args (com (c 4 (c (f args) (c (__chia__compile-list (r args)) ())))) (com ())) @))",
        "(defmac list ARGS (__chia__compile-list ARGS))",
        "(defun-inline / (A B) (f (divmod A B)))",
        "(defconstant *chialisp-version* 22)",
        "(defconstant AGG_SIG_UNSAFE 49)",
        "(defconstant AGG_SIG_ME 50)",
        "(defconstant CREATE_COIN 51)",
        "(defconstant RESERVE_FEE 52)",
        "(defconstant CREATE_COIN_ANNOUNCEMENT 60)",
        "(defconstant ASSERT_COIN_ANNOUNCEMENT 61)",
        "(defconstant CREATE_PUZZLE_ANNOUNCEMENT 62)",
        "(defconstant ASSERT_PUZZLE_ANNOUNCEMENT 63)",
        "(defconstant ASSERT_MY_COIN_ID 70)",
        "(defconstant ASSERT_MY_PARENT_ID 71)",
        "(defconstant ASSERT_MY_PUZZLEHASH 72)",
        "(defconstant ASSERT_MY_AMOUNT 73)",
        "(defconstant ASSERT_SECONDS_RELATIVE 80)",
        "(defconstant ASSERT_SECONDS_ABSOLUTE 81)",
        "(defconstant ASSERT_HEIGHT_RELATIVE 82)",
        "(defconstant ASSERT_HEIGHT_ABSOLUTE 83)",
        "(defconstant ONE 1)",
        "(defconstant TWO 2)",
        "(defconstant A_KW 2)",
        "(defconstant Q_KW 1)",
        "(defconstant C_KW 4)",
        "(defun-inline update-hash-for-parameter-hash (parameter-hash environment-hash) (sha256 TWO (sha256 ONE C_KW) (sha256 TWO (sha256 TWO (sha256 ONE Q_KW) parameter-hash) (sha256 TWO environment-hash (sha256 ONE ())))))",
        "(defun build-curry-list (reversed-curry-parameter-hashes environment-hash) (a (i reversed-curry-parameter-hashes (com (build-curry-list (r reversed-curry-parameter-hashes) (update-hash-for-parameter-hash (f reversed-curry-parameter-hashes) environment-hash))) (com environment-hash)) @))",
        "(defun-inline tree-hash-of-apply (function-hash environment-hash) (sha256 TWO (sha256 ONE A_KW) (sha256 TWO (sha256 TWO (sha256 ONE Q_KW) function-hash) (sha256 TWO environment-hash (sha256 ONE ())))))",
        "(defun puzzle-hash-of-curried-function (function-hash . reversed-curry-parameter-hashes) (tree-hash-of-apply function-hash (build-curry-list reversed-curry-parameter-hashes (sha256 ONE ONE))))",
        "()",
    ];
    for (i, r) in res.iter().enumerate() {
        assert_eq!(r.to_string(), expected_lines[i]);
    }
    assert_eq!(res.len(), expected_lines.len());
}
