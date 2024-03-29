use std::borrow::Borrow;
use std::rc::Rc;

use regex::Regex;

use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{BodyForm, CompileForm, CompilerOpts};
use crate::compiler::frontend::{compile_bodyform, frontend};
use crate::compiler::optimize::bodyform::{
    replace_in_bodyform, retrieve_bodyform, visit_detect_in_bodyform, BodyformPathArc,
    PathDetectVisitorResult,
};
use crate::compiler::sexp::parse_sexp;
use crate::compiler::srcloc::Srcloc;

#[test]
fn test_bodyform_simple_traversal_0() {
    let progfile = "*test*";
    let srcloc = Srcloc::start(progfile);
    let parsed = parse_sexp(
        srcloc,
        indoc! {"
(mod ()
  (call1
    (let ((A 99) (B test)) (+ A B))
    (call2
      (mod () z)
      (lambda () r)
      )
    (call3 9)
    )
  )"}
        .bytes(),
    )
    .expect("should parse");
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(progfile));
    let compiled = frontend(opts.clone(), &parsed).expect("should fe");

    let tests: &[(Vec<BodyformPathArc>, Option<&str>)] = &[
        (
            vec![
                BodyformPathArc::CallArgument(1),
                BodyformPathArc::LetBinding(0),
            ],
            Some(r"99"),
        ),
        (
            vec![
                BodyformPathArc::CallArgument(2),
                BodyformPathArc::CallArgument(1),
            ],
            Some(r"\(mod \(\) z\)"),
        ),
        (
            vec![BodyformPathArc::CallArgument(1), BodyformPathArc::BodyOf],
            Some(r"\([+] A_[$]_[0-9]+ B_[$]_[0-9]+\)"),
        ),
        (
            vec![
                BodyformPathArc::CallArgument(1),
                BodyformPathArc::LetBinding(10),
            ],
            None,
        ),
        (
            vec![
                BodyformPathArc::CallArgument(1),
                BodyformPathArc::BodyOf,
                BodyformPathArc::BodyOf,
            ],
            None,
        ),
    ];
    for (traversal1, want) in tests.iter() {
        let retrieved1 = retrieve_bodyform(&traversal1, compiled.exp.borrow(), &|b| b.clone());
        if let Some(r) = want {
            let re = Regex::new(r).unwrap();
            assert!(re.is_match(&retrieved1.unwrap().to_sexp().to_string()));
        } else {
            assert!(retrieved1.is_none());
        }
    }
}

#[test]
fn test_replace_in_bodyform() {
    let progfile = "*test*";
    let srcloc = Srcloc::start(progfile);
    let parsed = parse_sexp(
        srcloc.clone(),
        indoc! {"
(mod ()
  (call1
    (let ((A 99) (B test)) (+ A B))
    (call2
      (mod () z)
      (lambda () r)
      )
    (call3 9)
    )
  )"}
        .bytes(),
    )
    .expect("should parse");
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(progfile));
    let compiled = frontend(opts.clone(), &parsed).expect("should fe");
    let compile_body = |b: &str| {
        let new_parsed = parse_sexp(srcloc.clone(), b.bytes()).unwrap();
        compile_bodyform(opts.clone(), new_parsed[0].clone()).unwrap()
    };

    // replacement-list, replace_with, expect
    let tests: &[(Vec<BodyformPathArc>, &str, Option<&str>)] = &[
        (
            vec![
                BodyformPathArc::CallArgument(1),
                BodyformPathArc::BodyOf,
                BodyformPathArc::CallArgument(2),
            ],
            "()",
            Some(
                r"(call1 (let ((A_[$]_[0-9]+ 99) (B_[$]_[0-9]+ test)) ([+] A_[$]_[0-9]+ ())) (call2 (mod () z) (lambda () r)) (call3 9))",
            ),
        ),
        (
            vec![
                BodyformPathArc::CallArgument(1),
                BodyformPathArc::LetBinding(1),
            ],
            "()",
            Some(
                r"(call1 (let ((A_[$]_[0-9]+ 99) (B_[$]_[0-9]+ ())) ([+] A_[$]_[0-9]+ B_[$]_[0-9]+)) (call2 (mod () z) (lambda () r)) (call3 9))",
            ),
        ),
        (
            vec![
                BodyformPathArc::CallArgument(2),
                BodyformPathArc::CallArgument(1),
                BodyformPathArc::BodyOf,
            ],
            "()",
            Some(
                r"(call1 (let ((A_[$]_[0-9]+ 99) (B_[$]_[0-9]+ test)) ([+] A_[$]_[0-9]+ B_[$]_[0-9]+)) (call2 (mod () ()) (lambda () r)) (call3 9))",
            ),
        ),
        (
            vec![
                BodyformPathArc::CallArgument(2),
                BodyformPathArc::CallArgument(2),
                BodyformPathArc::BodyOf,
            ],
            "()",
            Some(
                r"(call1 (let ((A_[$]_[0-9]+ 99) (B_[$]_[0-9]+ test)) ([+] A_[$]_[0-9]+ B_[$]_[0-9]+)) (call2 (mod () z) (lambda () ())) (call3 9))",
            ),
        ),
    ];

    for (path, replacement, expect) in tests.iter() {
        eprintln!("replacement {replacement}");
        let replace_spec = vec![PathDetectVisitorResult {
            path: path.clone(),
            subexp: compile_body(*replacement),
            context: (),
        }];
        let replaced = replace_in_bodyform(
            &replace_spec,
            compiled.exp.borrow(),
            &|spec: &PathDetectVisitorResult<()>, _old: &BodyForm| spec.subexp.clone(),
        );
        if let Some(r) = expect {
            let got = replaced.unwrap().to_sexp().to_string();
            eprintln!("got  {got}");
            eprintln!("want {r}");
            let escaped_re = r.replace("(", r"\(").replace(")", r"\)");
            let re = Regex::new(&escaped_re).unwrap();
            assert!(re.is_match(&got));
        } else {
            assert!(replaced.is_none());
        }
    }
}

fn make_test_case_for_visitor(program: &str) -> CompileForm {
    let progfile = "*test*";
    let srcloc = Srcloc::start(progfile);
    let parsed = parse_sexp(srcloc.clone(), program.bytes()).expect("should parse");
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(progfile));
    frontend(opts.clone(), &parsed).expect("should fe")
}

#[test]
fn test_visitor_0() {
    let compiled = make_test_case_for_visitor(indoc! {"
(mod ()
  (call1
    (let ((A 99) (B test)) (+ A B))
    (call2
      (mod () z)
      (lambda () r)
      )
    (call3 9)
    )
  )"});
    let res = visit_detect_in_bodyform(
        &|_path, _orig, _here| {
            let e: Result<Option<()>, ()> = Err(());
            e
        },
        compiled.exp.borrow(),
    );
    assert!(res.is_err());
    let res = visit_detect_in_bodyform(
        &|_path, _orig, here| {
            if here.to_sexp().to_string() == "z" {
                let res: Result<Option<()>, ()> = Ok(Some(()));
                return res;
            }
            Ok(None)
        },
        compiled.exp.borrow(),
    )
    .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].subexp.to_sexp().to_string(), "z");
    assert_eq!(
        res[0].path,
        vec![
            BodyformPathArc::CallArgument(2),
            BodyformPathArc::CallArgument(1),
            BodyformPathArc::BodyOf
        ]
    );
    let res = visit_detect_in_bodyform(
        &|_path, _orig, here| {
            if here.to_sexp().to_string() == "r" || here.to_sexp().to_string() == "99" {
                let res: Result<Option<()>, ()> = Ok(Some(()));
                return res;
            }
            Ok(None)
        },
        compiled.exp.borrow(),
    )
    .unwrap();
    assert_eq!(res.len(), 2);
    assert_eq!(res[0].subexp.to_sexp().to_string(), "99");
    assert_eq!(
        res[0].path,
        vec![
            BodyformPathArc::CallArgument(1),
            BodyformPathArc::LetBinding(0)
        ]
    );
    assert_eq!(res[1].subexp.to_sexp().to_string(), "r");
    assert_eq!(
        res[1].path,
        vec![
            BodyformPathArc::CallArgument(2),
            BodyformPathArc::CallArgument(2),
            BodyformPathArc::BodyOf
        ]
    );
}

#[test]
fn test_visitor_rest_alone() {
    let compiled = make_test_case_for_visitor(indoc! {"
        (mod (X)
          (F &rest X)
          )
    "});
    let res = visit_detect_in_bodyform(
        &|_path, _orig, here| {
            if here.to_sexp().to_string() == "X" {
                let res: Result<Option<()>, ()> = Ok(Some(()));
                return res;
            }

            Ok(None)
        },
        compiled.exp.borrow(),
    )
    .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].path, vec![BodyformPathArc::CallArgument(1)]);
}

#[test]
fn test_visitor_rest_with_list() {
    let compiled = make_test_case_for_visitor(indoc! {"
        (mod (X)
          (F X &rest (F &rest X))
          )
    "});
    let res = visit_detect_in_bodyform(
        &|_path, _orig, here| {
            if here.to_sexp().to_string() == "X" {
                let res: Result<Option<()>, ()> = Ok(Some(()));
                return res;
            }

            Ok(None)
        },
        compiled.exp.borrow(),
    )
    .unwrap();
    assert_eq!(res.len(), 2);
    assert_eq!(res[0].path, vec![BodyformPathArc::CallArgument(1)]);
    assert_eq!(
        res[1].path,
        vec![
            BodyformPathArc::CallArgument(2),
            BodyformPathArc::CallArgument(1)
        ]
    );
}
