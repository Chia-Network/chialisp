use crate::compiler::sexp::decode_string;
use crate::tests::classic::run::{do_basic_brun, do_basic_run};

#[derive(Debug)]
struct RueOperatorCase {
    args: Vec<&'static str>,
}

#[derive(Debug)]
struct RueOperatorSpec {
    name: &'static str,
    cases: Vec<RueOperatorCase>,
}

fn rue_operator_case(args: &[&'static str]) -> RueOperatorCase {
    RueOperatorCase {
        args: args.to_vec(),
    }
}

#[test]
fn test_rue_clvm_operators() {
    const G1: &str = "0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    const G1_ALT: &str = "0xb78f3c5f26a009143a3481dd09656ad7b3915e5ffd83aa2bc3ee49cd007a1f7a20791eb0036f86f457d3a6e80dbe5cde";
    const G2: &str = "0x8e955b28f90781845102a0e429c2684fc1e7585b0dc93fc127190f526397a48765f5126d60464b2298526e3cf629ea42128b704ce49fd029e0124eeb21156191662ff98cc0e8a304a8a1205fdada060b7f1181ace11dcc5d711a679b44b1df18";
    const COIN_PARENT: &str = "0x6e6561726c792d706172656e742d636f696e2d69640000000000000000000000";
    const COIN_PUZZLE: &str = "0x706172656e742d70757a7a6c652d686173680000000000000000000000000000";
    const BLS_PK: &str =
        "0x86243290bbcbfd9ae75bdece7981965350208eb5e99b04d5cd24e955ada961f8c0a162dee740be7bdc6c3c0613ba2eb1";
    const BLS_SIG: &str = "0xb00ab9a8af54804b43067531d96c176710c05980fccf8eee1ae12a4fd543df929cce860273af931fe4fdbc407d495f73114ab7d17ef08922e56625daada0497582340ecde841a9e997f2f557653c21c070119662dd2efa47e2d6c5e2de00eefa";
    const BLS_MSG: &str = "0x0102030405";
    const K1_PK: &str = "0x02390b19842e100324163334b16947f66125b76d4fa4a11b9ccdde9b7398e64076";
    const K1_HASH: &str = "0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2";
    const K1_SIG: &str = "0x481477e62a1d02268127ae89cc58929e09ad5d30229721965ae35965d098a5f630205a7e69f4cb8084f16c7407ed7312994ffbf87ba5eb1aee16682dd324943e";
    const R1_PK: &str = "0x033e1a1b2ccbc35883c60fdfc3f4a02175096ade6271fe85517ca5772594bbd0dc";
    const R1_HASH: &str = "0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2";
    const R1_SIG: &str = "0xeae2f488080919bd0a7069c24cdd9c6ce2db423861b0c9d4236cdadbd0005f6d8f3709e6eb19249fd9c8bea664aba35218e67ea4b0f2239488dc3147f336e1e6";
    const G1_DST: &str = "\"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_AUG_\"";
    const G2_DST: &str = "\"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_AUG_\"";

    let specs = vec![
        RueOperatorSpec {
            name: "q",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["1", "2"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "a",
            cases: vec![rue_operator_case(&["(q . 42)", "()"])],
        },
        RueOperatorSpec {
            name: "i",
            cases: vec![rue_operator_case(&["1", "2", "3"])],
        },
        RueOperatorSpec {
            name: "c",
            cases: vec![rue_operator_case(&["7", "3"])],
        },
        RueOperatorSpec {
            name: "f",
            cases: vec![rue_operator_case(&["(3 . 4)"])],
        },
        RueOperatorSpec {
            name: "r",
            cases: vec![rue_operator_case(&["(3 . 4)"])],
        },
        RueOperatorSpec {
            name: "l",
            cases: vec![rue_operator_case(&["(3 . 4)"])],
        },
        RueOperatorSpec {
            name: "x",
            cases: vec![rue_operator_case(&["99"])],
        },
        RueOperatorSpec {
            name: "=",
            cases: vec![
                rue_operator_case(&["5", "5"]),
                rue_operator_case(&["5", "6"]),
            ],
        },
        RueOperatorSpec {
            name: ">s",
            cases: vec![rue_operator_case(&["\"zest\"", "\"testing\""])],
        },
        RueOperatorSpec {
            name: "sha256",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "substr",
            cases: vec![
                rue_operator_case(&["\"hello\"", "1"]),
                rue_operator_case(&["\"hello\"", "1", "4"]),
            ],
        },
        RueOperatorSpec {
            name: "strlen",
            cases: vec![rue_operator_case(&["\"hello\""])],
        },
        RueOperatorSpec {
            name: "concat",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "+",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "-",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "*",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "/",
            cases: vec![rue_operator_case(&["7", "3"])],
        },
        RueOperatorSpec {
            name: "divmod",
            cases: vec![rue_operator_case(&["7", "3"])],
        },
        RueOperatorSpec {
            name: ">",
            cases: vec![rue_operator_case(&["7", "3"])],
        },
        RueOperatorSpec {
            name: "ash",
            cases: vec![rue_operator_case(&["7", "3"])],
        },
        RueOperatorSpec {
            name: "lsh",
            cases: vec![rue_operator_case(&["7", "3"])],
        },
        RueOperatorSpec {
            name: "logand",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "logior",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "logxor",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "lognot",
            cases: vec![rue_operator_case(&["1"])],
        },
        RueOperatorSpec {
            name: "point_add",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&[G1]),
                rue_operator_case(&[G1, G1_ALT]),
                rue_operator_case(&[G1, G1_ALT, G1, G1_ALT, G1]),
            ],
        },
        RueOperatorSpec {
            name: "pubkey_for_exp",
            cases: vec![rue_operator_case(&["1"])],
        },
        RueOperatorSpec {
            name: "not",
            cases: vec![rue_operator_case(&["1"])],
        },
        RueOperatorSpec {
            name: "any",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "all",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "softfork",
            cases: vec![rue_operator_case(&["279", "()", "(4 1 1)", "()"])],
        },
        RueOperatorSpec {
            name: "coinid",
            cases: vec![rue_operator_case(&[COIN_PARENT, COIN_PUZZLE, "1"])],
        },
        RueOperatorSpec {
            name: "g1_subtract",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&[G1]),
                rue_operator_case(&[G1, G1_ALT]),
                rue_operator_case(&[G1, G1_ALT, G1, G1_ALT, G1]),
            ],
        },
        RueOperatorSpec {
            name: "g1_multiply",
            cases: vec![rue_operator_case(&[G1, "2"])],
        },
        RueOperatorSpec {
            name: "g1_negate",
            cases: vec![rue_operator_case(&[G1])],
        },
        RueOperatorSpec {
            name: "g2_add",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&[G2]),
                rue_operator_case(&[G2, G2]),
            ],
        },
        RueOperatorSpec {
            name: "g2_subtract",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&[G2]),
                rue_operator_case(&[G2, G2]),
            ],
        },
        RueOperatorSpec {
            name: "g2_multiply",
            cases: vec![rue_operator_case(&[G2, "2"])],
        },
        RueOperatorSpec {
            name: "g2_negate",
            cases: vec![rue_operator_case(&[G2])],
        },
        RueOperatorSpec {
            name: "g1_map",
            cases: vec![
                rue_operator_case(&["0xabcdef"]),
                rue_operator_case(&["0xabcdef", G1_DST]),
            ],
        },
        RueOperatorSpec {
            name: "g2_map",
            cases: vec![
                rue_operator_case(&["0xabcdef"]),
                rue_operator_case(&["0xabcdef", G2_DST]),
            ],
        },
        RueOperatorSpec {
            name: "bls_pairing_identity",
            cases: vec![rue_operator_case(&[])],
        },
        RueOperatorSpec {
            name: "bls_verify",
            cases: vec![rue_operator_case(&[BLS_SIG, BLS_PK, BLS_MSG])],
        },
        RueOperatorSpec {
            name: "modpow",
            cases: vec![rue_operator_case(&["2", "6", "5"])],
        },
        RueOperatorSpec {
            name: "%",
            cases: vec![rue_operator_case(&["13", "10"])],
        },
        RueOperatorSpec {
            name: "keccak256",
            cases: vec![
                rue_operator_case(&[]),
                rue_operator_case(&["1"]),
                rue_operator_case(&["7", "3"]),
                rue_operator_case(&["1", "2", "3"]),
                rue_operator_case(&["1", "2", "3", "4", "5"]),
            ],
        },
        RueOperatorSpec {
            name: "secp256k1_verify",
            cases: vec![rue_operator_case(&[K1_PK, K1_HASH, K1_SIG])],
        },
        RueOperatorSpec {
            name: "secp256r1_verify",
            cases: vec![rue_operator_case(&[R1_PK, R1_HASH, R1_SIG])],
        },
    ];

    let prim_operator_names: Vec<String> = crate::compiler::prims::prims()
        .into_iter()
        .map(|(name, _)| decode_string(&name))
        .collect();
    let tested_operator_names: Vec<String> = specs.iter().map(|s| s.name.to_string()).collect();
    assert_eq!(tested_operator_names, prim_operator_names);

    let render_chialisp_body = |calls: &[String]| -> String {
        if calls.len() == 1 {
            calls[0].clone()
        } else {
            let mut out = "()".to_string();
            for call in calls.iter().rev() {
                out = format!("(c {call} {out})");
            }
            out
        }
    };
    let render_clvm_list = |calls: &[String]| -> String {
        if calls.len() == 1 {
            calls[0].clone()
        } else {
            let mut out = "()".to_string();
            for call in calls.iter().rev() {
                out = format!("(c {call} {out})");
            }
            out
        }
    };

    for spec in specs.iter() {
        let mut next_name_idx = 0usize;
        let mut parameter_names = Vec::new();
        let mut run_values = Vec::new();
        let mut call_text_by_name = Vec::new();
        let mut call_text_by_value = Vec::new();

        for case in spec.cases.iter() {
            if case.args.is_empty() {
                call_text_by_name.push(format!("({})", spec.name));
                call_text_by_value.push(format!("({})", spec.name));
                continue;
            }

            let mut case_arg_names = Vec::new();
            let mut case_arg_values = Vec::new();
            for arg in case.args.iter() {
                let arg_name = format!("A{next_name_idx}");
                next_name_idx += 1;
                parameter_names.push(arg_name.clone());
                run_values.push((*arg).to_string());
                case_arg_names.push(arg_name);
                case_arg_values.push(format!("(q . {arg})"));
            }
            call_text_by_name.push(format!("({} {})", spec.name, case_arg_names.join(" ")));
            call_text_by_value.push(format!("({} {})", spec.name, case_arg_values.join(" ")));
        }

        let body_by_name = render_chialisp_body(&call_text_by_name);
        let expected_body = if spec.name == "q" {
            render_clvm_list(&call_text_by_name)
        } else {
            render_clvm_list(&call_text_by_value)
        };
        let expected_result = do_basic_brun(&vec!["brun".to_string(), expected_body])
            .trim()
            .to_string();
        let run_input = if run_values.is_empty() {
            "()".to_string()
        } else {
            format!("({})", run_values.join(" "))
        };
        let params = parameter_names.join(" ");

        let classic_program_source = format!("(mod ({params}) {body_by_name})");
        let classic_compiled =
            do_basic_run(&vec!["run".to_string(), classic_program_source.clone()])
                .trim()
                .to_string();
        assert!(
            !classic_compiled.starts_with("FAIL"),
            "classic compile failed for {}",
            spec.name
        );
        let classic_output = do_basic_brun(&vec![
            "brun".to_string(),
            classic_compiled.clone(),
            run_input.clone(),
        ])
        .trim()
        .to_string();
        assert_eq!(
            classic_output, expected_result,
            "classic output mismatch for operator {} with source {} and input {}",
            spec.name, classic_program_source, run_input
        );

        let rue_program_source =
            format!("(mod ({params}) (include *standard-cl-rue1*) {body_by_name})");
        let rue_compiled = do_basic_run(&vec!["run".to_string(), rue_program_source.clone()])
            .trim()
            .to_string();
        assert!(
            !rue_compiled.starts_with("FAIL"),
            "rue compile failed for {}",
            spec.name
        );
        let rue_output = do_basic_brun(&vec![
            "brun".to_string(),
            rue_compiled.clone(),
            run_input.clone(),
        ])
        .trim()
        .to_string();
        assert_eq!(
            rue_output, classic_output,
            "rue output mismatch for operator {} with classic source {} and rue source {} and classic compiled {} and rue compiled {} and input {}",
            spec.name, classic_program_source, rue_program_source, classic_compiled, rue_compiled, run_input
        );
    }
}

const KECCAK_TEST_SIG: &str = "\"baz(uint32,bool)\"";
const KECCAK_TEST_RESULT: &str =
    "0xcdcd77c0992ec5bbfc459984220f8c45084cc24d9b6efed1fae540db8de801d2";

#[test]
fn test_keccak_compilation() {
    for p in [
        "(mod X (keccak256 X))",
        "(mod X (include *standard-cl-24*) (keccak256 X))",
    ]
    .iter()
    {
        let program = do_basic_run(&vec!["run".to_string(), p.to_string()]);
        let result = do_basic_brun(&vec![
            "brun".to_string(),
            program,
            KECCAK_TEST_SIG.to_string(),
        ]);
        assert_eq!(result.trim(), KECCAK_TEST_RESULT,);
    }
}

#[test]
fn test_keccak_opversion() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "--operators-version".to_string(),
        "1".to_string(),
        "(mod () (keccak256 999))".to_string(),
    ]);
    assert_eq!(program.trim(), "FAIL: unimplemented operator 62");
}

#[test]
fn test_reproduce_variable_repr_bug_deinline() {
    let source_program = "(mod (A) (include *standard-cl-24*) (defun F (X Y Z)
    (assign Q X R Q (list R Y Z))) (F &rest A))";

    let compile = |p: &str| do_basic_run(&vec!["run".to_string(), p.to_string()]);

    let output = compile(&source_program);
    // Produce the same program 30 times.  Demonstrates that this program didn't
    // produce a stable output (otherwise we wouldn't know the bug was fixed).
    let mut different = false;
    for _ in 0..30 {
        different |= output != compile(&source_program);
    }

    assert!(different);

    // Bump sigil
    let new_program = &source_program.replace("cl-24", "cl-25");

    let output = compile(&new_program);
    // Produce the same program 30 times.  The output should not be unstable.
    for _ in 0..30 {
        assert_eq!(output, compile(&new_program));
    }
}

#[test]
fn test_let_star_3_deep_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (a) (include *standard-cl-rue1*) (let* ((x (+ a 1)) (y (+ x 1)) (z (* a y))) (+ x y z)))".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), program, "(100)".to_string()]);
    assert_eq!(result.trim().to_string(), "10403".to_string());
}

#[test]
fn test_let_parallel_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (a) (include *standard-cl-rue1*) (let ((x (+ a 1)) (y (+ a 2))) (+ x y)))"
            .to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), program, "(100)".to_string()]);
    assert_eq!(result.trim().to_string(), "203".to_string());
}

#[test]
fn test_assign_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (a) (include *standard-cl-rue1*) (assign x (+ a 1) (+ x 1)))".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), program, "(100)".to_string()]);
    assert_eq!(result.trim().to_string(), "102".to_string());
}

#[test]
fn test_assign_destructure_reports_unsupported_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (a) (include *standard-cl-rue1*) (assign (x y) (list a a) (+ x y)))".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), program, "(100)".to_string()]);
    assert_eq!(result.trim().to_string(), "200".to_string());
}

#[test]
fn test_at_capture_destructure_1_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (A (@ Z (B C)) D) (include *standard-cl-rue1*) A)".to_string(),
    ]);
    assert_eq!(program.trim(), "2");
}

#[test]
fn test_at_capture_destructure_2_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (A (@ Z (B C)) D) (include *standard-cl-rue1*) Z)".to_string(),
    ]);
    assert_eq!(program.trim(), "5");
}

#[test]
fn test_at_capture_destructure_3_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (A (@ Z (B C)) D) (include *standard-cl-rue1*) B)".to_string(),
    ]);
    assert_eq!(program.trim(), "9");
}

#[test]
fn test_at_capture_destructure_4_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (A (@ Z (B C)) D) (include *standard-cl-rue1*) C)".to_string(),
    ]);
    assert_eq!(program.trim(), "21");
}

#[test]
fn test_at_capture_destructure_5_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (A (@ Z (B C)) D) (include *standard-cl-rue1*) D)".to_string(),
    ]);
    assert_eq!(program.trim(), "11");
}

#[test]
fn test_assign_at_capture_destructure_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (D) (include *standard-cl-rue1*) (assign (@ A (B C)) (list 2 3) (+ (l A) B C D)))"
            .to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), program, "(4)".to_string()]);
    assert_eq!(result.trim(), "10");
}

#[test]
fn test_return_function_can_be_run_no_env() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-rue1*) (defun F (X) (+ X 1)) F)".to_string(),
    ]);
    let function_f = do_basic_brun(&vec!["brun".to_string(), program]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), function_f, "(3)".to_string()]).trim(),
        "4"
    );
}

#[test]
fn test_return_function_can_be_run_env_env() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-rue1*) (defun G (X) (* X 3)) (defun F (X) (G (+ X 1))) F)"
            .to_string(),
    ]);
    let function_f = do_basic_brun(&vec!["brun".to_string(), program]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), function_f, "(3)".to_string()]).trim(),
        "12"
    );
}

#[test]
fn test_lambda_include_capture_rue() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (S) (include *standard-cl-rue1*) (a (lambda ((& S) X) (+ S X)) (list 5)))"
            .to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), program, "(3)".to_string()]);
    assert_eq!(result.trim(), "8");
}

#[test]
fn test_module_level_constants_defconst_and_defconstant_rue() {
    let defconst_program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-rue1*) (defconst A 41) (+ A 1))".to_string(),
    ]);
    let defconstant_program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-rue1*) (defconstant A 41) (+ A 1))".to_string(),
    ]);

    assert_eq!(
        do_basic_brun(&vec![
            "brun".to_string(),
            defconst_program,
            "()".to_string()
        ])
        .trim(),
        "42"
    );
    assert_eq!(
        do_basic_brun(&vec![
            "brun".to_string(),
            defconstant_program,
            "()".to_string()
        ])
        .trim(),
        "42"
    );
}

#[test]
fn test_defconst_calling_function_is_precomputed_rue() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-rue1*) (defun G (X) (* X 3)) (defconst K (G 4)) K)"
            .to_string(),
    ]);
    assert_eq!(compiled.trim(), "(1 . 12)");
}

#[test]
fn test_module_level_constant_in_main_function_and_returned_function_rue() {
    let program_main_use = do_basic_run(&vec![
        "run".to_string(),
        "(mod (X) (include *standard-cl-rue1*) (defun G (N) (* N 3)) (defconst K (G 4)) (+ X K))"
            .to_string(),
    ]);
    assert_eq!(
        do_basic_brun(&vec![
            "brun".to_string(),
            program_main_use,
            "(5)".to_string()
        ])
        .trim(),
        "17"
    );

    let program_function_use = do_basic_run(&vec![
        "run".to_string(),
        "(mod (X) (include *standard-cl-rue1*) (defun G (N) (* N 3)) (defconst K (G 4)) (defun add-k (Y) (+ Y K)) (add-k X))".to_string(),
    ]);
    assert_eq!(
        do_basic_brun(&vec![
            "brun".to_string(),
            program_function_use,
            "(5)".to_string()
        ])
        .trim(),
        "17"
    );

    let program_returns_function = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-rue1*) (defun G (N) (* N 3)) (defconst K (G 4)) (defun F (X) (+ X K)) F)".to_string(),
    ]);
    let function_f = do_basic_brun(&vec!["brun".to_string(), program_returns_function]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), function_f, "(3)".to_string()]).trim(),
        "15"
    );
}

#[test]
fn test_rue_no_repeat_large_constants() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (N) (include *standard-cl-rue1*) (defconst K (sha256 17)) (* K K K N))".to_string(),
    ]);
    let constant_to_count =
        b"33648946896879551350753991616036334622602839139780100591470253765180571691018";
    let occurrences = program
        .as_bytes()
        .windows(constant_to_count.len())
        .filter(|&w| w == constant_to_count)
        .count();
    assert_eq!(occurrences, 1);
}

#[test]
fn test_rue_if_in_inline() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (S) (include *standard-cl-rue1*) (defun-inline F (X) (if X 103 107)) (defun G (X) (F (- X 1))) (G S))".to_string(),
    ]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), program.clone(), "(1)".to_string()]).trim(),
        "107"
    );
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), program, "(2)".to_string()]).trim(),
        "103"
    );
}

#[test]
fn test_rue_argument_parent_access_1() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (X Y) (include *standard-cl-rue1*) (defun F (A B C D) (if (@ D 1) (+ A D) A)) (list (F 1 79 89 X) (F 2 79 89)))".to_string(),
    ]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), program, "(99)".to_string()]).trim(),
        "(100 2)"
    );
}

#[test]
fn test_rue_argument_parent_access_1_inline() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        format!("(mod (X Y) (include *standard-cl-rue1*) (defun-inline F (A B C D) (if (@ D 1) (+ A D) A)) (list (F 1 79 89 X) (F 2 79 89)))")
    ]);
    eprintln!("program {program}");
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), program, "(99)".to_string()]).trim(),
        "(100 2)"
    );
}

#[test]
fn test_rue_argument_parent_access_2() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (X Y) (include *standard-cl-rue1*) (defun F ((C D)) (list (@ D 1) (@ D 2))) (F (list X Y)))".to_string()
    ]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), program, "(99 103)".to_string()]).trim(),
        "((103) (99 103))"
    );
}

#[test]
fn test_rue_argument_parent_access_2_inline() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod (X Y) (include *standard-cl-rue1*) (defun-inline F ((C D)) (list (@ D 1) (@ D 2))) (F (list X Y)))".to_string()
    ]);
    assert_eq!(
        do_basic_brun(&vec!["brun".to_string(), program, "(99 103)".to_string()]).trim(),
        "((103) (99 103))"
    );
}
