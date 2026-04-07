use crate::tests::classic::run::{do_basic_brun, do_basic_run};

#[test]
fn test_fuzz_constant_gen() {
    let test_program = "(mod X %1 (defconst C %c) (concat C X))";

    let test_parallel_run = |test| {
        let final_program_classic = test_program.replace("%1", "").replace("%c", test);
        let final_program_23_1 = test_program
            .replace("%1", "(include *standard-cl-23.1*)")
            .replace("%c", test);
        let classic = do_basic_run(&vec!["run".to_string(), final_program_classic]);
        let new_ver = do_basic_run(&vec!["run".to_string(), final_program_23_1]);
        let classic_output = do_basic_brun(&vec!["brun".to_string(), classic, "0x01".to_string()]);
        let new_output = do_basic_brun(&vec!["brun".to_string(), new_ver, "0x01".to_string()]);
        assert_eq!(classic_output, new_output);
    };

    for test in [
        "0x00",
        "(concat 0x00 0x00)",
        "(concat 0 0x00)",
        "(concat 1 0x00)",
        "(concat -1 0x00)",
        "(concat -129 0x00)",
        "(concat 0x00 0)",
        "(concat 0x00 1)",
        "(concat 0x00 -1)",
        "(concat 0x00 -129)",
        "(concat 0x00 (sha256 0x00 0))",
        "(concat 0x00 (sha256 0 0x00 0))",
    ]
    .iter()
    {
        test_parallel_run(test);
    }
}

#[test]
fn test_truthy_in_constant_generation() {
    let test_program = "(mod X %1 (defconst C (if %c 1 0)) X)";

    let test_parallel_run = |test| {
        let final_program_classic = test_program.replace("%1", "").replace("%c", test);
        let final_program_23_1 = test_program
            .replace("%1", "(include *standard-cl-23.1*)")
            .replace("%c", test);
        let classic = do_basic_run(&vec!["run".to_string(), final_program_classic]);
        let new_ver = do_basic_run(&vec!["run".to_string(), final_program_23_1]);
        let classic_output = do_basic_brun(&vec!["brun".to_string(), classic, "0x01".to_string()]);
        let new_output = do_basic_brun(&vec!["brun".to_string(), new_ver, "0x01".to_string()]);
        assert_eq!(classic_output, new_output);
    };

    for test in [
        "0",
        "\"\"",
        "()",
        "0x00",
        "(concat 0 0)",
        "(concat 0x00 0x00)",
        "(concat 0 0x00)",
        "(concat 1 0x00)",
        "(concat -1 0x00)",
        "(concat -129 0x00)",
        "(concat 0x00 0)",
        "(concat 0x00 1)",
        "(concat 0x00 -1)",
        "(concat 0x00 -129)",
        "(concat 0x00 (sha256 0x00 0))",
        "(concat 0x00 (sha256 0 0x00 0))",
    ]
    .iter()
    {
        test_parallel_run(test);
    }
}
