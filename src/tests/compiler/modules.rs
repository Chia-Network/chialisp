use std::borrow::Borrow;
use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm::__type_compatibility__::{Bytes, Stream, UnvalidatedBytesFromType};
use crate::classic::clvm::serialize::{sexp_from_stream, SimpleCreateCLVMObject};
use crate::classic::clvm_tools::binutils::{assemble, disassemble};
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};

use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::{
    CompileErr, CompilerOpts, CompilerOutput, HasCompilerOptsDelegation,
};
use crate::compiler::dialect::detect_modern;
use crate::compiler::sexp::{decode_string, enlist, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;

pub enum DesiredOutcome<'a> {
    Error,
    ContentEquals,
    Run(&'a str),
}

use DesiredOutcome::{ContentEquals, Error, Run};

#[derive(Clone)]
pub struct TestModuleCompilerOpts {
    opts: Rc<dyn CompilerOpts>,
    written_files: Rc<RefCell<HashMap<String, Vec<u8>>>>,
}

impl TestModuleCompilerOpts {
    pub fn new(opts: Rc<dyn CompilerOpts>) -> Self {
        TestModuleCompilerOpts {
            opts: opts,
            written_files: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    pub fn get_written_file<'a>(&'a self, name: &str) -> Option<Vec<u8>> {
        let files_ref: &RefCell<HashMap<String, Vec<u8>>> = self.written_files.borrow();
        let files: &HashMap<String, Vec<u8>> = &files_ref.borrow();
        files.get(name).map(|f| f.to_vec())
    }
}

impl HasCompilerOptsDelegation for TestModuleCompilerOpts {
    fn compiler_opts(&self) -> Rc<dyn CompilerOpts> {
        self.opts.clone()
    }

    fn update_compiler_opts<F: FnOnce(Rc<dyn CompilerOpts>) -> Rc<dyn CompilerOpts>>(
        &self,
        f: F,
    ) -> Rc<dyn CompilerOpts> {
        let new_opts = f(self.opts.clone());
        Rc::new(TestModuleCompilerOpts {
            written_files: self.written_files.clone(),
            opts: new_opts,
        })
    }

    fn override_write_new_file(&self, target: &str, content: &[u8]) -> Result<(), CompileErr> {
        let mut wf: RefMut<'_, HashMap<String, Vec<u8>>> = self.written_files.borrow_mut();
        wf.insert(target.to_string(), content.to_vec());
        Ok(())
    }
}

pub struct HexArgumentOutcome<'a> {
    pub hexfile: &'a str,
    pub argument: &'a str,
    pub outcome: DesiredOutcome<'a>,
}

pub struct PerformCompileResult {
    pub compiled: CompilerOutput,
    pub source_opts: TestModuleCompilerOpts,
}

pub fn perform_compile_of_file(
    allocator: &mut Allocator,
    runner: Rc<dyn TRunProgram>,
    filename: &str,
    content: &str,
) -> Result<PerformCompileResult, CompileErr> {
    let loc = Srcloc::start(filename);
    let parsed: Vec<Rc<SExp>> = parse_sexp(loc.clone(), content.bytes()).expect("should parse");
    let listed = Rc::new(enlist(loc.clone(), &parsed));
    let nodeptr = convert_to_clvm_rs(allocator, listed.clone()).expect("should convert");
    let dialect = detect_modern(allocator, nodeptr);
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(filename))
        .set_dialect(dialect)
        .set_search_paths(&["resources/tests/module".to_string()]);
    let source_opts = TestModuleCompilerOpts::new(orig_opts);
    let opts: Rc<dyn CompilerOpts> = Rc::new(source_opts.clone());
    let mut symbol_table = HashMap::new();
    let mut includes = Vec::new();
    let compiled = compile_file(
        allocator,
        runner.clone(),
        opts,
        &content,
        &mut symbol_table,
        &mut includes,
    )?;
    Ok(PerformCompileResult {
        compiled,
        source_opts,
    })
}

pub fn hex_to_clvm(allocator: &mut Allocator, hex_data: &[u8]) -> clvmr::allocator::NodePtr {
    let mut hex_stream = Stream::new(Some(
        Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(decode_string(
            &hex_data,
        ))))
        .expect("should be valid hex"),
    ));
    sexp_from_stream(
        allocator,
        &mut hex_stream,
        Box::new(SimpleCreateCLVMObject {}),
    )
    .expect("hex data should decode as sexp")
    .1
}

fn test_compile_and_run_program_with_modules(
    filename: &str,
    content: &str,
    runs: &[HexArgumentOutcome],
) {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let compile_result = perform_compile_of_file(&mut allocator, runner.clone(), filename, content);

    let compile_result = if runs.is_empty() {
        assert!(compile_result.is_err());
        return;
    } else {
        compile_result.expect("Was expected to compile")
    };

    for run in runs.iter() {
        let hex_data = compile_result
            .source_opts
            .get_written_file(run.hexfile)
            .expect(&format!(
                "should have written hex data {} beside the source file",
                run.hexfile
            ));
        let compiled_node = hex_to_clvm(&mut allocator, &hex_data);

        if matches!(&run.outcome, ContentEquals) {
            let disassembled = disassemble(&allocator, compiled_node, None);
            assert_eq!(run.argument, disassembled);
            continue;
        }

        let assembled_env = assemble(&mut allocator, run.argument).expect("should assemble");
        let run_result = runner.run_program(&mut allocator, compiled_node, assembled_env, None);
        if let Run(res) = &run.outcome {
            let have_outcome = run_result.expect("expected success").1;
            assert_eq!(&disassemble(&mut allocator, have_outcome, None), res);
        } else {
            assert!(run_result.is_err());
        }
    }
}

#[test]
fn test_simple_module_compilation() {
    let filename = "resources/tests/module/modtest1.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/modtest1.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(3)",
                outcome: Run("3"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(13)",
                outcome: Error,
            },
        ],
    );
}

#[test]
fn test_simple_module_compilation_assign_rewrite() {
    let filename = "resources/tests/module/modtest1_assign.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/modtest1_assign.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(3)",
                outcome: Run("3"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(13)",
                outcome: Error,
            },
        ],
    );
}

#[test]
fn test_simple_module_compilation_lambda_rewrite() {
    let filename = "resources/tests/module/modtest1_lambda.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/modtest1_lambda.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(3 13)",
                outcome: Run("16"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(13 3)",
                outcome: Error,
            },
        ],
    );
}

#[test]
fn test_simple_module_compilation_lambda_rewrite_with_body() {
    let filename = "resources/tests/module/modtest2_lambda.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/modtest2_lambda.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(3 13)",
                outcome: Run("(+ 39)"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(13 3)",
                outcome: Error,
            },
        ],
    );
}

#[test]
fn test_simple_module_compilation_import_program_0() {
    let filename = "resources/tests/module/basic_program_import.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/basic_program_import.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "()",
                outcome: Run("(0x9df8d4d222276747372a532a1cd736cdd5c6800c39906c9695489d36286ed215 (+ 2 (q . 1)))")
            }
        ]
    );
}

#[test]
fn test_simple_module_compilation_import_program_1() {
    let filename = "resources/tests/module/two_program_import.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/two_program_import.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(13 73)",
                outcome: Run("(0xfab05b1fd610159ee4a46a120ece34b5a947f88057c4287f4506e513910a5d29 14 (a (q 16 5 (q . 1)) (c (q () 16 5 (q . 1)) 1)) 0x225de92687b39de0e3228e15d54bc9896f275027d5b6bb6bde309afbfb22b519 146 (a (q 18 5 (q . 2)) (c (q () 18 5 (q . 2)) 1)))")
            }
        ]
    );
}

#[test]
fn test_simple_module_compilation_import_classic_program() {
    let filename = "resources/tests/module/classic_program_import.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/classic_program_import.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "()",
                outcome: Run("(0x27a343d6617931e67a7eb27a41f7c4650b5fa79d8b5132af1b4eae959bdf2272 (* 2 (q . 13)))")
            }
        ]
    );
}

#[test]
fn test_function_with_argument_names_overlapping_primitives() {
    let filename = "resources/tests/module/test_prepend.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test_prepend.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "((1 2 3) (4 5 6))",
            outcome: Run("(q 2 3 4 5 6)"),
        }],
    );
}

#[test]
fn test_handcalc() {
    let filename = "resources/tests/module/test_handcalc.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test_handcalc.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "()",
            outcome: Run("()"),
        }],
    );
}

#[test]
fn test_factorial() {
    let filename = "resources/tests/module/test_factorial.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test_factorial.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "(4)",
            outcome: Run("24"),
        }],
    );
}

#[test]
fn test_deep_compare() {
    let filename = "resources/tests/module/test_deep_compare.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test_deep_compare.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(() 1)",
                outcome: Run("-1"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "(1 ())",
                outcome: Run("1"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "((1 2 3) (1 2 3))",
                outcome: Run("()"),
            },
            HexArgumentOutcome {
                hexfile: hex_filename,
                argument: "((1 2 3) (1 2 4))",
                outcome: Run("-1"),
            },
        ],
    );
}

#[test]
fn test_import_constant() {
    let filename = "resources/tests/module/test-import-constant.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test-import-constant.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "(5)",
            outcome: Run("6"),
        }],
    );
}

#[test]
fn test_export_constant() {
    let filename = "resources/tests/module/test-export-constant.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test-export-constant_add-5-and-9.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "()",
            outcome: Run("14"),
        }],
    );
}

#[test]
fn test_export_foreign_function() {
    let filename = "resources/tests/module/test-export-foreign-function.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test-export-foreign-function_factorial.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "(4)",
            outcome: Run("24"),
        }],
    );
}

#[test]
fn test_export_foreign_constant() {
    let filename = "resources/tests/module/test-export-foreign-constant.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test-export-foreign-constant_ONE.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "()",
            outcome: Run("1"),
        }],
    );
}

#[test]
fn test_import_renamed() {
    let filename = "resources/tests/module/test_factorial_renamed.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_filename = "resources/tests/module/test_factorial_renamed.hex";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_filename,
            argument: "(5)",
            outcome: Run("120"),
        }],
    );
}

#[test]
fn test_program_exporting_constant_from_program() {
    let filename = "resources/tests/module/test-export-constant-from-program.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let c_hex_filename = "resources/tests/module/test-export-constant-from-program_C.hex";
    let c_program_hex_filename = "resources/tests/module/programs/single-constant_C.hex";
    let c_value = "10197";

    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: c_hex_filename,
                argument: c_value,
                outcome: ContentEquals,
            },
            HexArgumentOutcome {
                hexfile: c_program_hex_filename,
                argument: c_value,
                outcome: ContentEquals,
            },
        ],
    );
}

#[test]
fn test_program_export_constant_and_function() {
    let filename = "resources/tests/module/test-export-constant-and-function.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let d_hex_filename = "resources/tests/module/test-export-constant-and-function_D.hex";
    let f_hex_filename = "resources/tests/module/test-export-constant-and-function_F.hex";
    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: d_hex_filename,
                argument: "(19191 (a (q 16 (q . 19191) 5) (c (q (+ (q . 19191) 5)) 1)) 0x3e6c399d8b10babad835468467a4b837036357ddfb8c320ba39a914c63152967 0xda99392a41a05d70cc42102af36ba888e964e57b587016e14e5e3faf66807dff)",
                outcome: ContentEquals,
            },
            HexArgumentOutcome {
                hexfile: f_hex_filename,
                argument: "(a (q 16 (q . 19191) 5) (c (q (+ (q . 19191) 5)) 1))",
                outcome: ContentEquals,
            },
        ]
    );
}

#[test]
fn test_detect_illegal_constant_arrangement() {
    let filename = "resources/tests/module/illegal-constant-arrangement-1.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    test_compile_and_run_program_with_modules(filename, &content, &[]);
}

#[test]
fn test_legal_all_tabled() {
    let filename = "resources/tests/module/legal-all-tabled.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_file = "resources/tests/module/legal-all-tabled_F.hex";
    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_file,
            argument: "(3)",
            outcome: Run("6"),
        }],
    );
}

#[test]
fn test_constant_with_lambda() {
    let filename = "resources/tests/module/tear-off-lambda-from-constant.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_file = "resources/tests/module/tear-off-lambda-from-constant_A.hex";
    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_file,
            argument: "(1)",
            outcome: Run("0x7ccbd86581a0f129fa2772ec828a9d46d8639d6a14d5b9e2b03d8aa039221ba4"),
        }],
    );
}

#[test]
fn test_constant_single_round() {
    let filename = "resources/tests/module/test_staged_constants.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let hex_file = "resources/tests/module/test_staged_constants_D.hex";
    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[HexArgumentOutcome {
            hexfile: hex_file,
            argument: "31",
            outcome: ContentEquals,
        }],
    );
}

#[test]
fn test_constant_multiple_rounds() {
    let filename = "resources/tests/module/test_staged_constants_multiple_iters.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let f_hex_file = "resources/tests/module/test_staged_constants_multiple_iters_F.hex";
    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: f_hex_file,
                argument: "(0x930f21b24db8799bc5a496643a8cfa81648421140318ce7f5eb7a67037fbb266 (a (q 16 5 (q . 3)) (c (q (+ 5 (q . 3))) 1)))",
                outcome: ContentEquals
            }
        ]
    );
}

#[test]
fn test_lib_reduce() {
    let filename = "resources/tests/module/test-reduce.clsp";
    let content = fs::read_to_string(filename).expect("file should exist");
    let f_hex_file = "resources/tests/module/test-reduce.clsp.hex";
    test_compile_and_run_program_with_modules(
        filename,
        &content,
        &[
            HexArgumentOutcome {
                hexfile: f_hex_file,
                argument: "()",
                outcome: ContentEquals
            }
        ]
    );
}