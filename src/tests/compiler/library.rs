//! Compiler regressions using local snapshots of real Chialisp library code.
use std::{collections::HashMap, fs, rc::Rc};

use clvmr::Allocator;

use crate::classic::clvm_tools::comp_input::RunAndCompileInputData;
use crate::classic::clvm_tools::stages::stage_0::{
    DefaultProgramRunner, RunProgramOption, TRunProgram,
};
use crate::classic::platform::argparse::ArgumentValue;
use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::compiler::compile_file;
use crate::compiler::comptypes::{CompileErr, CompilerOutput};
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::tests::compiler::modules::TestModuleCompilerOpts;

mod pack;
mod shapes;

const FIXTURES: &str = "resources/tests/compiler-library";

fn compile(source: &str, filename: &str) -> Result<Rc<SExp>, CompileErr> {
    let arguments = HashMap::from([
        (
            "path_or_code".into(),
            ArgumentValue::ArgString(None, source.into()),
        ),
        (
            "include".into(),
            ArgumentValue::ArgArray(vec![ArgumentValue::ArgString(None, FIXTURES.into())]),
        ),
    ]);
    let mut allocator = Allocator::new();
    // Use the CLI's dialect/optimization selection, as Gaming's build does.
    let parsed = RunAndCompileInputData::new(&mut allocator, &arguments).unwrap();
    // Module outputs stay in memory; no generated files or external checkout.
    let opts = TestModuleCompilerOpts::new(parsed.opts.set_filename(filename));
    let compiled = compile_file(
        &mut allocator,
        Rc::new(DefaultProgramRunner::new()),
        Rc::new(opts),
        source,
        &mut HashMap::new(),
    )?;
    Ok(match compiled {
        CompilerOutput::Program(_, program) => Rc::new(program),
        CompilerOutput::Module(module) => module
            .components
            .iter()
            .find(|component| component.shortname == b"program")
            .expect("exported program")
            .content
            .clone(),
    })
}

fn compile_fixture(name: &str) -> Rc<SExp> {
    let filename = format!("{FIXTURES}/{name}.clsp");
    let source = fs::read_to_string(&filename).unwrap();
    compile(&source, &filename).unwrap_or_else(|error| panic!("{filename}: {error:?}"))
}

fn sexp(text: &str) -> Rc<SExp> {
    parse_sexp(Srcloc::start("library-test"), text.bytes()).unwrap()[0].clone()
}

fn evaluate(program: Rc<SExp>, input: &str) -> (u64, Vec<u8>) {
    let mut allocator = Allocator::new();
    let program = convert_to_clvm_rs(&mut allocator, program).unwrap();
    let input = convert_to_clvm_rs(&mut allocator, sexp(input)).unwrap();
    let result = DefaultProgramRunner::new()
        .run_program(
            &mut allocator,
            program,
            input,
            Some(RunProgramOption {
                max_cost: Some(1_000_000_000),
                ..Default::default()
            }),
        )
        .unwrap();
    (
        result.0,
        clvmr::serde::node_to_bytes(&allocator, result.1).unwrap(),
    )
}

fn assert_result(program: Rc<SExp>, input: &str, expected: &str) {
    let (_, result) = evaluate(program, input);
    let mut allocator = Allocator::new();
    let expected_node = convert_to_clvm_rs(&mut allocator, sexp(expected)).unwrap();
    assert_eq!(
        result,
        clvmr::serde::node_to_bytes(&allocator, expected_node).unwrap(),
        "input {input}, expected {expected}"
    );
}

#[test]
fn test_list_library_fixtures() {
    for (name, examples) in [
        (
            "test_prepend",
            vec![
                ("(() ())", "()"),
                ("((1 2) (3 4))", "(1 2 3 4)"),
                ("((1 2) 3)", "(1 2 . 3)"),
            ],
        ),
        ("test_range", vec![("(0)", "()"), ("(5)", "(0 1 2 3 4)")]),
        (
            "test_reverse",
            vec![("(())", "()"), ("((1 2 3 4))", "(4 3 2 1)")],
        ),
        (
            "test_map_enumerate",
            vec![("(())", "()"), ("((11 22 33))", "((11 0) (22 1) (33 2))")],
        ),
        (
            "smoke_test_sort",
            vec![("(())", "()"), ("((3 -1 3 0 2))", "(-1 0 2 3 3)")],
        ),
        (
            "smoke_test_deep_compare",
            vec![("()", "(1 1 1 1 1 1 1 1 1 1 1)")],
        ),
        (
            "smoke_test_permutations",
            vec![
                ("(())", "(())"),
                ("((1 2))", "((1 2) (2 1))"),
                ("((1 1))", "((1 1) (1 1))"),
            ],
        ),
    ] {
        let program = compile_fixture(name);
        for (input, expected) in examples {
            assert_result(program.clone(), input, expected);
        }
    }
}

#[test]
fn test_sort_all_permutations() {
    assert_result(compile_fixture("test_sort"), "()", "1");
}

#[test]
fn test_permutation_counts_and_uniqueness() {
    let program = compile_fixture("test_permutations");
    for input in ["(0 0)", "(3 5)"] {
        assert_result(program.clone(), input, "()");
    }
}
