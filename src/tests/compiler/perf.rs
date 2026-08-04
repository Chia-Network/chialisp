use std::collections::HashMap;
use std::fs;
use std::rc::Rc;
use std::time::{Duration, Instant};

use clvm_rs::allocator::{Allocator, NodePtr};

use crate::classic::clvm::sexp::sexp_as_bin;
use crate::classic::clvm_tools::clvmc::compile_clvm_text;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{CompilerOpts, CompilerOutput};
use crate::compiler::sexp::decode_string;
use crate::tests::compiler::modules::{perform_compile_of_file, TestModuleCompilerOpts};

#[derive(Clone, Copy, Debug)]
enum CompileFlow {
    Program,
    Module,
}

#[derive(Debug)]
struct PerfCase {
    name: &'static str,
    source_path: &'static str,
    include_paths: &'static [&'static str],
    compile_flow: CompileFlow,
}

#[derive(Debug)]
struct OutputMeasurement {
    name: String,
    byte_size: usize,
    hex_size: usize,
}

#[derive(Debug)]
struct CaseMeasurement {
    case: &'static PerfCase,
    compile_duration: Duration,
    outputs: Vec<OutputMeasurement>,
}

const PERF_CASES: &[PerfCase] = &[
    PerfCase {
        name: "validation_taproot",
        source_path: "resources/tests/bridgeref/validation_taproot.clsp",
        include_paths: &["resources/tests/bridge-includes"],
        compile_flow: CompileFlow::Program,
    },
    PerfCase {
        name: "gameref21_referee",
        source_path: "resources/tests/gameref21/referee.clsp",
        include_paths: &["resources/tests/gameref21"],
        compile_flow: CompileFlow::Program,
    },
    PerfCase {
        name: "cl23_handcalc_includes",
        source_path: "resources/tests/game-referee-in-cl23/test_handcalc.clsp",
        include_paths: &["resources/tests/game-referee-in-cl23"],
        compile_flow: CompileFlow::Program,
    },
    PerfCase {
        name: "module_handcalc",
        source_path: "resources/tests/module/test_handcalc.clsp",
        include_paths: &["resources/tests/module"],
        compile_flow: CompileFlow::Module,
    },
    PerfCase {
        name: "module_deinline_blowup",
        source_path: "resources/tests/module/deinline_module_blowup.clsp",
        include_paths: &["resources/tests/module"],
        compile_flow: CompileFlow::Module,
    },
    PerfCase {
        name: "did_innerpuz",
        source_path: "resources/tests/did_innerpuz.clsp",
        include_paths: &["resources/tests"],
        compile_flow: CompileFlow::Program,
    },
    PerfCase {
        name: "cse_tricky_assign",
        source_path: "resources/tests/strict/cse_tricky_assign.clsp",
        include_paths: &[
            "resources/tests/game-referee-after-cl21",
            "resources/tests/strict",
        ],
        compile_flow: CompileFlow::Program,
    },
    PerfCase {
        name: "pool_member_innerpuz",
        source_path: "resources/tests/cldb_tree/pool_member_innerpuz.cl",
        include_paths: &["resources/tests", "resources/tests/usecheck-work"],
        compile_flow: CompileFlow::Program,
    },
];

#[test]
fn compile_performance_suite() {
    for case in PERF_CASES {
        let measurement = measure_case(case);
        report_measurement(&measurement);

        // Keep timing and size as reported measurements. Some large compile
        // paths can emit equivalent programs with small byte-size variation.
        assert!(
            !measurement.outputs.is_empty(),
            "{} should produce at least one compiled CLVM output",
            case.name
        );

        for output in measurement.outputs.iter() {
            assert!(
                output.byte_size > 0,
                "{}:{} should have non-empty bytecode",
                case.name,
                output.name
            );
        }
    }
}

fn measure_case(case: &'static PerfCase) -> CaseMeasurement {
    match case.compile_flow {
        CompileFlow::Program => measure_program_case(case),
        CompileFlow::Module => measure_module_case(case),
    }
}

fn measure_program_case(case: &'static PerfCase) -> CaseMeasurement {
    let source = fs::read_to_string(case.source_path).expect("perf fixture should exist");
    let mut allocator = Allocator::new();
    let mut symbol_table = HashMap::new();
    let opts = compiler_opts(case);

    let start = Instant::now();
    let node = compile_clvm_text(
        &mut allocator,
        opts,
        &mut symbol_table,
        &source,
        case.source_path,
        false,
    )
    .expect("perf fixture should compile");
    let compile_duration = start.elapsed();

    CaseMeasurement {
        case,
        compile_duration,
        outputs: vec![measure_node(&mut allocator, "program".to_string(), node)],
    }
}

fn measure_module_case(case: &'static PerfCase) -> CaseMeasurement {
    let source = fs::read_to_string(case.source_path).expect("perf fixture should exist");
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let source_opts = TestModuleCompilerOpts::new(compiler_opts(case));

    let start = Instant::now();
    let compile_result = perform_compile_of_file(
        &mut allocator,
        runner,
        source_opts,
        case.source_path,
        &source,
    )
    .expect("perf module fixture should compile");
    let compile_duration = start.elapsed();

    let outputs = match compile_result.compiled {
        CompilerOutput::Program(_, sexp) => {
            let node = convert_to_clvm_rs(&mut allocator, Rc::new(sexp))
                .expect("compiled program should convert to CLVM");
            vec![measure_node(&mut allocator, "program".to_string(), node)]
        }
        CompilerOutput::Module(module) => module
            .components
            .iter()
            .map(|component| {
                let node = convert_to_clvm_rs(&mut allocator, component.content.clone())
                    .expect("compiled module component should convert to CLVM");
                measure_node(&mut allocator, decode_string(&component.shortname), node)
            })
            .collect(),
    };

    CaseMeasurement {
        case,
        compile_duration,
        outputs,
    }
}

fn compiler_opts(case: &PerfCase) -> Rc<dyn CompilerOpts> {
    let include_paths = case
        .include_paths
        .iter()
        .map(|path| path.to_string())
        .collect::<Vec<_>>();
    Rc::new(DefaultCompilerOpts::new(case.source_path)).set_search_paths(&include_paths)
}

fn measure_node(allocator: &mut Allocator, name: String, node: NodePtr) -> OutputMeasurement {
    let bytecode = sexp_as_bin(allocator, node);
    let byte_size = bytecode.data().len();
    OutputMeasurement {
        name,
        byte_size,
        hex_size: byte_size * 2,
    }
}

fn report_measurement(measurement: &CaseMeasurement) {
    let outputs = measurement
        .outputs
        .iter()
        .map(|output| {
            format!(
                "{}:{}B:{}hex",
                output.name, output.byte_size, output.hex_size
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    eprintln!(
        "compile_perf case={} flow={:?} source={} duration_us={} total_byte_size={} outputs={}",
        measurement.case.name,
        measurement.case.compile_flow,
        measurement.case.source_path,
        measurement.compile_duration.as_micros(),
        total_byte_size(measurement),
        outputs
    );
}

fn total_byte_size(measurement: &CaseMeasurement) -> usize {
    measurement
        .outputs
        .iter()
        .map(|output| output.byte_size)
        .sum()
}
