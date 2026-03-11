use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm_tools::cmds::{cldb_hierarchy, CldbHierarchyArgs, YamlElement};
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::cldb::{hex_to_modern_sexp, CldbNoOverride, CldbRun, CldbRunEnv, FAVOR_HEX};
use crate::compiler::clvm::{start_step, RunStep};
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::prims;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;

fn json_to_yamlelement(json: &serde_json::Value) -> YamlElement {
    match json {
        serde_json::Value::Object(map) => {
            let mut yaml_map = BTreeMap::new();
            for (k, v) in map.iter() {
                let yamelled_wing = json_to_yamlelement(v);
                yaml_map.insert(k.to_string(), yamelled_wing);
            }
            YamlElement::Subtree(yaml_map)
        }
        serde_json::Value::String(s) => YamlElement::String(s.clone()),
        serde_json::Value::Array(v) => {
            YamlElement::Array(v.iter().map(|v| json_to_yamlelement(v)).collect())
        }
        _ => todo!(),
    }
}

fn yaml_to_yamlelement(yaml: &BTreeMap<String, YamlElement>) -> YamlElement {
    YamlElement::Subtree(yaml.clone())
}

trait StepOfCldbViewer {
    fn show(&mut self, _step: &RunStep, _output: Option<BTreeMap<String, String>>) -> bool {
        true
    }
}

fn run_clvm_in_cldb<V>(
    program_name: &str,
    program_lines: Rc<Vec<String>>,
    program: Rc<SExp>,
    symbols: HashMap<String, String>,
    args: Rc<SExp>,
    viewer: &mut V,
    flags: u32,
) -> Option<String>
where
    V: StepOfCldbViewer,
{
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());

    let mut prim_map = HashMap::new();
    for p in prims::prims().iter() {
        prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
    }
    let step = start_step(program, args);
    let cldbenv = CldbRunEnv::new(
        Some(program_name.to_string()),
        program_lines,
        Box::new(CldbNoOverride::new_symbols(symbols)),
    );
    let mut cldbrun = CldbRun::new(runner, Rc::new(prim_map), Box::new(cldbenv), step);
    cldbrun.set_flags(flags);

    let mut output: BTreeMap<String, String> = BTreeMap::new();

    loop {
        if cldbrun.is_ended() {
            return output.get("Final").cloned();
        }

        if let Some(result) = cldbrun.step(&mut allocator) {
            output = result;
            if !viewer.show(&cldbrun.current_step(), Some(output.clone())) {
                return None;
            }
        }
    }
}

struct DoesntWatchCldb {}

impl StepOfCldbViewer for DoesntWatchCldb {}

#[test]
fn test_run_clvm_in_cldb() {
    let program_name = "fact.clsp";
    let program_code = "(mod (X) (include *standard-cl-21*) (defun fact (X) (if (> X 1) (* X (fact (- X 1))) 1)) (fact X))";
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts = Rc::new(DefaultCompilerOpts::new(program_name));
    let mut symbols = HashMap::new();
    let args = parse_sexp(Srcloc::start("*args*"), "(5)".bytes()).expect("should parse")[0].clone();

    let program = compile_file(
        &mut allocator,
        runner.clone(),
        opts,
        &program_code,
        &mut symbols,
    )
    .expect("should compile");

    let program_lines: Vec<String> = program_code.lines().map(|x| x.to_string()).collect();

    assert_eq!(
        run_clvm_in_cldb(
            program_name,
            Rc::new(program_lines),
            Rc::new(program),
            symbols,
            args,
            &mut DoesntWatchCldb {},
            0,
        ),
        Some("120".to_string())
    );
}

#[test]
fn test_cldb_hex_to_modern_sexp_smoke_0() {
    let mut allocator = Allocator::new();
    let symbol_table = HashMap::new();
    let input_program = "ff01ff03ff0580";
    let result_succeed = hex_to_modern_sexp(
        &mut allocator,
        &symbol_table,
        Srcloc::start("*test*"),
        input_program,
    )
    .unwrap();
    assert_eq!(result_succeed.to_string(), "(1 3 5)");
}

#[test]
fn test_cldb_hex_to_modern_sexp_fail_half_cons() {
    let mut allocator = Allocator::new();
    let symbol_table = HashMap::new();
    let input_program = "ff01ff03ff05";
    let result = hex_to_modern_sexp(
        &mut allocator,
        &symbol_table,
        Srcloc::start("*test*"),
        input_program,
    );
    assert!(result.is_err());
}

#[test]
fn test_cldb_hex_to_modern_sexp_fail_odd_hex() {
    let mut allocator = Allocator::new();
    let symbol_table = HashMap::new();
    let input_program = "ff01ff03ff058";
    let result = hex_to_modern_sexp(
        &mut allocator,
        &symbol_table,
        Srcloc::start("*test*"),
        input_program,
    );
    assert!(result.is_err());
}

fn compile_and_run_program_with_tree(
    input_file: &str,
    input_program_text: &str,
    args_text: &str,
    search_paths: &[String],
    flags: u32,
) -> Vec<BTreeMap<String, YamlElement>> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts = Rc::new(DefaultCompilerOpts::new(&input_file))
        .set_optimize(false)
        .set_search_paths(search_paths);

    let mut use_symbol_table = HashMap::new();

    let program = compile_file(
        &mut allocator,
        runner.clone(),
        opts.clone(),
        &input_program_text,
        &mut use_symbol_table,
    )
    .expect("should compile");
    let args = parse_sexp(program.loc(), args_text.as_bytes().iter().copied())
        .expect("should parse args")[0]
        .clone();

    let mut prim_map = HashMap::new();
    for p in prims::prims().iter() {
        prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
    }
    let program_lines: Rc<Vec<String>> =
        Rc::new(input_program_text.lines().map(|x| x.to_string()).collect());

    cldb_hierarchy(CldbHierarchyArgs {
        runner,
        prim_map: Rc::new(prim_map),
        input_file_name: Some(input_file.to_owned()),
        lines: program_lines,
        symbol_table: Rc::new(use_symbol_table),
        prog: Rc::new(program),
        args,
        flags,
    })
}

fn run_program_as_tree_from_hex(
    file_name: &str,
    input_program: &str,
    input_args: &str,
    symbol_table: HashMap<String, String>,
) -> Vec<BTreeMap<String, YamlElement>> {
    let mut allocator = Allocator::new();
    let prog_srcloc = Srcloc::start("*program*");
    let args_srcloc = Srcloc::start("*args*");

    let program = hex_to_modern_sexp(
        &mut allocator,
        &symbol_table,
        prog_srcloc.clone(),
        &input_program,
    )
    .expect("should decode from hex");

    let args = hex_to_modern_sexp(
        &mut allocator,
        &symbol_table,
        args_srcloc.clone(),
        &input_args,
    )
    .expect("should decode from hex");

    let mut prim_map = HashMap::new();
    for p in prims::prims().iter() {
        prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
    }
    let program_lines = Rc::new(vec![]);
    let runner = Rc::new(DefaultProgramRunner::new());
    cldb_hierarchy(CldbHierarchyArgs {
        runner,
        prim_map: Rc::new(prim_map),
        input_file_name: Some(file_name.to_owned()),
        lines: program_lines,
        symbol_table: Rc::new(symbol_table),
        prog: program,
        args,
        flags: 0,
    })
}

fn compare_run_output(
    result: Vec<BTreeMap<String, YamlElement>>,
    run_entries: Vec<serde_json::Value>,
) {
    assert_eq!(result.len(), run_entries.len());
    for i in 0..result.len() {
        let result_entry = result[i].clone();
        let want_entry = &run_entries[i];
        let want_yaml = json_to_yamlelement(want_entry);
        let have_yaml = yaml_to_yamlelement(&result_entry);
        assert_eq!(want_yaml, have_yaml);
    }
}

#[test]
fn test_cldb_hierarchy_mode() {
    let json_text = fs::read_to_string("resources/tests/cldb_tree/test.json")
        .expect("test resources should exist: test.json");
    let run_entries: Vec<serde_json::Value> =
        serde_json::from_str(&json_text).expect("should contain json");
    let input_program = fs::read_to_string("resources/tests/cldb_tree/test_rec_1.cl")
        .expect("test resources should exist: test_rec_1.cl");
    let input_file = "./test_rec_1.cl";

    let result =
        compile_and_run_program_with_tree(&input_file, &input_program, "(3 2)", &vec![], 0);

    compare_run_output(result, run_entries);
}

#[test]
fn test_execute_program_and_capture_arguments() {
    let compiled_symbols_text =
        fs::read_to_string("resources/tests/cldb_tree/pool_member_innerpuz_extra.sym")
            .expect("should exist");
    let compiled_symbols: HashMap<String, String> =
        serde_json::from_str(&compiled_symbols_text).expect("should decode");
    let result = run_program_as_tree_from_hex(
        "./pool_member_innerpuz.hex",
        &fs::read_to_string("resources/tests/cldb_tree/pool_member_innerpuz.hex")
            .expect("should exist"),
        &fs::read_to_string("resources/tests/cldb_tree/pool_member_innerpuz_args.hex")
            .expect("should exist"),
        compiled_symbols,
    );

    let json_text = fs::read_to_string("resources/tests/cldb_tree/pool_member_innerpuz_run.json")
        .expect("test resources should exist");
    let run_entries: Vec<serde_json::Value> =
        serde_json::from_str(&json_text).expect("should contain json");

    compare_run_output(result, run_entries);
}

struct ExpectFailure {
    throw: bool,
    found_desired: bool,
    want_result: Option<String>,
}

impl ExpectFailure {
    fn new(throw: bool, want_result: Option<String>) -> Self {
        ExpectFailure {
            throw,
            want_result,
            found_desired: false,
        }
    }

    fn correct_result(&self) -> bool {
        self.found_desired
    }
}

impl StepOfCldbViewer for ExpectFailure {
    fn show(&mut self, _step: &RunStep, output: Option<BTreeMap<String, String>>) -> bool {
        if let Some(o) = output {
            if let Some(_) = o.get("Failure") {
                let did_throw = o.get("Operator") == Some(&"8".to_string());
                if let Some(desired_outcome) = &self.want_result {
                    self.found_desired =
                        did_throw == self.throw && o.get("Arguments") == Some(desired_outcome);
                } else {
                    self.found_desired = did_throw == self.throw;
                }
                return false;
            }
        }

        return true;
    }
}

#[test]
fn test_cldb_explicit_throw() {
    let program_name = "*test*";
    let loc = Srcloc::start(program_name);
    let program =
        parse_sexp(loc.clone(), b"(x (q . 2))".iter().copied()).expect("should parse")[0].clone();
    let args = Rc::new(SExp::Nil(loc));
    let program_lines = Rc::new(Vec::new());

    let mut watcher = ExpectFailure::new(true, Some("(2)".to_string()));

    assert_eq!(
        run_clvm_in_cldb(
            program_name,
            program_lines,
            program,
            HashMap::new(),
            args,
            &mut watcher,
            0
        ),
        None
    );

    assert!(watcher.correct_result());
}

#[test]
fn test_clvm_operator_with_weird_tail() {
    let filename = "test-weird-tail.clvm";
    let loc = Srcloc::start(filename);
    let program = "(+ (q . 3) (q . 5) . \"\")";
    let parsed = parse_sexp(loc.clone(), program.as_bytes().iter().copied()).expect("should parse");
    let args = Rc::new(SExp::Nil(loc));
    let program_lines = Rc::new(vec![program.to_string()]);

    assert_eq!(
        run_clvm_in_cldb(
            filename,
            program_lines,
            parsed[0].clone(),
            HashMap::new(),
            args,
            &mut DoesntWatchCldb {},
            0,
        ),
        Some("8".to_string())
    );
}

#[test]
fn test_cldb_with_favor_hex() {
    let filename = "favor_hex.clvm";
    let loc = Srcloc::start(filename);
    let program = "(concat (1 . 1) (1 . 1122334455))";
    let parsed = parse_sexp(loc.clone(), program.as_bytes().iter().copied()).expect("should parse");
    let args = Rc::new(SExp::Nil(loc));
    let program_lines = Rc::new(vec![program.to_string()]);

    assert_eq!(
        run_clvm_in_cldb(
            filename,
            program_lines,
            parsed[0].clone(),
            HashMap::new(),
            args,
            &mut DoesntWatchCldb {},
            FAVOR_HEX,
        ),
        Some("0x0142e576f7".to_string())
    );
}

#[test]
fn test_cldb_hierarchy_hex() {
    let json_text = fs::read_to_string("resources/tests/cldb_tree/hex.json")
        .expect("test resources should exist: test.json");
    let run_entries: Vec<serde_json::Value> =
        serde_json::from_str(&json_text).expect("should contain json");
    let input_program = "(mod () (concat 1 1122334455))".to_string();

    let input_file = "test_with_hex.clsp";

    let result =
        compile_and_run_program_with_tree(&input_file, &input_program, "()", &vec![], FAVOR_HEX);

    compare_run_output(result, run_entries);
}

#[test]
fn test_cldb_hierarchy_before_hex() {
    let json_text = fs::read_to_string("resources/tests/cldb_tree/pre_hex.json")
        .expect("test resources should exist: test.json");
    let run_entries: Vec<serde_json::Value> =
        serde_json::from_str(&json_text).expect("should contain json");
    let input_program = "(mod () (concat 1 1122334455))".to_string();

    let input_file = "test_with_hex.clsp";

    let result = compile_and_run_program_with_tree(&input_file, &input_program, "()", &vec![], 0);

    compare_run_output(result, run_entries);
}

#[test]
fn test_cldb_operators_outside_guard() {
    let filename = "coinid.clvm";
    let loc = Srcloc::start(filename);
    let inputs_outputs = [
        (
            "(coinid (sha256 (q . 3)) (sha256 (q . 3)) (q . 4))",
            "()",
            "0x9f7f12b86a583805a4442879b7b5b531469e45c7e753e5fd431058e90bf3fbec"
        ),
        (
            "(modpow (q . 2) (q . 8) (q . 10))",
            "()",
            "6"
        ),
        (
            "(% (q . 13) (q . 5))",
            "()",
            "3"
        ),
        (
            // resources/tests/bls/modern-bls-verify-signature.clsp
            "(2 (1 59 (1 . 0xb00ab9a8af54804b43067531d96c176710c05980fccf8eee1ae12a4fd543df929cce860273af931fe4fdbc407d495f73114ab7d17ef08922e56625daada0497582340ecde841a9e997f2f557653c21c070119662dd2efa47e2d6c5e2de00eefa) (1 . 0x86243290bbcbfd9ae75bdece7981965350208eb5e99b04d5cd24e955ada961f8c0a162dee740be7bdc6c3c0613ba2eb1) 5) (4 (1) 1))",
            "(0x0102030405)",
            "()"
        )
    ];

    for (program, arg_str, expected) in inputs_outputs {
        let parsed = parse_sexp(loc.clone(), program.bytes()).expect("should parse");
        let args_parsed = parse_sexp(loc.clone(), arg_str.bytes()).expect("should parse");
        let program_lines = Rc::new(vec![program.to_string()]);

        assert_eq!(
            run_clvm_in_cldb(
                filename,
                program_lines,
                parsed[0].clone(),
                HashMap::new(),
                args_parsed[0].clone(),
                &mut DoesntWatchCldb {},
                FAVOR_HEX,
            ),
            Some(expected.to_string())
        );
    }
}
