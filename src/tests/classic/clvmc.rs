use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm_tools::clvmc::{compile_clvm, compile_clvm_text};
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::tests::classic::run::read_json_from_file;

#[test]
fn test_compile_clvm_function_1() {
    let mut symbol_hash_table = HashMap::new();
    let bridge_hex_file = "validation_taproot.clvm2.hex";
    // Try to remove it if it exists.
    fs::remove_file(bridge_hex_file).ok();
    let _ = compile_clvm(
        "resources/tests/bridgeref/validation_taproot.clsp",
        bridge_hex_file,
        &["resources/tests/bridge-includes".to_string()],
        &mut symbol_hash_table,
    )
    .expect("should compile");
    let bridge_hex = fs::read_to_string(bridge_hex_file).expect("should have been created");
    fs::remove_file(bridge_hex_file).expect("should have existed");
    let read_in_hex_data =
        fs::read_to_string("resources/tests/bridgeref/validation_taproot.clvm.hex.reference")
            .expect("should exist");
    let read_in_symbol_data =
        read_json_from_file("resources/tests/bridgeref/validation_taproot.sym.reference");
    assert_eq!(bridge_hex, read_in_hex_data);
    for (key, value) in read_in_symbol_data.iter() {
        if key != "source_file" {
            assert_eq!(symbol_hash_table.get(key), Some(value));
        }
    }
}

#[test]
fn test_compile_clvm_with_previous_data() {
    let bridge_hex_file = "validation_taproot.clvm.hex";
    let bridge_hex =
        fs::read_to_string("resources/tests/bridgeref/validation_taproot.clvm.hex.reference")
            .expect("should have been created");
    let mut symbol_hash_table = HashMap::new();

    fs::write(bridge_hex_file, bridge_hex).expect("should write");
    compile_clvm(
        "resources/tests/bridgeref/validation_taproot.clsp",
        bridge_hex_file,
        &["resources/tests/bridge-includes".to_string()],
        &mut symbol_hash_table,
    )
    .expect("should compile");
    fs::remove_file(bridge_hex_file).expect("should have existed");
}

#[test]
fn test_classic_compile_error_output() {
    let mut allocator = Allocator::new();
    let mut symbols = HashMap::new();
    let path = "*test*";
    let content = "(mod (X) (xxx X))";
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(path));
    let res = compile_clvm_text(
        &mut allocator,
        opts.clone(),
        &mut symbols,
        &content,
        &path,
        true,
    )
    .map_err(|e| e.format(&allocator, opts));
    assert_eq!(
        Err(
            "error can't compile (\"xxx\" 88), unknown operator compiling (\"xxx\" 88)".to_string()
        ),
        res
    );
}

#[test]
fn test_modern_compile_error_output() {
    let mut allocator = Allocator::new();
    let mut symbols = HashMap::new();
    let path = "*test*";
    let content = indoc! {
    "(mod (X)
           (include *standard-cl-23*)
           (+ X1 X)
           )
        "};
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(path));
    let res = compile_clvm_text(
        &mut allocator,
        opts.clone(),
        &mut symbols,
        &content,
        &path,
        true,
    )
    .map_err(|e| e.format(&allocator, opts));
    assert_eq!(
        Err("*test*(3):4-*test*(3):6: Unbound use of X1 as a variable name".to_string()),
        res
    );
}
