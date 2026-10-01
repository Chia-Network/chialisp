use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use clvmr::allocator::Allocator;
use serde_json::Value as JsonValue;
use toml::{Table, Value};

use chialisp::classic::clvm_tools::clvmc::CompileError;
use chialisp::classic::clvm_tools::comp_input::RunAndCompileInputData;
use chialisp::classic::platform::argparse::ArgumentValue;
use chialisp::compiler::comptypes::CompileErr;
use chialisp::compiler::srcloc::Srcloc;

fn do_compile(title: &str, filename: &str) -> Result<(), CompileError> {
    let mut allocator = Allocator::new();
    let mut arguments: HashMap<String, ArgumentValue> = HashMap::new();
    arguments.insert(
        "include".to_string(),
        ArgumentValue::ArgArray(vec![
            ArgumentValue::ArgString(None, "clsp".to_string()),
            ArgumentValue::ArgString(None, ".".to_string()),
        ]),
    );

    let file_content = fs::read_to_string(filename).map_err(|e| {
        CompileErr(
            Srcloc::start(filename),
            format!("failed to read {filename}: {e:?}"),
        )
    })?;

    arguments.insert(
        "path_or_code".to_string(),
        ArgumentValue::ArgString(Some(filename.to_string()), file_content),
    );

    let parsed = RunAndCompileInputData::new(&mut allocator, &arguments).map_err(|e| {
        CompileError::Modern(
            Srcloc::start("*error*"),
            format!("error building chialisp {title}: {e}"),
        )
    })?;
    let mut symbol_table = HashMap::new();

    parsed.compile_modern(&mut allocator, &mut symbol_table)?;

    Ok(())
}

fn compile_chialisp() -> Result<(), CompileError> {
    let srcloc = Srcloc::start("chialisp.toml");
    let chialisp_toml_text = fs::read_to_string("chialisp.toml").map_err(|e| {
        CompileError::Modern(
            srcloc.clone(),
            format!("Error reading chialisp.toml: {e:?}"),
        )
    })?;

    let chialisp_toml = chialisp_toml_text
        .parse::<Table>()
        .map_err(|e| CompileError::Modern(srcloc, format!("Error parsing chialisp.toml: {e:?}")))?;

    if let Some(Value::Table(t)) = chialisp_toml.get("compile") {
        for (k, v) in t.iter() {
            if let Value::String(s) = v {
                do_compile(k, s)?;
            }
        }
    }

    Ok(())
}

fn generate_protocol_timeout_bounds(out_dir: &Path) {
    let path = Path::new("shared/protocol-constants/constants.json");
    let source = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let value: JsonValue = serde_json::from_str(&source)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
    let bound = |field: &str, side: &str| {
        value[field][side]
            .as_u64()
            .unwrap_or_else(|| panic!("{field}.{side} must be an unsigned integer"))
    };
    let generated = format!(
        "pub const GAME_TIMEOUT_BLOCKS_MIN: u64 = {};\n\
         pub const GAME_TIMEOUT_BLOCKS_MAX: u64 = {};\n\
         pub const SESSION_TIMEOUT_BLOCKS_MIN: u64 = {};\n\
         pub const SESSION_TIMEOUT_BLOCKS_MAX: u64 = {};\n",
        bound("gameTimeoutBlocks", "min"),
        bound("gameTimeoutBlocks", "max"),
        bound("sessionTimeoutBlocks", "min"),
        bound("sessionTimeoutBlocks", "max"),
    );
    fs::write(out_dir.join("protocol_timeout_bounds.rs"), generated)
        .expect("write generated protocol timeout bounds");
}

// Compile chialisp programs in this tree.
fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    generate_protocol_timeout_bounds(&out_dir);
    if std::env::var("CHIALISP_NOCOMPILE").is_err() {
        if let Err(e) = compile_chialisp() {
            panic!("error compiling chialisp: {e:?}");
        }
    }
}
