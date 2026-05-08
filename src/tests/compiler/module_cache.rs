//! Tests for module export disk cache (`try_from_cache`) and cache key wiring.

use std::collections::HashMap;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm::__type_compatibility__::Stream;
use crate::classic::clvm::sexp::sexp_as_bin;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::compiler::{compile_file, try_from_cache, DefaultCompilerOpts};
use crate::compiler::comptypes::{
    BodyForm, CompileForm, CompilerOpts, CompilerOutput, Export, ExportFunctionDesc,
    ExportProgramDesc, NameAndLoc,
};
use crate::compiler::diskcache::module_cache_key_hex;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::tests::compiler::modules::TestModuleCompilerOpts;

const TEST_CLSP: &str = "module_cache_try.clsp";

fn sample_program_hex() -> String {
    let mut allocator = Allocator::new();
    let loc = Srcloc::start(&"*hex*".to_string());
    let sexp = parse_sexp(loc.clone(), "(q . 99)".bytes()).expect("parse")[0].clone();
    let converted = convert_to_clvm_rs(&mut allocator, sexp).expect("convert");
    let mut stream = Stream::new(None);
    stream.write(sexp_as_bin(&mut allocator, converted));
    stream.get_value().hex()
}

fn minimal_module_compileform() -> CompileForm {
    let loc = Srcloc::start(&TEST_CLSP.to_string());
    CompileForm {
        loc: loc.clone(),
        include_forms: Vec::new(),
        args: Rc::new(SExp::Nil(loc.clone())),
        helpers: Vec::new(),
        exp: Rc::new(BodyForm::Quoted(SExp::Nil(loc.clone()))),
    }
}

fn main_export() -> Export {
    let loc = Srcloc::start(&TEST_CLSP.to_string());
    Export::MainProgram(ExportProgramDesc {
        loc: loc.clone(),
        kw_loc: None,
        args: Rc::new(SExp::Nil(loc.clone())),
        expr: Rc::new(BodyForm::Quoted(SExp::Nil(loc))),
    })
}

fn function_export(name: &[u8], as_name: Option<Vec<u8>>) -> Export {
    let loc = Srcloc::start(&TEST_CLSP.to_string());
    Export::Function(Box::new(ExportFunctionDesc {
        loc: loc.clone(),
        kw_loc: None,
        name: NameAndLoc {
            value: name.to_vec(),
            loc: None,
        },
        as_loc: None,
        as_name: as_name.map(|value| NameAndLoc { value, loc: None }),
    }))
}

fn cache_dir_for(cf: &CompileForm) -> String {
    format!(".chialisp/{}/", module_cache_key_hex(cf))
}

#[test]
fn try_from_cache_returns_none_when_cache_missing() {
    let cf = minimal_module_compileform();
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(TEST_CLSP));
    let wrapped = TestModuleCompilerOpts::new(orig_opts);
    let opts: Rc<dyn CompilerOpts> = Rc::new(wrapped);
    let out = try_from_cache(opts, &cf, &[main_export()]).expect("try_from_cache");
    assert!(out.is_none());
}

#[test]
fn try_from_cache_returns_none_when_second_export_missing() {
    let cf = minimal_module_compileform();
    let hex = sample_program_hex();
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(TEST_CLSP));
    let wrapped = TestModuleCompilerOpts::new(orig_opts);
    let prefix = cache_dir_for(&cf);
    wrapped.set_file_content(
        format!("{}{}", prefix, TEST_CLSP.replace(".clsp", ".hex")),
        hex.as_bytes().to_vec(),
    );
    let opts: Rc<dyn CompilerOpts> = Rc::new(wrapped);
    let exports = vec![main_export(), function_export(b"F", None)];
    let out = try_from_cache(opts, &cf, &exports).expect("try_from_cache");
    assert!(out.is_none());
}

#[test]
fn try_from_cache_returns_none_on_invalid_cached_hex() {
    let cf = minimal_module_compileform();
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(TEST_CLSP));
    let wrapped = TestModuleCompilerOpts::new(orig_opts);
    let prefix = cache_dir_for(&cf);
    wrapped.set_file_content(
        format!("{}{}", prefix, TEST_CLSP.replace(".clsp", ".hex")),
        b"not_valid_hex_clvm_zzzz".to_vec(),
    );
    let opts: Rc<dyn CompilerOpts> = Rc::new(wrapped);
    let out = try_from_cache(opts, &cf, &[main_export()]).expect("try_from_cache");
    assert!(out.is_none());
}

#[test]
fn try_from_cache_hits_when_main_program_hex_present() {
    let cf = minimal_module_compileform();
    let hex = sample_program_hex();
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(TEST_CLSP));
    let wrapped = TestModuleCompilerOpts::new(orig_opts);
    let prefix = cache_dir_for(&cf);
    let main_hex_path = format!("{}{}", prefix, TEST_CLSP.replace(".clsp", ".hex"));
    wrapped.set_file_content(main_hex_path.clone(), hex.as_bytes().to_vec());
    let opts: Rc<dyn CompilerOpts> = Rc::new(wrapped.clone());

    let out = try_from_cache(opts, &cf, &[main_export()]).expect("try_from_cache");
    let Some(CompilerOutput::Module(mo)) = out else {
        panic!("expected module output");
    };
    assert_eq!(mo.components.len(), 1);
    assert_eq!(mo.components[0].shortname, b"program".to_vec());
    assert_eq!(
        mo.components[0].filename,
        TEST_CLSP.replace(".clsp", ".hex")
    );

    let hash_path = format!("{}_hash.hex", TEST_CLSP.trim_end_matches(".clsp"));
    assert!(
        wrapped.get_written_file(&hash_path).is_some(),
        "expected treehash sidecar {hash_path}, got {:?}",
        wrapped.list_written_files()
    );
}

#[test]
fn try_from_cache_uses_as_name_for_hex_path() {
    let cf = minimal_module_compileform();
    let hex = sample_program_hex();
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(TEST_CLSP));
    let wrapped = TestModuleCompilerOpts::new(orig_opts);
    let prefix = cache_dir_for(&cf);
    let export = function_export(b"F", Some(b"RenamedF".to_vec()));
    // Matches `create_hex_output_path`: `<stem>_<func>.hex` with func dotted with `hex`.
    let rel_hex = format!("{}_{}.hex", TEST_CLSP.trim_end_matches(".clsp"), "RenamedF");
    let cache_path = format!("{prefix}{rel_hex}");
    wrapped.set_file_content(cache_path, hex.as_bytes().to_vec());
    let opts: Rc<dyn CompilerOpts> = Rc::new(wrapped);

    let out = try_from_cache(opts, &cf, &[export]).expect("try_from_cache");
    let Some(CompilerOutput::Module(mo)) = out else {
        panic!("expected module output");
    };
    assert_eq!(mo.components.len(), 1);
    assert_eq!(mo.components[0].shortname, b"F".to_vec());
}

fn compile_module_source(
    source_opts: &TestModuleCompilerOpts,
    content: &str,
) -> Result<CompilerOutput, crate::compiler::comptypes::CompileErr> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts: Rc<dyn CompilerOpts> = Rc::new(source_opts.clone());
    let mut symbols = HashMap::new();
    compile_file(&mut allocator, runner, opts, content, &mut symbols)
}

/// Compile a real module through compile_file (which calls compile_pre_forms +
/// add_main_fingerprint), then compile again with the cache populated. The second
/// compile should produce a cache hit via try_from_cache and yield the same hex.
#[test]
fn disk_cache_hit_after_full_compile() {
    let filename = "resources/tests/module/programs/three-outputs-common.clsp";
    let content = std::fs::read_to_string(filename).expect("read");
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(filename))
        .set_search_paths(&["resources/tests/module".to_string()]);
    let wrapped = TestModuleCompilerOpts::new(orig_opts);

    let out1 = compile_module_source(&wrapped, &content).expect("first compile");
    let CompilerOutput::Module(m1) = &out1 else {
        panic!("expected module output");
    };
    assert!(!m1.components.is_empty());
    let _first_hexes: Vec<String> = m1
        .components
        .iter()
        .map(|c| c.content.to_string())
        .collect();

    // The written_files map now contains `.chialisp/<key>/<export>.hex` entries
    // from set_cache_element, plus the output hex files. A second compile should
    // see them via try_from_cache.
    let out2 = compile_module_source(&wrapped, &content).expect("second compile");
    let CompilerOutput::Module(m2) = &out2 else {
        panic!("expected module output on re-compile");
    };
    // The fresh compile produces both primary exports and _hash sidecars as
    // components; the cache-hit path only returns the primary exports (hashes
    // are written as sidecar files). Filter to primary exports for comparison.
    let primary =
        |cs: &[crate::compiler::comptypes::CompileModuleComponent]| -> Vec<(Vec<u8>, String)> {
            cs.iter()
                .filter(|c| !c.shortname.ends_with(b"_hash"))
                .map(|c| (c.shortname.clone(), c.content.to_string()))
                .collect::<Vec<_>>()
        };
    let p1 = primary(&m1.components);
    let p2 = primary(&m2.components);
    assert_eq!(p1.len(), p2.len());
    for ((name1, hex1), (name2, hex2)) in p1.iter().zip(p2.iter()) {
        assert_eq!(name1, name2);
        assert_eq!(
            hex1,
            hex2,
            "re-compiled hex must match for {}",
            String::from_utf8_lossy(name1)
        );
    }
}

/// Editing the source should change the main fingerprint and cause a cache miss.
#[test]
fn disk_cache_miss_after_source_edit() {
    let filename = "resources/tests/module/programs/three-outputs-common.clsp";
    let content = std::fs::read_to_string(filename).expect("read");
    let orig_opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(filename))
        .set_search_paths(&["resources/tests/module".to_string()]);
    let wrapped = TestModuleCompilerOpts::new(orig_opts);

    let out1 = compile_module_source(&wrapped, &content).expect("first compile");
    let CompilerOutput::Module(m1) = &out1 else {
        panic!("expected module output");
    };

    // Change the source slightly: multiply by 5 instead of 3 in E.
    let mutated = content.replace("(* X 3)", "(* X 5)");
    assert_ne!(content, mutated);

    let out2 = compile_module_source(&wrapped, &mutated).expect("recompile with edits");
    let CompilerOutput::Module(m2) = &out2 else {
        panic!("expected module output");
    };

    // The hex should differ because the cache key changed (main fingerprint changed).
    let hexes1: Vec<String> = m1
        .components
        .iter()
        .map(|c| c.content.to_string())
        .collect();
    let hexes2: Vec<String> = m2
        .components
        .iter()
        .map(|c| c.content.to_string())
        .collect();
    assert_ne!(
        hexes1, hexes2,
        "edited source must produce different hex (cache should miss)"
    );
}
