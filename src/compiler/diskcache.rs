use std::rc::Rc;

use crate::compiler::clvm::sha256tree_from_atom;
use crate::compiler::comptypes::{CompileErr, CompileForm, CompilerOpts};
use crate::compiler::sexp::decode_string;

fn cache_key(cf: &CompileForm) -> String {
    let mut include_fingerprints = Vec::new();
    for include in cf.include_forms.iter() {
        include_fingerprints.extend_from_slice(&include.fingerprint);
    }
    hex::encode(sha256tree_from_atom(&include_fingerprints))
}

/// Try to get an element from the cache, exposing errors.
///
/// Module style outputs are separately built with CompileForm input programs.  They produce
/// outputs according to their Export list.  Since exports interact when they're in the common
/// set, the output of each export is fully determined by the dialect, compileform, the list of
/// exports and itself, which means we can use the hash of these inputs as the majority of the
/// cache key.
///
/// Ultimately, the exports are the output artifacts.  A CompilerOutput with all exports settled
/// needn't be processed further.
///
/// So we're given a CompilerOutput and we elide code generation and optimization for its exports
/// when all the exports associated with this particular configuration are available.
pub fn try_element_from_cache(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    export_path: &str,
) -> Option<String> {
    let key = cache_key(cf);
    let hex_file_name = format!(".chialisp/{key}/{export_path}");
    opts.read_new_file(cf.loc().file.to_string(), hex_file_name.clone())
        .ok()
        .map(|data| decode_string(&data.1))
}

pub fn set_cache_element_error(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    export_path: &str,
    export_hex: &str,
) -> Result<(), CompileErr> {
    let key = cache_key(cf);
    let hex_file_name = format!(".chialisp/{key}/{export_path}");
    opts.write_new_file(&hex_file_name, export_hex.as_bytes())?;
    Ok(())
}

/// Set an element in the cache.  Use the current dialect and compileform as the majority
/// of key material.  We add a file path and content to determine an exact hex serialization of
/// an export.
pub fn set_cache_element(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    export_path: &str,
    export_hex: &str,
) {
    set_cache_element_error(opts, cf, export_path, export_hex).ok();
}
