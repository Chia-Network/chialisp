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

/// Exposes the cache-key segment used under `.chialisp/<key>/` (tests and tooling only).
#[cfg(test)]
pub fn module_cache_key_hex(cf: &CompileForm) -> String {
    cache_key(cf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::comptypes::{BodyForm, CompileForm, IncludeDesc, IncludeProcessType};
    use crate::compiler::sexp::SExp;
    use crate::compiler::srcloc::Srcloc;

    fn empty_compileform(loc: Srcloc) -> CompileForm {
        CompileForm {
            loc: loc.clone(),
            include_forms: Vec::new(),
            args: Rc::new(SExp::Nil(loc.clone())),
            helpers: Vec::new(),
            exp: Rc::new(BodyForm::Quoted(SExp::Nil(loc.clone()))),
        }
    }

    #[test]
    fn cache_key_stable_for_empty_includes() {
        let loc = Srcloc::start(&"a.clsp".to_string());
        let cf = empty_compileform(loc);
        let k = cache_key(&cf);
        assert_eq!(
            k,
            "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"
        );
    }

    #[test]
    fn cache_key_changes_with_concatenated_fingerprints() {
        let loc = Srcloc::start(&"b.clsp".to_string());
        let mut cf = empty_compileform(loc.clone());
        let fp = |prefix: &[u8]| {
            let mut a = [0u8; 32];
            a[..prefix.len()].copy_from_slice(prefix);
            a
        };
        let desc = |fp: [u8; 32]| IncludeDesc {
            kw: loc.clone(),
            nl: loc.clone(),
            name: b"x".to_vec(),
            kind: None,
            fingerprint: fp,
        };
        cf.include_forms.push(desc(fp(&[1, 2, 3])));
        let k1 = cache_key(&cf);
        cf.include_forms.push(desc(fp(&[4, 5])));
        let k2 = cache_key(&cf);
        assert_ne!(k1, k2);
        cf.include_forms.truncate(1);
        let k1_again = cache_key(&cf);
        assert_eq!(k1, k1_again);
    }

    #[test]
    fn cache_key_main_fingerprint_style() {
        let loc = Srcloc::start(&"c.clsp".to_string());
        let mut cf = empty_compileform(loc.clone());
        let mut main_fp = [0u8; 32];
        main_fp[0] = 0xab;
        main_fp[1] = 0xcd;
        cf.include_forms.push(IncludeDesc {
            kw: loc.clone(),
            nl: loc.clone(),
            name: b"main".to_vec(),
            kind: Some(IncludeProcessType::Compiled),
            fingerprint: main_fp,
        });
        let k = cache_key(&cf);
        assert!(!k.is_empty());
        assert_ne!(
            k,
            "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"
        );
    }
}
