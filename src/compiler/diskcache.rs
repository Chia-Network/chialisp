use std::rc::Rc;

use crate::compiler::clvm::sha256tree_from_atom;
use crate::compiler::comptypes::{CompileErr, CompileForm, CompilerOpts};
use crate::compiler::sexp::decode_string;

fn cache_key(opts: Rc<dyn CompilerOpts>, cf: &CompileForm) -> String {
    let dialect = opts.dialect();
    let mut key_material = b"module-cache-v2".to_vec();

    if let Some(stepping) = dialect.stepping {
        key_material.push(1);
        key_material.extend_from_slice(&stepping.to_le_bytes());
    } else {
        key_material.push(0);
    }
    key_material.push(u8::from(dialect.strict));
    key_material.push(u8::from(dialect.int_fix));
    key_material.push(u8::from(dialect.extra_numeric_constants));
    key_material.push(u8::from(dialect.cse_dominance));

    for include in cf.include_forms.iter() {
        key_material.extend_from_slice(&include.fingerprint);
    }
    hex::encode(sha256tree_from_atom(&key_material))
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
    let key = cache_key(opts.clone(), cf);
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
    let key = cache_key(opts.clone(), cf);
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
pub fn module_cache_key_hex(opts: Rc<dyn CompilerOpts>, cf: &CompileForm) -> String {
    cache_key(opts, cf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compiler::DefaultCompilerOpts;
    use crate::compiler::comptypes::{BodyForm, CompileForm, IncludeDesc, IncludeProcessType};
    use crate::compiler::dialect::AcceptedDialect;
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

    fn default_opts(filename: &str) -> Rc<dyn CompilerOpts> {
        Rc::new(DefaultCompilerOpts::new(filename))
    }

    #[test]
    fn cache_key_stable_for_empty_includes() {
        let loc = Srcloc::start(&"a.clsp".to_string());
        let cf = empty_compileform(loc);
        let k = cache_key(default_opts("a.clsp"), &cf);
        assert_eq!(
            k,
            "bb1dfe276a7165264b23349a3d349c470f451aa964f89c172bfb5bc8fdf9d548"
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
        let k1 = cache_key(default_opts("b.clsp"), &cf);
        cf.include_forms.push(desc(fp(&[4, 5])));
        let k2 = cache_key(default_opts("b.clsp"), &cf);
        assert_ne!(k1, k2);
        cf.include_forms.truncate(1);
        let k1_again = cache_key(default_opts("b.clsp"), &cf);
        assert_eq!(k1, k1_again);
    }

    #[test]
    fn cache_key_changes_with_cse_dominance() {
        let loc = Srcloc::start(&"cse.clsp".to_string());
        let cf = empty_compileform(loc);
        let dialect_before = AcceptedDialect {
            stepping: Some(26),
            strict: true,
            int_fix: true,
            extra_numeric_constants: false,
            cse_dominance: false,
        };
        let dialect_after = AcceptedDialect {
            cse_dominance: true,
            ..dialect_before.clone()
        };
        let before = default_opts("cse.clsp").set_dialect(dialect_before);
        let after = default_opts("cse.clsp").set_dialect(dialect_after);

        assert_ne!(cache_key(before, &cf), cache_key(after, &cf));
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
        let k = cache_key(default_opts("c.clsp"), &cf);
        assert!(!k.is_empty());
        assert_ne!(
            k,
            "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"
        );
    }
}
