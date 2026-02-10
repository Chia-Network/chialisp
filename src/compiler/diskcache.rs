use std::fmt::Debug;
use std::rc::Rc;

use cached::IOCached;
use cached::stores::{DiskCache, DiskCacheBuilder, DiskCacheError};

use crate::compiler::clvm::sha256tree;
use crate::compiler::comptypes::{CompileErr, CompileForm, CompilerOpts, Export};
use crate::compiler::sexp::{enlist, SExp};
use crate::compiler::srcloc::Srcloc;

fn dc_error_to_cerr<E: Debug>(loc: Srcloc) -> Box<dyn Fn(E) -> CompileErr> {
    Box::new(move |e: E| CompileErr(loc.clone(), format!("{e:?}")))
}

fn cache_key(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
) -> String {
    let cf_sexp = cf.to_sexp();
    let export_sexp_list: Vec<Rc<SExp>> = exports.iter().map(|e| e.to_sexp()).collect();
    let exports_sexp = enlist(cf.loc(), &export_sexp_list);
    let dialect_sexp = opts.dialect().to_sexp(cf.loc());
    hex::encode(&sha256tree(Rc::new(enlist(cf.loc(), &[
        dialect_sexp,
        exports_sexp.into(),
        cf_sexp,
    ]))))
}

pub fn try_element_from_cache_error(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
    export_path: &str
) -> Result<Option<String>, CompileErr> {
    let mut builder: DiskCacheBuilder<String, String> = DiskCache::new("chialisp");
    let build_error = dc_error_to_cerr(cf.loc());
    builder = builder.set_disk_directory(".chialisp");
    let dc = builder.build().map_err(build_error)?;
    let key = cache_key(opts.clone(), cf, exports);
    let dc_error: Box<dyn Fn(DiskCacheError) -> CompileErr> = dc_error_to_cerr(cf.loc());
    let hex_file_name = format!("{}!{}", key, export_path);
    dc.cache_get(&hex_file_name).map_err(dc_error)
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
    exports: &[Export],
    export_path: &str
) -> Option<String> {
    try_element_from_cache_error(opts, cf, exports, export_path).ok().and_then(|x| x)
}

pub fn set_cache_element_error(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
    export_path: &str,
    export_hex: &str,
) -> Result<(), CompileErr> {
    let mut builder: DiskCacheBuilder<String, String> = DiskCache::new("chialisp");
    let build_error = dc_error_to_cerr(cf.loc());
    let dc_error: Box<dyn Fn(DiskCacheError) -> CompileErr> = dc_error_to_cerr(cf.loc());
    builder = builder.set_disk_directory(".chialisp");
    let dc = builder.build().map_err(build_error)?;
    let key = cache_key(opts.clone(), cf, exports);
    let hex_data_key = format!("{}!{}", key, export_path);
    dc.cache_set(hex_data_key, export_hex.to_string()).map_err(dc_error)?;
    Ok(())
}

/// Set an element in the cache.  Use the current dialect, compileform and exports as the majority
/// of key material.  We add a file path and content to determine an exact hex serialization of
/// an export.
pub fn set_cache_element(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
    export_path: &str,
    export_hex: &str,
) {
    set_cache_element_error(opts, cf, exports, export_path, export_hex).ok();
}
