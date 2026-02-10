#[cfg(not(test))]
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::rc::Rc;
#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
use cached::stores::SizedCache;
#[cfg(not(test))]
use cached::stores::{DiskCache, DiskCacheBuilder, DiskCacheError};

#[cfg(test)]
use cached::Cached;
#[cfg(not(test))]
use cached::IOCached;

use crate::compiler::clvm::sha256tree;
use crate::compiler::comptypes::{CompileErr, CompileForm, CompilerOpts, Export};
use crate::compiler::sexp::{enlist, SExp};
use crate::compiler::srcloc::Srcloc;

#[cfg(test)]
lazy_static! {
    pub static ref TEST_CACHE: Mutex<SizedCache<String, String>> =
        Mutex::new(SizedCache::with_size(1000 * 1000));
}

#[cfg(test)]
struct SizedCacheForTest;

trait AnyCache {
    fn cache_get(&self, loc: Srcloc, key: &String) -> Result<Option<String>, CompileErr>;
    fn cache_set(&mut self, loc: Srcloc, key: String, value: String) -> Result<(), CompileErr>;
}

fn dc_error_to_cerr<E: Debug>(loc: Srcloc) -> Box<dyn Fn(E) -> CompileErr> {
    Box::new(move |e: E| CompileErr(loc.clone(), format!("{e:?}")))
}

#[cfg(not(test))]
fn get_cache(loc: Srcloc) -> Result<Box<DiskCache<String, String>>, CompileErr> {
    let mut builder: DiskCacheBuilder<String, String> = DiskCache::new("chialisp");
    let build_error = dc_error_to_cerr(loc);
    builder = builder.set_disk_directory(".chialisp");
    builder.build().map_err(build_error).map(Box::new)
}

#[cfg(not(test))]
impl AnyCache for Box<DiskCache<String, String>> {
    fn cache_get(&self, loc: Srcloc, key: &String) -> Result<Option<String>, CompileErr> {
        let dc_error: Box<dyn Fn(DiskCacheError) -> CompileErr> = dc_error_to_cerr(loc);
        let dc_ref: &DiskCache<String, String> = self.borrow();
        dc_ref.cache_get(key).map_err(dc_error)
    }

    fn cache_set(&mut self, loc: Srcloc, key: String, value: String) -> Result<(), CompileErr> {
        let dc_error: Box<dyn Fn(DiskCacheError) -> CompileErr> = dc_error_to_cerr(loc);
        let dc_ref: &mut DiskCache<String, String> = self.borrow_mut();
        dc_ref.cache_set(key, value).map_err(dc_error)?;
        Ok(())
    }
}

#[cfg(test)]
fn get_cache(_loc: Srcloc) -> Result<SizedCacheForTest, CompileErr> {
    Ok(SizedCacheForTest)
}

#[cfg(test)]
impl AnyCache for SizedCacheForTest {
    fn cache_get(&self, loc: Srcloc, key: &String) -> Result<Option<String>, CompileErr> {
        let lock_error = dc_error_to_cerr(loc);
        let mut test_ref = TEST_CACHE.lock().map_err(lock_error)?;
        Ok(test_ref.cache_get(key).map(|c| c.clone()))
    }

    fn cache_set(&mut self, loc: Srcloc, key: String, value: String) -> Result<(), CompileErr> {
        let lock_error = dc_error_to_cerr(loc);
        let mut test_ref = TEST_CACHE.lock().map_err(lock_error)?;
        test_ref.cache_set(key, value);
        Ok(())
    }
}

fn cache_key(opts: Rc<dyn CompilerOpts>, cf: &CompileForm, exports: &[Export]) -> String {
    let cf_sexp = cf.to_sexp();
    let export_sexp_list: Vec<Rc<SExp>> = exports.iter().map(|e| e.to_sexp()).collect();
    let exports_sexp = enlist(cf.loc(), &export_sexp_list);
    let dialect_sexp = opts.dialect().to_sexp(cf.loc());
    hex::encode(sha256tree(Rc::new(enlist(
        cf.loc(),
        &[dialect_sexp, exports_sexp.into(), cf_sexp],
    ))))
}

pub fn try_element_from_cache_error(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
    export_path: &str,
) -> Result<Option<String>, CompileErr> {
    let dc = get_cache(cf.loc())?;
    let key = cache_key(opts.clone(), cf, exports);
    let hex_file_name = format!("{}!{}", key, export_path);
    dc.cache_get(cf.loc(), &hex_file_name)
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
    export_path: &str,
) -> Option<String> {
    try_element_from_cache_error(opts, cf, exports, export_path)
        .ok()
        .and_then(|x| x)
}

pub fn set_cache_element_error(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
    export_path: &str,
    export_hex: &str,
) -> Result<(), CompileErr> {
    let mut dc = get_cache(cf.loc())?;
    let key = cache_key(opts.clone(), cf, exports);
    let hex_data_key = format!("{}!{}", key, export_path);
    dc.cache_set(cf.loc(), hex_data_key, export_hex.to_string())?;
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
