use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

lazy_static! {
    pub static ref ARGNAME_CTR: AtomicUsize = AtomicUsize::new(0);
}

/// As (gensym ...) in lisp.
pub fn gensym(name: Vec<u8>) -> Vec<u8> {
    let count = ARGNAME_CTR.fetch_add(1, Ordering::SeqCst);
    let mut result_vec = name;
    let number_value = format!("{}", count + 1);
    result_vec.append(&mut "_$_".as_bytes().to_vec());
    result_vec.append(&mut number_value.as_bytes().to_vec());
    result_vec
}
