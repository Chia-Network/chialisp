use std::cell::Cell;

thread_local! {
    // Numbering for the names the compiler makes up. Per thread, and started
    // over for each nested compilation unit (see GensymUnit), so that what a
    // unit compiles to depends on its own source alone.
    static ARGNAME_CTR: Cell<usize> = const { Cell::new(0) };
}

/// As (gensym ...) in lisp.
pub fn gensym(name: Vec<u8>) -> Vec<u8> {
    let count = ARGNAME_CTR.with(|ctr| {
        let count = ctr.get();
        ctr.set(count + 1);
        count
    });
    let mut result_vec = name;
    let number_value = format!("{}", count + 1);
    result_vec.append(&mut "_$_".as_bytes().to_vec());
    result_vec.append(&mut number_value.as_bytes().to_vec());
    result_vec
}

/// Numbers a nested compilation unit from zero, as a top-level compile of the
/// same source would, and resumes the enclosing unit's count when dropped.
pub(crate) struct GensymUnit {
    outer_count: usize,
}

impl GensymUnit {
    pub(crate) fn begin() -> Self {
        GensymUnit {
            outer_count: ARGNAME_CTR.with(|ctr| ctr.replace(0)),
        }
    }
}

impl Drop for GensymUnit {
    fn drop(&mut self) {
        ARGNAME_CTR.with(|ctr| ctr.set(self.outer_count));
    }
}
