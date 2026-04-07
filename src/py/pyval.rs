use num::BigInt;

use pyo3::types::{PyAny, PyBytes, PyList, PyTuple};
use pyo3::{prelude::*, IntoPyObjectExt};

use std::borrow::Borrow;
use std::ffi::CString;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::bi_zero;
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;

pub fn map_err_to_pyerr<PA>(srcloc: Srcloc, r: PyResult<PA>) -> Result<PA, RunFailure> {
    r.map_err(|e| RunFailure::RunErr(srcloc, format!("{e}")))
}

pub fn python_value_to_clvm(val: Bound<'_, PyAny>) -> Result<Rc<SExp>, RunFailure> {
    let srcloc = Srcloc::start("*python*");
    val.downcast::<PyList>()
        .ok()
        .map(|l| {
            if l.is_empty() {
                Ok(Rc::new(SExp::Nil(srcloc.clone())))
            } else {
                let mut result = SExp::Nil(srcloc.clone());
                for i_rev in 0..l.len() {
                    let i = l.len() - i_rev - 1;
                    let any_of_elt = map_err_to_pyerr(srcloc.clone(), l.get_item(i))?;
                    result = SExp::Cons(
                        srcloc.clone(),
                        python_value_to_clvm(any_of_elt)?,
                        Rc::new(result),
                    );
                }
                Ok(Rc::new(result))
            }
        })
        .map(Some)
        .unwrap_or_else(|| {
            val.downcast::<PyTuple>()
                .map(|t| {
                    if t.len() != 2 {
                        Err(RunFailure::RunErr(
                            srcloc.clone(),
                            "tuple must have len 2".to_string(),
                        ))
                    } else {
                        let any_of_e0 = map_err_to_pyerr(srcloc.clone(), t.get_item(0))?;
                        let any_of_e1 = map_err_to_pyerr(srcloc.clone(), t.get_item(1))?;
                        Ok(Rc::new(SExp::Cons(
                            srcloc.clone(),
                            python_value_to_clvm(any_of_e0)?,
                            python_value_to_clvm(any_of_e1)?,
                        )))
                    }
                })
                .ok()
        })
        .map(Some)
        .unwrap_or_else(|| {
            val.downcast::<PyBytes>()
                .map(|b| Ok(Rc::new(SExp::Atom(srcloc.clone(), b.as_bytes().to_vec()))))
                .ok()
        })
        .map(Some)
        .unwrap_or_else(|| {
            let stringified = format!("{val}");
            stringified
                .parse::<BigInt>()
                .map(|i| {
                    if i == bi_zero() {
                        Ok(Rc::new(SExp::Nil(srcloc.clone())))
                    } else {
                        Ok(Rc::new(SExp::Integer(srcloc.clone(), i)))
                    }
                })
                .ok()
        })
        .unwrap_or_else(|| {
            Err(RunFailure::RunErr(
                srcloc.clone(),
                "no way to convert python value to clvm".to_string(),
            ))
        })
}

pub fn clvm_value_to_python(py: Python, val: Rc<SExp>) -> PyResult<Py<PyAny>> {
    val.proper_list()
        .map(|lst| {
            let mut vallist = Vec::new();
            for i in lst {
                vallist.push(clvm_value_to_python(py, Rc::new(i.clone()))?);
            }
            PyList::new(py, &vallist)?.into_py_any(py)
        })
        .unwrap_or_else(|| match val.borrow() {
            SExp::Cons(_, a, b) => PyTuple::new(
                py,
                vec![
                    clvm_value_to_python(py, a.clone())?,
                    clvm_value_to_python(py, b.clone())?,
                ],
            )?
            .into_py_any(py),
            SExp::Integer(_, i) => {
                let int_val: Py<PyAny> = map_err_to_pyerr(
                    val.loc(),
                    py.eval(&CString::new(format!("int({i})")).unwrap(), None, None)
                        .map(|x| x.into_py_any(py))?,
                )
                .unwrap();
                Ok(int_val)
            }
            SExp::Atom(_, v) => PyBytes::new(py, v).into_py_any(py),
            SExp::QuotedString(_, _, v) => PyBytes::new(py, v).into_py_any(py),
            SExp::Nil(_) => {
                let emptybytes: Vec<u8> = vec![];
                PyList::new(py, &emptybytes)?.into_py_any(py)
            }
        })
}
