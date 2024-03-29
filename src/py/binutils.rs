use std::collections::HashMap;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::serialize::{sexp_from_stream, SimpleCreateCLVMObject};
use clvm_rs::allocator::{Allocator, NodePtr, SExp};

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use crate::classic::clvm_tools::binutils;

create_exception!(mymodule, ConvError, PyException);

fn convert_to_external(
    allocator: &Allocator,
    cons: &PyAny,
    from_bytes: &PyAny,
    root_node: NodePtr,
) -> PyResult<PyObject> {
    let mut stack: Vec<NodePtr> = vec![root_node];
    let mut finished = HashMap::new();

    while let Some(node) = stack.last() {
        let node = *node; // To avoid borrowing issues with the stack
        match allocator.sexp(node) {
            SExp::Pair(left, right) => {
                let left_finished = finished.contains_key(&left);
                let right_finished = finished.contains_key(&right);

                if left_finished && right_finished {
                    stack.pop();

                    let result: PyObject = Python::with_gil(|py| {
                        let a = finished.get(&left).unwrap();
                        let b = finished.get(&right).unwrap();
                        let args = PyTuple::new(py, &[a, b]);
                        cons.call1(args)
                    })?
                    .into();

                    finished.insert(node, result);
                } else {
                    if !left_finished {
                        stack.push(left);
                    }
                    if !right_finished {
                        stack.push(right);
                    }
                }
            }
            SExp::Atom => {
                stack.pop();

                if !finished.contains_key(&node) {
                    let converted: PyObject = Python::with_gil(|py| {
                        let bytes = PyBytes::new(py, allocator.atom(node));
                        let args = PyTuple::new(py, &[bytes]);
                        from_bytes.call1(args)
                    })?
                    .into();
                    finished.insert(node, converted);
                }
            }
        }
    }

    finished
        .get(&root_node)
        .cloned()
        .ok_or_else(|| ConvError::new_err("error converting assembled value"))
}

#[pyfunction]
pub fn assemble_generic(cons: &PyAny, from_bytes: &PyAny, args: String) -> PyResult<PyObject> {
    let mut allocator = Allocator::new();
    let assembled =
        binutils::assemble(&mut allocator, &args).map_err(|e| ConvError::new_err(e.to_string()))?;
    convert_to_external(&allocator, cons, from_bytes, assembled)
}

#[pyfunction]
pub fn disassemble_generic(program_bytes: &PyBytes) -> PyResult<String> {
    let mut allocator = Allocator::new();
    let mut stream = Stream::new(Some(Bytes::new(Some(BytesFromType::Raw(
        program_bytes.as_bytes().to_vec(),
    )))));

    let sexp = sexp_from_stream(
        &mut allocator,
        &mut stream,
        Box::new(SimpleCreateCLVMObject {}),
    )
    .map_err(|e| ConvError::new_err(e.to_string()))?;

    let disassembled = binutils::disassemble(&allocator, sexp.1, Some(0));
    Ok(disassembled)
}

pub fn create_binutils_module(py: Python) -> PyResult<&'_ PyModule> {
    let m = PyModule::new(py, "binutils")?;
    m.add_function(wrap_pyfunction!(assemble_generic, m)?)?;
    m.add_function(wrap_pyfunction!(disassemble_generic, m)?)?;
    Ok(m)
}
