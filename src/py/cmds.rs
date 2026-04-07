use pyo3::prelude::*;

use crate::classic::clvm_tools::cmds;

#[pyfunction]
pub fn brun_main(args: Vec<String>) {
    cmds::brun(&args);
}

#[pyfunction]
pub fn run_main(args: Vec<String>) {
    cmds::run(&args);
}

#[pyfunction]
pub fn opc_main(args: Vec<String>) {
    cmds::opc(&args);
}

#[pyfunction]
pub fn opd_main(args: Vec<String>) {
    cmds::opd(&args);
}

#[pyfunction]
pub fn cldb_main(args: Vec<String>) {
    cmds::cldb(&args);
}

/// A Python module implemented in Rust.
pub fn create_cmds_module(py: Python) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "cmds")?;
    m.add_function(wrap_pyfunction!(brun_main, &m)?)?;
    m.add_function(wrap_pyfunction!(run_main, &m)?)?;
    m.add_function(wrap_pyfunction!(opc_main, &m)?)?;
    m.add_function(wrap_pyfunction!(opd_main, &m)?)?;
    m.add_function(wrap_pyfunction!(cldb_main, &m)?)?;
    Ok(m)
}
