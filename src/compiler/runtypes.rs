use core::fmt::Display;
use alloc::rc::Rc;

use crate::compiler::comptypes::CompileErr;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunFailure {
    RunErr(Srcloc, String),
    RunExn(Srcloc, Rc<SExp>),
}

impl Display for RunFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            RunFailure::RunExn(l, s) => {
                l.fmt(formatter)?;
                formatter.write_str(": throw(x) ")?;
                s.fmt(formatter)?;
            }
            RunFailure::RunErr(l, s) => {
                l.fmt(formatter)?;
                formatter.write_str(": ")?;
                s.fmt(formatter)?;
            }
        }
        Ok(())
    }
}

impl From<RunFailure> for CompileErr {
    fn from(item: RunFailure) -> Self {
        match item {
            RunFailure::RunExn(l, s) => CompileErr(l.clone(), format!("Runtime exception: {s}")),
            RunFailure::RunErr(l, s) => CompileErr(l.clone(), format!("Runtime error: {s}")),
        }
    }
}

impl From<CompileErr> for RunFailure {
    fn from(e: CompileErr) -> Self {
        match e {
            CompileErr(l, e) => RunFailure::RunErr(l, e),
        }
    }
}
