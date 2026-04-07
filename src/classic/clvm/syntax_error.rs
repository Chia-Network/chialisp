use std::io;
use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxErr {
    pub msg: String,
}

impl Error for SyntaxErr {}

impl SyntaxErr {
    pub fn new(s: String) -> Self {
        SyntaxErr { msg: s }
    }
}

impl fmt::Display for SyntaxErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl From<SyntaxErr> for io::Error {
    fn from(err: SyntaxErr) -> Self {
        io::Error::other(err.msg)
    }
}
