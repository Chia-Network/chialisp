// use core::io; // TODO: io - commented out for no_std compatibility (core::io is unstable)
use alloc::string::String;
use core::{error::Error, fmt};

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

// TODO: io - commented out for no_std compatibility (core::io is unstable)
// impl From<SyntaxErr> for io::Error {
//     fn from(err: SyntaxErr) -> Self {
//         io::Error::other(err.msg)
//     }
// }
