use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::Bytes;

pub const NEW_BIT_CONSTANTS: u32 = 1;

#[derive(Debug)]
pub enum IRRepr {
    Cons(Rc<IRRepr>, Rc<IRRepr>),
    Null,
    Quotes(Bytes),
    Int(Bytes, bool),
    Hex(Bytes),
    Octal(Bytes),
    Binary(Bytes),
    Symbol(String),
}
