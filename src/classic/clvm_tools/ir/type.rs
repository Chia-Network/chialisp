use alloc::rc::Rc;
use alloc::string::String;

use crate::classic::clvm::__type_compatibility__::Bytes;

#[derive(Debug)]
pub enum IRRepr {
    Cons(Rc<IRRepr>, Rc<IRRepr>),
    Null,
    Quotes(Bytes),
    Int(Bytes, bool),
    Hex(Bytes),
    Symbol(String),
}
