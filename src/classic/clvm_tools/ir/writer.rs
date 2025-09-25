use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::{vec, vec::Vec};
use core::borrow::Borrow;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::casts::{bigint_from_bytes, TConvertOption};

use crate::classic::clvm_tools::ir::r#type::IRRepr;

#[derive(Debug)]
enum IROutputState {
    Start(Rc<IRRepr>),
    MaybeSep(Rc<IRRepr>),
    ListOf(Rc<IRRepr>),
    DotThen(Rc<IRRepr>),
    EndParen,
}

#[derive(Debug)]
struct IROutputIterator {
    state: Vec<IROutputState>,
}

impl IROutputIterator {
    fn new(ir_sexp: Rc<IRRepr>) -> IROutputIterator {
        IROutputIterator {
            state: vec![IROutputState::Start(ir_sexp)],
        }
    }
}

impl Iterator for IROutputIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.state.pop() {
                None => {
                    return None;
                }
                Some(IROutputState::EndParen) => {
                    return Some(")".to_string());
                }
                Some(IROutputState::Start(v)) => match v.borrow() {
                    IRRepr::Cons(l, r) => {
                        self.state.push(IROutputState::ListOf(Rc::new(IRRepr::Cons(
                            l.clone(),
                            r.clone(),
                        ))));
                        return Some("(".to_string());
                    }
                    IRRepr::Null => {
                        return Some("()".to_string());
                    }
                    IRRepr::Quotes(q) => {
                        return Some(q.to_formal_string());
                    }
                    IRRepr::Int(i, signed) => {
                        let opts = TConvertOption { signed: *signed };
                        return Some(bigint_from_bytes(i, Some(opts)).to_string());
                    }
                    IRRepr::Hex(h) => {
                        return Some("0x".to_string() + &h.hex());
                    }
                    IRRepr::Symbol(s) => {
                        return Some(s.to_string());
                    }
                },
                Some(IROutputState::MaybeSep(sub)) => match sub.borrow() {
                    IRRepr::Null => {
                        self.state.push(IROutputState::EndParen);
                    }
                    _ => {
                        self.state.push(IROutputState::ListOf(sub.clone()));
                        return Some(" ".to_string());
                    }
                },
                Some(IROutputState::ListOf(v)) => match v.borrow() {
                    IRRepr::Cons(l, r) => {
                        self.state.push(IROutputState::MaybeSep(r.clone()));
                        self.state.push(IROutputState::Start(l.clone()));
                    }
                    IRRepr::Null => {
                        self.state.push(IROutputState::EndParen);
                    }
                    _ => {
                        self.state.push(IROutputState::EndParen);
                        self.state.push(IROutputState::DotThen(v.clone()));
                        return Some(". ".to_string());
                    }
                },
                Some(IROutputState::DotThen(v)) => match v.borrow() {
                    IRRepr::Cons(l, r) => {
                        self.state.push(IROutputState::ListOf(r.clone()));
                        self.state.push(IROutputState::Start(l.clone()));
                    }
                    _ => {
                        self.state.push(IROutputState::Start(v.clone()));
                    }
                },
            }
        }
    }
}

pub fn write_ir_to_stream(ir_sexp: Rc<IRRepr>, f: &mut Stream) {
    for b in IROutputIterator::new(ir_sexp) {
        f.write(Bytes::new(Some(BytesFromType::String(b))));
    }
}

pub fn write_ir(ir_sexp: Rc<IRRepr>) -> String {
    let mut s = Stream::new(None);
    write_ir_to_stream(ir_sexp, &mut s);
    s.get_value().decode()
}
