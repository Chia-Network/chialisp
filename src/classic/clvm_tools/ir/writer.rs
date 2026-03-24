use std::borrow::Borrow;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::casts::{bigint_from_bytes, TConvertOption};

use crate::classic::clvm_tools::ir::r#type::{IRRepr, NEW_BIT_CONSTANTS};

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
    language_flags: u32,
}

impl IROutputIterator {
    fn new(ir_sexp: Rc<IRRepr>, flags: u32) -> IROutputIterator {
        IROutputIterator {
            state: vec![IROutputState::Start(ir_sexp)],
            language_flags: flags,
        }
    }
}

fn output_with_radix(bits: usize, bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::default();
    let raw_content_bits = 8 * bytes.len();
    let digit_mask = (1 << bits) - 1;
    let digits = (raw_content_bits + bits) / bits;
    let digit_bits = bits * digits;
    let mut buffer_bit = digit_bits % 8;
    let mut buffer: u32 = 0;

    if bytes.is_empty() {
        result.push(b'0');
        return result;
    }

    // If the leftmost byte is zero, then we must include a binary or octal digit
    // to indicate that it should be padded.
    let mut need_padding = bytes[0] == 0;

    if need_padding && bytes.len() == 1 {
        result.push(b'0');
        result.push(b'0');
        return result;
    }

    let mut produce_output = false;

    for byte in bytes.iter() {
        buffer = (buffer << 8) | *byte as u32;
        buffer_bit += 8;
        while buffer_bit >= bits {
            buffer_bit -= bits;
            let digit_value = (buffer >> buffer_bit) & digit_mask;
            if digit_value != 0 || need_padding && !produce_output && buffer_bit < bits {
                produce_output = true;
                need_padding = false;
            }
            if produce_output {
                result.push(b'0' + (digit_value as u8));
            }
        }
        // Regardless of anything else, start producing output on the second
        // byte.
        produce_output = true;
    }

    result
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
                    IRRepr::Octal(o) => {
                        if (self.language_flags & NEW_BIT_CONSTANTS) != 0 {
                            return Some(
                                "0o".to_string()
                                    + &String::from_utf8_lossy(&output_with_radix(3, o.data())),
                            );
                        }

                        return Some("0x".to_string() + &o.hex());
                    }
                    IRRepr::Binary(b) => {
                        if (self.language_flags & NEW_BIT_CONSTANTS) != 0 {
                            return Some(
                                "0b".to_string()
                                    + &String::from_utf8_lossy(&output_with_radix(1, b.data())),
                            );
                        }

                        return Some("0x".to_string() + &b.hex());
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

pub fn write_ir_to_stream(ir_sexp: Rc<IRRepr>, f: &mut Stream, language_flags: u32) {
    for b in IROutputIterator::new(ir_sexp, language_flags) {
        f.write(Bytes::new(Some(BytesFromType::String(b))));
    }
}

pub fn write_ir(ir_sexp: Rc<IRRepr>, language_flags: u32) -> String {
    let mut s = Stream::new(None);
    write_ir_to_stream(ir_sexp, &mut s, language_flags);
    s.get_value().decode()
}
