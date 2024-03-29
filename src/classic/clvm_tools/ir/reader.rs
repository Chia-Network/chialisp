use std::mem::swap;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{
    Bytes, BytesFromType, Stream, UnvalidatedBytesFromType,
};
use crate::classic::clvm::casts::bigint_to_bytes_clvm;
use crate::classic::clvm::syntax_error::SyntaxErr;
use crate::classic::clvm_tools::ir::r#type::IRRepr;
use crate::util::Number;

pub struct IRReader {
    stream: Stream,
}

// XXX Allows us to track line and column later if desired.
impl IRReader {
    fn read(&mut self, n: usize) -> Bytes {
        self.stream.read(n)
    }

    fn backup(&mut self, n: usize) {
        let cur_seek = self.stream.get_seek();
        if n > cur_seek {
            self.stream.set_seek(0);
        } else {
            self.stream.set_seek((cur_seek - n) as i64);
        }
    }

    pub fn read_expr(&mut self) -> Result<IRRepr, SyntaxErr> {
        consume_object(self)
    }

    pub fn new(s: Stream) -> Self {
        IRReader { stream: s }
    }
}

pub fn is_eol(chval: u8) -> bool {
    chval == b'\r' || chval == b'\n'
}

pub fn is_space(chval: u8) -> bool {
    chval == b' ' || chval == b'\t' || is_eol(chval)
}

pub fn consume_whitespace(s: &mut IRReader) {
    let mut in_comment = false;

    // This also deals with comments
    // eslint-disable-next-line no-constant-condition
    loop {
        let b = s.read(1);
        if b.length() == 0 {
            return;
        }

        let ch = b.at(0);
        if in_comment {
            if is_eol(ch) {
                in_comment = false;
            } else {
                continue;
            }
        }

        if ch == b';' {
            in_comment = true;
            continue;
        }

        if is_space(ch) {
            continue;
        }

        break;
    }

    s.backup(1);
}

pub fn consume_quoted(s: &mut IRReader, q: u8) -> Result<IRRepr, SyntaxErr> {
    let starting_at = s.stream.get_seek() - 1;
    let mut bs = false;
    let mut qchars = vec![q];

    loop {
        let b = s.read(1);
        if b.length() == 0 {
            return Err(SyntaxErr::new(format!(
                "unterminated string starting at {}: {}",
                starting_at,
                Bytes::new(Some(BytesFromType::Raw(qchars))).decode()
            )));
        }

        if bs {
            bs = false;
            qchars.push(b.at(0));
        } else if b.at(0) == b'\\' {
            bs = true;
        } else if b.at(0) == q {
            break;
        } else {
            qchars.push(b.at(0));
        }
    }

    // Exclude first quote that was captured.
    qchars = qchars.iter().skip(1).copied().collect();
    Ok(IRRepr::Quotes(Bytes::new(Some(BytesFromType::Raw(qchars)))))
}

pub fn is_hex(chars: &[u8]) -> bool {
    chars.len() > 2 && chars[0] == b'0' && (chars[1] == b'x' || chars[1] == b'X')
}

pub fn is_dec(chars: &[u8]) -> bool {
    let mut first = true;

    for ch in chars.iter() {
        if first {
            first = false;
            if *ch == b'-' {
                continue;
            }
        }
        if *ch > b'9' || *ch < b'0' {
            return false;
        }
    }

    true
}

pub fn interpret_atom_value(chars: &[u8]) -> Result<IRRepr, SyntaxErr> {
    // The Decimal and Hex representation of atoms in the program
    if chars.is_empty() {
        Ok(IRRepr::Null)
    } else if is_hex(chars) {
        let mut string_bytes = if chars.len() % 2 > 0 {
            // Pad an odd-length hex constant from the program text
            // with a zero in the High nibble
            Bytes::new(Some(BytesFromType::Raw(vec![b'0'])))
        } else {
            // This is the even-length hex constant case
            Bytes::new(None)
        };
        string_bytes =
            string_bytes.concat(&Bytes::new(Some(BytesFromType::Raw(chars[2..].to_vec()))));

        Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(string_bytes.decode())))
            .map(IRRepr::Hex)
    } else if let Some(n) = String::from_utf8(chars.to_vec())
        .ok()
        .and_then(|s| s.parse::<Number>().ok())
        .map(|n| bigint_to_bytes_clvm(&n))
    {
        Ok(IRRepr::Int(n, true))
    } else {
        let string_bytes = Bytes::new(Some(BytesFromType::Raw(chars.to_vec())));
        Ok(IRRepr::Symbol(string_bytes.decode()))
    }
}

pub fn consume_atom(s: &mut IRReader, b: &Bytes) -> Result<Option<IRRepr>, SyntaxErr> {
    let mut result_vec = b.data().to_vec();
    loop {
        let b = s.read(1);
        if b.length() == 0 {
            if result_vec.is_empty() {
                return Ok(None);
            } else {
                return interpret_atom_value(&result_vec).map(Some);
            }
        }

        if b.at(0) == b'(' || b.at(0) == b')' || is_space(b.at(0)) {
            s.backup(1);
            return interpret_atom_value(&result_vec).map(Some);
        }

        result_vec.push(b.at(0));
    }
}

fn enlist_ir(vec: &mut [IRRepr], tail: IRRepr) -> IRRepr {
    let mut result = tail;
    for i_reverse in 0..vec.len() {
        let i = vec.len() - i_reverse - 1;
        let mut next_head = IRRepr::Null;
        swap(&mut vec[i], &mut next_head);
        result = IRRepr::Cons(Rc::new(next_head), Rc::new(result));
    }
    result
}

pub fn consume_cons_body(s: &mut IRReader) -> Result<IRRepr, SyntaxErr> {
    let mut result = vec![];

    loop {
        consume_whitespace(s);

        let b = s.read(1);
        if b.length() == 0 {
            return Err(SyntaxErr::new("missing )".to_string()));
        }

        if b.at(0) == b')' {
            return Ok(enlist_ir(&mut result, IRRepr::Null));
        }

        if b.at(0) == b'(' {
            let v = consume_cons_body(s)?;
            result.push(v);
            continue;
        }

        if b.at(0) == b'.' {
            consume_whitespace(s);
            let v = consume_object(s)?;
            consume_whitespace(s);
            let b = s.read(1);
            if b.length() == 0 || b.at(0) != b')' {
                return Err(SyntaxErr::new("missing )".to_string()));
            }
            return Ok(enlist_ir(&mut result, v));
        }

        if b.at(0) == b'\"' || b.at(0) == b'\'' {
            let v = consume_quoted(s, b.at(0))?;
            result.push(v);
            continue;
        } else if let Some(f) = consume_atom(s, &b)? {
            result.push(f);
            continue;
        } else {
            return Err(SyntaxErr::new("missing )".to_string()));
        }
    }
}

pub fn consume_object(s: &mut IRReader) -> Result<IRRepr, SyntaxErr> {
    consume_whitespace(s);
    let b = s.read(1);

    if b.length() == 0 {
        Ok(IRRepr::Null)
    } else if b.at(0) == b'(' {
        consume_cons_body(s)
    } else if b.at(0) == b'\"' || b.at(0) == b'\'' {
        consume_quoted(s, b.at(0))
    } else if let Some(ir) = consume_atom(s, &b)? {
        Ok(ir)
    } else {
        Err(SyntaxErr::new("empty stream".to_string()))
    }
}

pub fn read_ir(s: &str) -> Result<IRRepr, SyntaxErr> {
    let bytes_of_string = Bytes::new(Some(BytesFromType::Raw(s.as_bytes().to_vec())));
    let stream = Stream::new(Some(bytes_of_string));
    let mut reader = IRReader::new(stream);
    reader.read_expr()
}
