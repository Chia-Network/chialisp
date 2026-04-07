use std::borrow::Borrow;
use std::collections::HashSet;
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Record, Stream};
use crate::classic::clvm::OPERATORS_LATEST_VERSION;
use crate::classic::clvm::{keyword_from_atom, keyword_to_atom};
use crate::classic::clvm_tools::ir::r#type::IRRepr;
use crate::classic::clvm_tools::ir::reader::IRReader;
use crate::classic::clvm_tools::ir::writer::write_ir;

lazy_static! {
    pub static ref PRINTABLE_CHARS: HashSet<char> =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ".chars().collect();
}
pub fn is_printable_string(s: &str) -> bool {
    s.chars().all(|s_char| PRINTABLE_CHARS.contains(&s_char))
}

pub fn assemble_from_ir(
    allocator: &mut Allocator,
    ir_sexp: Rc<IRRepr>,
) -> Result<NodePtr, EvalErr> {
    match ir_sexp.borrow() {
        IRRepr::Null => Ok(NodePtr::NIL),
        IRRepr::Quotes(b) => allocator.new_atom(b.data()),
        IRRepr::Int(b, _signed) => allocator.new_atom(b.data()),
        IRRepr::Hex(b) => allocator.new_atom(b.data()),
        IRRepr::Symbol(s) => {
            let mut s_real_name = s.clone();
            if let Some(stripped) = s.strip_prefix('#') {
                s_real_name = stripped.to_string();
            }

            match keyword_to_atom(OPERATORS_LATEST_VERSION).get(&s_real_name) {
                Some(v) => allocator.new_atom(v),
                None => {
                    let v: Vec<u8> = s_real_name.as_bytes().to_vec();
                    allocator.new_atom(&v)
                }
            }
        }
        IRRepr::Cons(l, r) => assemble_from_ir(allocator, l.clone()).and_then(|l| {
            assemble_from_ir(allocator, r.clone()).and_then(|r| allocator.new_pair(l, r))
        }),
    }
}

fn has_oversized_sign_extension(atom: &Bytes) -> bool {
    let data = atom.data();

    // Can't have an extra sign extension if the number is too short.
    // With the exception of 0.
    if atom.length() < 2 {
        return data.len() == 1 && data[0] == 0;
    }

    if data[0] == 0 {
        // This is a canonical value.  The opposite is non-canonical.
        // 0x0080 -> 128
        // 0x0000 -> 0x0000.  Non canonical because the second byte
        // wouldn't suggest sign extension so the first 0 is redundant.
        return data[1] & 0x80 == 0;
    } else if data[0] == 0xff {
        // This is a canonical value.  The opposite is non-canonical.
        // 0xff00 -> -256
        // 0xffff -> 0xffff.  Non canonical because the second byte
        // would suggest sign extension so the first 0xff is redundant.
        return data[1] & 0x80 != 0;
    }

    false
}

pub fn ir_for_atom(
    atom: &Bytes,
    allow_keyword: bool,
    keyword_from_atom: &Record<Vec<u8>, String>,
) -> IRRepr {
    if atom.length() == 0 {
        return IRRepr::Null;
    }
    if atom.length() > 2 {
        if let Ok(v) = String::from_utf8(atom.data().to_vec()) {
            if is_printable_string(&v) {
                return IRRepr::Quotes(atom.clone());
            }
        }
    } else {
        if allow_keyword {
            if let Some(kw) = keyword_from_atom.get(atom.data()) {
                return IRRepr::Symbol(kw.to_string());
            }
        }

        // Determine whether the bytes identity an integer in canonical form.
        // It's not canonical if there is oversized sign extension.
        if atom.data() != &[0] && !has_oversized_sign_extension(atom) {
            return IRRepr::Int(atom.clone(), true);
        }
    }
    IRRepr::Hex(atom.clone())
}

/*
 * (2 2 (2) (2 3 4)) => (a 2 (a) (a 3 4))
 */
pub fn disassemble_to_ir_with_kw(
    allocator: &Allocator,
    sexp: NodePtr,
    keyword_from_atom: &Record<Vec<u8>, String>,
    mut allow_keyword: bool,
) -> IRRepr {
    match allocator.sexp(sexp) {
        SExp::Pair(l, r) => {
            if let SExp::Pair(_, _) = allocator.sexp(l) {
                allow_keyword = true;
            }

            let v0 = disassemble_to_ir_with_kw(allocator, l, keyword_from_atom, allow_keyword);
            let v1 = disassemble_to_ir_with_kw(allocator, r, keyword_from_atom, false);
            IRRepr::Cons(Rc::new(v0), Rc::new(v1))
        }

        SExp::Atom => {
            // sexp is the only node in scope.
            let atom = allocator.atom(sexp);
            let bytes = Bytes::new(Some(BytesFromType::Raw(atom.as_ref().to_vec())));
            ir_for_atom(&bytes, allow_keyword, keyword_from_atom)
        }
    }
}

pub fn disassemble_with_kw(
    allocator: &Allocator,
    sexp: NodePtr,
    keyword_from_atom: &Record<Vec<u8>, String>,
) -> String {
    let with_keywords = !matches!(allocator.sexp(sexp), SExp::Atom);
    let symbols = disassemble_to_ir_with_kw(allocator, sexp, keyword_from_atom, with_keywords);
    write_ir(Rc::new(symbols))
}

pub fn disassemble(allocator: &Allocator, sexp: NodePtr, version: Option<usize>) -> String {
    disassemble_with_kw(
        allocator,
        sexp,
        keyword_from_atom(version.unwrap_or(OPERATORS_LATEST_VERSION)),
    )
}

pub fn assemble(allocator: &mut Allocator, s: &str) -> Result<NodePtr, EvalErr> {
    let v = s.as_bytes().to_vec();
    let stream = Stream::new(Some(Bytes::new(Some(BytesFromType::Raw(v)))));
    let mut reader = IRReader::new(stream);
    reader
        .read_expr()
        .map_err(|e| EvalErr::InternalError(NodePtr::NIL, e.to_string()))
        .and_then(|ir| assemble_from_ir(allocator, Rc::new(ir)))
}
