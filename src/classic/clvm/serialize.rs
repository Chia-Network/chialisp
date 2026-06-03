/*
decoding:
read a byte
if it's 0xfe, it's nil (which might be same as 0)
if it's 0xff, it's a cons box. Read two items, build cons
otherwise, number of leading set bits is length in bytes to read size
0-0x7f are literal one byte values
leading bits is the count of bytes to read of size
0x80-0xbf is a size of one byte (perform logical and of first byte with 0x3f to get size)
0xc0-0xdf is a size of two bytes (perform logical and of first byte with 0x1f)
0xe0-0xef is 3 bytes ((perform logical and of first byte with 0xf))
0xf0-0xf7 is 4 bytes ((perform logical and of first byte with 0x7))
0xf7-0xfb is 5 bytes ((perform logical and of first byte with 0x3))
 */

use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::as_rust::{TToSexpF, TValStack};
use crate::classic::clvm::casts::int_from_bytes;
use crate::classic::clvm::sexp::{to_sexp_type, CastableType};
use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;
use clvm_rs::reduction::{Reduction, Response};

const MAX_SINGLE_BYTE: u32 = 0x7F;
const CONS_BOX_MARKER: u32 = 0xFF;

fn atom_size_blob(b: &Bytes) -> Result<(bool, Vec<u8>), String> {
    let size = b.length() as i64;
    if size == 0 {
        return Ok((false, vec![0x80]));
    } else if size == 1 && b.at(0) <= MAX_SINGLE_BYTE as u8 {
        return Ok((false, b.data().clone()));
    }

    if size < 0x40 {
        Ok((true, vec![0x80 | size as u8]))
    } else if size < 0x2000 {
        Ok((true, vec![0xC0 | ((size >> 8) as u8), (size & 0xFF) as u8]))
    } else if size < 0x100000 {
        Ok((
            true,
            vec![
                0xE0 | ((size >> 16) as u8),
                ((size >> 8) & 0xFF) as u8,
                (size & 0xFF) as u8,
            ],
        ))
    } else if size < 0x8000000 {
        Ok((
            true,
            vec![
                0xF0 | ((size >> 24) as u8),
                ((size >> 16) & 0xFF) as u8,
                ((size >> 8) & 0xFF) as u8,
                (size & 0xFF) as u8,
            ],
        ))
    } else if size < 0x400000000 {
        Ok((
            true,
            vec![
                0xF8 | ((size / (65536 * 65536)) as u8), // (size >> 32),
                ((size >> 24) & 0xFF) as u8,
                ((size >> 16) & 0xFF) as u8,
                ((size >> 8) & 0xFF) as u8,
                (size & 0xFF) as u8,
            ],
        ))
    } else {
        Err(format!("oversize bytes is unrepresentable {size:?}"))
    }
}

enum SExpToByteOp {
    Blob(Vec<u8>),
    Object(NodePtr),
}

struct SExpToBytesIterator<'a> {
    allocator: &'a mut Allocator,
    state: Vec<SExpToByteOp>,
}

impl<'a> SExpToBytesIterator<'a> {
    fn new(allocator: &'a mut Allocator, sexp: NodePtr) -> Self {
        SExpToBytesIterator {
            allocator,
            state: vec![SExpToByteOp::Object(sexp)],
        }
    }
}

impl Iterator for SExpToBytesIterator<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.pop().and_then(|step| match step {
            SExpToByteOp::Object(x) => match self.allocator.sexp(x) {
                SExp::Atom => {
                    // The only node we have in scope is x, so this atom
                    // capture is trivial.
                    let atom = self.allocator.atom(x);
                    let buf = atom.as_ref().to_vec();
                    let bytes = Bytes::new(Some(BytesFromType::Raw(buf.clone())));
                    match atom_size_blob(&bytes) {
                        Ok((original, b)) => {
                            if original {
                                self.state.push(SExpToByteOp::Blob(buf));
                            }
                            Some(b)
                        }
                        Err(_) => None,
                    }
                }
                SExp::Pair(f, r) => {
                    self.state.push(SExpToByteOp::Object(r));
                    self.state.push(SExpToByteOp::Object(f));
                    Some(vec![CONS_BOX_MARKER as u8])
                }
            },
            SExpToByteOp::Blob(b) => Some(b),
        })
    }
}

pub trait OpStackEntry {
    fn invoke<'a>(
        &self,
        allocator: &'a mut Allocator,
        op_stack: &mut TOpStack<'a>,
        val_stack: &mut TValStack,
        f: &mut Stream,
        to_sexp_f: Box<dyn TToSexpF<'a>>,
    ) -> Option<EvalErr>;
}

type TOpStack<'a> = Vec<Option<Box<dyn OpStackEntry>>>;

pub fn sexp_to_stream(allocator: &mut Allocator, sexp: NodePtr, f: &mut Stream) {
    for b in SExpToBytesIterator::new(allocator, sexp) {
        f.write(Bytes::new(Some(BytesFromType::Raw(b))));
    }
}

struct OpCons {}

impl OpStackEntry for OpCons {
    fn invoke<'a>(
        &self,
        allocator: &'a mut Allocator,
        _op_stack: &mut TOpStack<'a>,
        val_stack: &mut TValStack,
        _f: &mut Stream,
        to_sexp_f: Box<dyn TToSexpF<'a>>,
    ) -> Option<EvalErr> {
        match val_stack
            .pop()
            .and_then(|r| val_stack.pop().map(|l| (l, r)))
        {
            None => None,
            Some((l, r)) => {
                match to_sexp_f.invoke(allocator, CastableType::TupleOf(Rc::new(l), Rc::new(r))) {
                    Ok(c) => {
                        val_stack.push(CastableType::CLVMObject(c.1));
                        None
                    }
                    Err(e) => Some(e),
                }
            }
        }
    }
}

struct OpReadSexp {}

impl OpStackEntry for OpReadSexp {
    fn invoke<'a>(
        &self,
        allocator: &'a mut Allocator,
        op_stack: &mut TOpStack<'a>,
        val_stack: &mut TValStack,
        f: &mut Stream,
        to_sexp_f: Box<dyn TToSexpF<'a>>,
    ) -> Option<EvalErr> {
        let blob = f.read(1);
        if blob.length() == 0 {
            return Some(EvalErr::InternalError(
                NodePtr::NIL,
                "bad encoding".to_string(),
            ));
        }

        let b = blob.at(0);
        if b == CONS_BOX_MARKER as u8 {
            op_stack.push(Some(Box::new(OpCons {})));
            op_stack.push(Some(Box::new(OpReadSexp {})));
            op_stack.push(Some(Box::new(OpReadSexp {})));
            return None;
        }

        match atom_from_stream(allocator, f, b, to_sexp_f) {
            Ok(v) => {
                val_stack.push(CastableType::CLVMObject(v));
                None
            }
            Err(e) => Some(e),
        }
    }
}

pub struct SimpleCreateCLVMObject {}

impl<'a> TToSexpF<'a> for SimpleCreateCLVMObject {
    fn invoke(&self, allocator: &'a mut Allocator, v: CastableType) -> Response {
        to_sexp_type(allocator, v).map(|sexp| Reduction(1, sexp))
    }
}

pub fn sexp_from_stream<'a>(
    allocator: &'a mut Allocator,
    f: &mut Stream,
    to_sexp_f: Box<dyn TToSexpF<'a>>,
) -> Response {
    let mut op_stack: TOpStack = vec![Some(Box::new(OpReadSexp {}))];
    let mut val_stack: TValStack = vec![];

    while let Some(Some(func)) = op_stack.pop() {
        func.invoke(
            allocator,
            &mut op_stack,
            &mut val_stack,
            f,
            Box::new(SimpleCreateCLVMObject {}),
        );
    }

    if let Some(v) = val_stack.pop() {
        return to_sexp_f.invoke(allocator, v);
    }

    Err(EvalErr::InternalError(
        NodePtr::NIL,
        "No value left after conversion".to_string(),
    ))
}

pub fn atom_from_stream<'a>(
    allocator: &'a mut Allocator,
    f: &mut Stream,
    b_: u8,
    _to_sexp_f: Box<dyn TToSexpF<'a>>,
) -> Result<NodePtr, EvalErr> {
    let mut b = b_;

    if b == 0x80 {
        return Ok(NodePtr::NIL);
    } else if b <= MAX_SINGLE_BYTE as u8 {
        return allocator.new_atom(&[b]);
    }

    let mut bit_count = 0;
    let mut bit_mask = 0x80;

    while (b & bit_mask) != 0 {
        bit_count += 1;
        b ^= bit_mask;
        bit_mask >>= 1;
    }

    let mut size_blob = Bytes::new(Some(BytesFromType::Raw(vec![b])));
    if bit_count > 1 {
        let bin = f.read(bit_count - 1);
        if bin.length() != bit_count - 1 {
            return Err(EvalErr::InternalError(
                NodePtr::NIL,
                "bad encoding".to_string(),
            ));
        }
        size_blob = size_blob.concat(&bin);
    }
    int_from_bytes(size_blob, None).and_then(|size| {
        if size >= 0x400000000 {
            return Err(EvalErr::InternalError(
                NodePtr::NIL,
                "blob too large".to_string(),
            ));
        }
        let blob = f.read(size as usize);
        if blob.length() != size as usize {
            return Err(EvalErr::InternalError(
                NodePtr::NIL,
                "bad encoding".to_string(),
            ));
        }
        allocator.new_atom(blob.data())
    })
}
