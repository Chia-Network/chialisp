use std::ops::Index;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;

use crate::classic::clvm_tools::binutils::disassemble;
use crate::compiler::srcloc::Srcloc;

pub enum ASExp<T> {
    Pair(T, T),
    Atom,
}

#[derive(Debug)]
pub struct ClError(pub Srcloc, pub EvalErr);

impl std::fmt::Display for ClError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(formatter, "{}", self.1)
    }
}

impl From<ClError> for EvalErr {
    fn from(err: ClError) -> Self {
        err.1
    }
}

pub trait BufCarrier<'a> {
    fn as_ref(&'a self) -> &'a [u8];
}

#[derive(PartialEq, Eq)]
pub struct BufHolder<'a>(clvmr::Atom<'a>);

impl<'a> Index<usize> for BufHolder<'a> {
    type Output = u8;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.as_ref()[idx]
    }
}

impl<'a> BufCarrier<'a> for BufHolder<'a> {
    fn as_ref(&'a self) -> &'a [u8] {
        self.0.as_ref()
    }
}

pub trait ClassicAllocator {
    type NodePtr;
    fn loc(&self, node: &Self::NodePtr) -> Srcloc;
    fn sexp(&self, node: &Self::NodePtr) -> ASExp<Self::NodePtr>;
    fn atom<'a>(&'a self, node: &Self::NodePtr) -> BufHolder<'a>;
    fn is_nil(&self, node: &Self::NodePtr) -> bool;
    fn disassemble(&self, node: &Self::NodePtr, version: Option<usize>) -> String;
    fn allocator(&mut self) -> &mut Allocator;
    fn node_equal(&self, a: &Self::NodePtr, b: &Self::NodePtr) -> bool;
    fn map_err(&self, loc: Srcloc, err: EvalErr) -> ClError;
    fn new_atom(&mut self, loc: Srcloc, value: &[u8]) -> Result<Self::NodePtr, ClError>;
    fn new_pair(
        &mut self,
        loc: Srcloc,
        a: &Self::NodePtr,
        b: &Self::NodePtr,
    ) -> Result<Self::NodePtr, ClError>;
    fn import(&mut self, loc: Srcloc, node: NodePtr) -> Result<Self::NodePtr, ClError>;
    fn export(&self, node: &Self::NodePtr) -> NodePtr;
}

thread_local! {
    pub static DEFAULT_SRCLOC: Srcloc = Srcloc::start("*clvm*");
}

impl ClassicAllocator for Allocator {
    type NodePtr = clvmr::NodePtr;

    fn loc(&self, _node: &Self::NodePtr) -> Srcloc {
        DEFAULT_SRCLOC.with(|s| s.clone())
    }
    fn sexp(&self, node: &Self::NodePtr) -> ASExp<Self::NodePtr> {
        match clvmr::Allocator::sexp(self, *node) {
            SExp::Pair(a, b) => ASExp::Pair(a, b),
            SExp::Atom => ASExp::Atom,
        }
    }
    fn atom<'a>(&'a self, node: &Self::NodePtr) -> BufHolder<'a> {
        BufHolder(self.atom(*node))
    }
    fn is_nil(&self, node: &Self::NodePtr) -> bool {
        *node == NodePtr::NIL
    }
    fn disassemble(&self, node: &Self::NodePtr, version: Option<usize>) -> String {
        disassemble(self, *node, version)
    }
    fn allocator(&mut self) -> &mut Allocator {
        self
    }
    fn node_equal(&self, a: &Self::NodePtr, b: &Self::NodePtr) -> bool {
        *a == *b
    }
    fn map_err(&self, loc: Srcloc, err: EvalErr) -> ClError {
        ClError(loc, err)
    }
    fn new_atom(&mut self, loc: Srcloc, value: &[u8]) -> Result<Self::NodePtr, ClError> {
        let new_atom =
            clvmr::Allocator::new_atom(self, value).map_err(|e| self.map_err(loc.clone(), e))?;
        self.import(loc, new_atom)
    }
    fn new_pair(
        &mut self,
        loc: Srcloc,
        a: &Self::NodePtr,
        b: &Self::NodePtr,
    ) -> Result<Self::NodePtr, ClError> {
        let new_pair =
            clvmr::Allocator::new_pair(self, *a, *b).map_err(|e| self.map_err(loc.clone(), e))?;
        self.import(loc, new_pair)
    }
    fn import(&mut self, _loc: Srcloc, node: NodePtr) -> Result<Self::NodePtr, ClError> {
        Ok(node)
    }
    fn export(&self, node: &Self::NodePtr) -> NodePtr {
        *node
    }
}
