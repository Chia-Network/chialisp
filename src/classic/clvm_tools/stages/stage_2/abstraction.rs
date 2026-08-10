use std::rc::Rc;

use std::ops::Index;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;

use crate::classic::clvm::__type_compatibility__::bi_zero;
use crate::classic::clvm_tools::binutils::disassemble;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::compiler::clvm;
use crate::compiler::prims;
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::SExp as ModernSExp;
use crate::compiler::srcloc::Srcloc;
use crate::util::{number_from_u8, u8_from_number};

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
    /// Run CLVM using the representation best suited to this allocator.
    ///
    /// The supplied runner remains the fallback dialect for non-core
    /// operators. Located allocators may evaluate core CLVM without first
    /// exporting the program to clvmr.
    fn run_clvm(
        &mut self,
        runner: Rc<dyn TRunProgram>,
        program: &Self::NodePtr,
        args: &Self::NodePtr,
    ) -> Result<Self::NodePtr, ClError>;
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
    fn run_clvm(
        &mut self,
        runner: Rc<dyn TRunProgram>,
        program: &Self::NodePtr,
        args: &Self::NodePtr,
    ) -> Result<Self::NodePtr, ClError> {
        let loc = self.loc(program);
        runner
            .run_program(self, *program, *args, None)
            .map(|result| result.1)
            .map_err(|err| ClError(loc, err))
    }
}

/// A stage-2 node backed by the compiler's location-aware S-expression.
///
/// `raw` mirrors `sexp` in the contained CLVM allocator. Stage-2 occasionally
/// has to execute generated CLVM, so retaining both representations lets those
/// boundaries use clvmr without discarding locations from the source tree.
#[derive(Clone, Debug)]
pub struct SExpNode {
    pub sexp: Rc<ModernSExp>,
    raw: NodePtr,
}

impl std::fmt::Display for SExpNode {
    fn fmt(&self, formatter: &'_ mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(formatter, "{}", self.sexp)
    }
}

/// Adapts modern compiler S-expressions to the classic stage-2 compiler.
pub struct SExpClassicAllocator {
    allocator: Allocator,
}

impl Default for SExpClassicAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SExpClassicAllocator {
    pub fn new() -> Self {
        Self {
            allocator: Allocator::new(),
        }
    }

    pub fn from_sexp(&mut self, sexp: Rc<ModernSExp>) -> Result<SExpNode, ClError> {
        let loc = sexp.loc();
        let raw = match sexp.as_ref() {
            ModernSExp::Nil(_) => NodePtr::NIL,
            ModernSExp::Cons(_, first, rest) => {
                let first = self.from_sexp(first.clone())?;
                let rest = self.from_sexp(rest.clone())?;
                self.allocator
                    .new_pair(first.raw, rest.raw)
                    .map_err(|e| self.map_err(loc.clone(), e))?
            }
            ModernSExp::Integer(_, value) => if *value == bi_zero() {
                self.allocator.new_atom(&[])
            } else {
                self.allocator.new_atom(&u8_from_number(value.clone()))
            }
            .map_err(|e| self.map_err(loc.clone(), e))?,
            ModernSExp::QuotedString(_, _, value) | ModernSExp::Atom(_, value) => self
                .allocator
                .new_atom(value)
                .map_err(|e| self.map_err(loc.clone(), e))?,
        };
        Ok(SExpNode { sexp, raw })
    }
}

impl ClassicAllocator for SExpClassicAllocator {
    type NodePtr = SExpNode;

    fn loc(&self, node: &Self::NodePtr) -> Srcloc {
        node.sexp.loc()
    }

    fn sexp(&self, node: &Self::NodePtr) -> ASExp<Self::NodePtr> {
        match node.sexp.as_ref() {
            ModernSExp::Cons(_, first, rest) => {
                let SExp::Pair(raw_first, raw_rest) = self.allocator.sexp(node.raw) else {
                    unreachable!("modern and raw S-expression representations diverged")
                };
                ASExp::Pair(
                    SExpNode {
                        sexp: first.clone(),
                        raw: raw_first,
                    },
                    SExpNode {
                        sexp: rest.clone(),
                        raw: raw_rest,
                    },
                )
            }
            _ => ASExp::Atom,
        }
    }

    fn atom<'a>(&'a self, node: &Self::NodePtr) -> BufHolder<'a> {
        BufHolder(self.allocator.atom(node.raw))
    }

    fn is_nil(&self, node: &Self::NodePtr) -> bool {
        node.raw == NodePtr::NIL
    }

    fn disassemble(&self, node: &Self::NodePtr, version: Option<usize>) -> String {
        disassemble(&self.allocator, node.raw, version)
    }

    fn allocator(&mut self) -> &mut Allocator {
        &mut self.allocator
    }

    fn node_equal(&self, a: &Self::NodePtr, b: &Self::NodePtr) -> bool {
        a.raw == b.raw
    }

    fn map_err(&self, loc: Srcloc, err: EvalErr) -> ClError {
        ClError(loc, err)
    }

    fn new_atom(&mut self, loc: Srcloc, value: &[u8]) -> Result<Self::NodePtr, ClError> {
        let raw = self
            .allocator
            .new_atom(value)
            .map_err(|e| ClError(loc.clone(), e))?;
        let sexp = if value.is_empty() {
            ModernSExp::Nil(loc)
        } else {
            let integer = number_from_u8(value);
            if u8_from_number(integer.clone()) == value {
                ModernSExp::Integer(loc, integer)
            } else {
                ModernSExp::Atom(loc, value.to_vec())
            }
        };
        Ok(SExpNode {
            sexp: Rc::new(sexp),
            raw,
        })
    }

    fn new_pair(
        &mut self,
        loc: Srcloc,
        a: &Self::NodePtr,
        b: &Self::NodePtr,
    ) -> Result<Self::NodePtr, ClError> {
        let raw = self
            .allocator
            .new_pair(a.raw, b.raw)
            .map_err(|e| ClError(loc.clone(), e))?;
        Ok(SExpNode {
            sexp: Rc::new(ModernSExp::Cons(loc, a.sexp.clone(), b.sexp.clone())),
            raw,
        })
    }

    fn import(&mut self, loc: Srcloc, node: NodePtr) -> Result<Self::NodePtr, ClError> {
        let sexp = match self.allocator.sexp(node) {
            SExp::Atom if node == NodePtr::NIL => Rc::new(ModernSExp::Nil(loc)),
            SExp::Atom => {
                let value = self.allocator.atom(node);
                let value = value.as_ref();
                let integer = number_from_u8(value);
                if u8_from_number(integer.clone()) == value {
                    Rc::new(ModernSExp::Integer(loc, integer))
                } else {
                    Rc::new(ModernSExp::Atom(loc, value.to_vec()))
                }
            }
            SExp::Pair(first, rest) => {
                let first = self.import(loc.clone(), first)?;
                let rest = self.import(loc.clone(), rest)?;
                Rc::new(ModernSExp::Cons(loc, first.sexp, rest.sexp))
            }
        };
        Ok(SExpNode { sexp, raw: node })
    }

    fn export(&self, node: &Self::NodePtr) -> NodePtr {
        node.raw
    }

    fn run_clvm(
        &mut self,
        runner: Rc<dyn TRunProgram>,
        program: &Self::NodePtr,
        args: &Self::NodePtr,
    ) -> Result<Self::NodePtr, ClError> {
        let result = clvm::run(
            &mut self.allocator,
            runner,
            prims::prim_map(),
            program.sexp.clone(),
            args.sexp.clone(),
            None,
            None,
        )
        .map_err(|err| match err {
            RunFailure::RunErr(loc, message) => {
                ClError(loc, EvalErr::InternalError(NodePtr::NIL, message))
            }
            RunFailure::RunExn(loc, value) => ClError(
                loc,
                EvalErr::InternalError(NodePtr::NIL, format!("exception: {value}")),
            ),
        })?;

        self.from_sexp(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassicAllocator, SExpClassicAllocator};
    use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
    use crate::compiler::sexp::{enlist, SExp};
    use crate::compiler::srcloc::Srcloc;
    use num_bigint::ToBigInt;
    use std::rc::Rc;

    fn quote(loc: Srcloc, value: Rc<SExp>) -> Rc<SExp> {
        Rc::new(SExp::Cons(
            loc.clone(),
            Rc::new(SExp::Integer(loc, 1_i32.to_bigint().unwrap())),
            value,
        ))
    }

    #[test]
    fn located_allocator_runs_core_clvm_without_losing_result_locations() {
        let file = Rc::new("*located-runner-test*".to_string());
        let program_loc = Srcloc::new(file.clone(), 1, 1);
        let operator_loc = Srcloc::new(file.clone(), 1, 2);
        let value_loc = Srcloc::new(file.clone(), 1, 5);
        let nil_loc = Srcloc::new(file.clone(), 1, 10);
        let args_loc = Srcloc::new(file, 1, 12);

        let value = Rc::new(SExp::Atom(value_loc.clone(), b"value".to_vec()));
        let nil = Rc::new(SExp::Nil(nil_loc.clone()));
        let program = Rc::new(enlist(
            program_loc,
            &[
                Rc::new(SExp::Integer(
                    operator_loc.clone(),
                    4_i32.to_bigint().unwrap(),
                )),
                quote(value_loc.clone(), value),
                quote(nil_loc.clone(), nil),
            ],
        ));
        let expected_result_loc = program.loc();

        let mut allocator = SExpClassicAllocator::new();
        let program = allocator.from_sexp(program).unwrap();
        let args = allocator.from_sexp(Rc::new(SExp::Nil(args_loc))).unwrap();
        let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
        let result = allocator.run_clvm(runner, &program, &args).unwrap();

        match result.sexp.as_ref() {
            SExp::Cons(loc, first, rest) => {
                assert_eq!(*loc, expected_result_loc);
                assert_eq!(first.loc(), value_loc);
                assert_eq!(rest.loc(), nil_loc);
            }
            result => panic!("expected cons result, got {result}"),
        }
    }
}
