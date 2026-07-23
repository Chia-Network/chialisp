use clvm_rs::allocator::{Allocator, NodePtr};

use crate::classic::clvm::sexp::enlist;
use crate::classic::clvm_tools::node_path::NodePath;
use crate::classic::clvm_tools::stages::stage_2::abstraction::{ClassicAllocator, ClError};

lazy_static! {
    pub static ref QUOTE_ATOM: Vec<u8> = vec![1];
    pub static ref APPLY_ATOM: Vec<u8> = vec![2];
    pub static ref COM_ATOM: Vec<u8> = vec![b'c', b'o', b'm'];
}

pub fn quote<A: ClassicAllocator>(allocator: &mut A, sexp: &A::NodePtr) -> Result<A::NodePtr, ClError> {
    allocator
        .new_atom(allocator.loc(&sexp), &QUOTE_ATOM)
        .and_then(|q| allocator.new_pair(allocator.loc(&sexp), &q, &sexp))
}

// In original python code, the name of this function is `eval`,
// but since the name `eval` cannot be used in typescript context, change the name to `evaluate`.
pub fn evaluate<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
    args: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone
{
    let loc = allocator.loc(prog);
    let a = allocator.new_atom(loc, &APPLY_ATOM)?;
    enlist(allocator, &[a, prog.clone(), args.clone()])
}

pub fn run<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
    macro_lookup: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone
{
    /*
     * PROG => (e (com (q . PROG) (mac)) ARGS)
     *
     * The result can be evaluated with the stage_com eval
     * function.
     */
    let args = NodePath::new(None).as_path();
    let loc = allocator.loc(prog);
    let mac = quote(allocator, &macro_lookup)?;
    let com_sexp = allocator.new_atom(loc.clone(), &COM_ATOM)?;
    let arg_sexp = allocator.new_atom(loc, args.data())?;
    let to_eval = enlist(allocator, &[com_sexp, prog.clone(), mac])?;
    evaluate(allocator, &to_eval, &arg_sexp)
}

pub fn brun(allocator: &mut Allocator, prog: NodePtr, args: NodePtr) -> Result<NodePtr, ClError> {
    let quoted_prog = quote(allocator, &prog)?;
    let quoted_args = quote(allocator, &args)?;
    evaluate(allocator, &quoted_prog, &quoted_args)
}
