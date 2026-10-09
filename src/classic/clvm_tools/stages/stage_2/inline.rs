use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm::sexp::{enlist, proper_list};
use crate::classic::clvm_tools::stages::stage_2::abstraction::{
    ASExp, BufCarrier, ClError, ClassicAllocator,
};
use crate::compiler::gensym::gensym;

use crate::util::Number;
use num_bigint::ToBigInt;
use std::collections::HashMap;

// If this is an at capture of the form
// (@ name substructure)
// then return name and substructure.
pub fn is_at_capture<A: ClassicAllocator>(
    allocator: &A,
    tree_first: &A::NodePtr,
    tree_rest: &A::NodePtr,
) -> Option<(A::NodePtr, A::NodePtr)>
where
    A::NodePtr: Clone,
{
    if let (ASExp::Atom, Some(spec)) = (
        allocator.sexp(tree_first),
        proper_list(allocator, tree_rest, true),
    ) {
        let first_atom = allocator.atom(tree_first);
        if first_atom.as_ref() == b"@" && spec.len() == 2 {
            return Some((spec[0].clone(), spec[1].clone()));
        }
    }

    None
}

// (unquote X)
fn wrap_in_unquote<A: ClassicAllocator>(
    allocator: &mut A,
    code: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(code);
    let unquote_atom = allocator.new_atom(loc, "unquote".as_bytes())?;
    enlist(allocator, &[unquote_atom, code.clone()])
}

// (__chia__enlist X)
fn wrap_in_compile_time_list<A: ClassicAllocator>(
    allocator: &mut A,
    code: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(code);
    let chia_enlist_atom = allocator.new_atom(loc, "__chia__enlist".as_bytes())?;
    enlist(allocator, &[chia_enlist_atom, code.clone()])
}

// Create the sequence of individual tree moves that will translate to
// (f ...) and (r ...) wrapping to select the given path from a larger structure.
fn create_path_selection_plan(path: Number, operators: &mut Vec<bool>) {
    if path > bi_one() {
        operators.push(path.clone() % 2_u32.to_bigint().unwrap() == bi_one());
        create_path_selection_plan(path / 2_u32.to_bigint().unwrap(), operators)
    }
}

// Given a path and code to be wrapped, generate a lookup by path into that code.
fn wrap_path_selection<A: ClassicAllocator>(
    allocator: &mut A,
    path: Number,
    wrapped: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let mut operator_stack = Vec::new();
    let mut tail = wrapped.clone();
    create_path_selection_plan(path, &mut operator_stack);
    for o in operator_stack.iter() {
        let head_op = if *o { vec![6] } else { vec![5] };
        let loc = allocator.loc(wrapped);
        let head_atom = allocator.new_atom(loc, &head_op)?;
        tail = enlist(allocator, &[head_atom, tail])?;
    }
    Ok(tail)
}

// Called for each top level argument (left branch) of the argument list of
// an inline function that does destructuring (has any substructure or non
// linearity in its argument list).
//
// If further captures are encountered, we record them in selections but
// must continue their substructure as though it belongs to the current capture
// as the classic macro system handles destructuring on the source text rather
// than the argument values, so we must eliminate all deep references past the
// top of the argument list.
fn formulate_path_selections_for_destructuring_arg<A: ClassicAllocator>(
    allocator: &mut A,
    arg_sexp: &A::NodePtr,
    arg_path: Number,
    arg_depth: Number,
    referenced_from: Option<A::NodePtr>,
    selections: &mut HashMap<Vec<u8>, A::NodePtr>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(arg_sexp);
    match allocator.sexp(arg_sexp) {
        ASExp::Pair(a, b) => {
            let next_depth = arg_depth.clone() * 2_u32.to_bigint().unwrap();
            if let Some((capture, substructure)) = is_at_capture(allocator, &a, &b) {
                if let ASExp::Atom = allocator.sexp(&capture) {
                    let (new_arg_path, new_arg_depth, tail) = if let Some(prev_ref) =
                        referenced_from
                    {
                        (arg_path, arg_depth, prev_ref)
                    } else {
                        let capture_code = wrap_in_unquote(allocator, &capture)?;
                        let qtail =
                            wrap_path_selection(allocator, arg_path + arg_depth, &capture_code)?;
                        (bi_zero(), bi_one(), qtail)
                    };

                    // Was cbuf from capture.
                    let capture_atom = allocator.atom(&capture);
                    selections.insert(capture_atom.as_ref().to_vec(), tail.clone());

                    return formulate_path_selections_for_destructuring_arg(
                        allocator,
                        &substructure,
                        new_arg_path,
                        new_arg_depth,
                        Some(tail),
                        selections,
                    )
                    .map(|_| arg_sexp.clone());
                }
            }

            if referenced_from.is_some() {
                let f = formulate_path_selections_for_destructuring_arg(
                    allocator,
                    &a,
                    arg_path.clone(),
                    next_depth.clone(),
                    referenced_from.clone(),
                    selections,
                )?;
                let r = formulate_path_selections_for_destructuring_arg(
                    allocator,
                    &b,
                    arg_depth + arg_path,
                    next_depth,
                    referenced_from,
                    selections,
                )?;
                allocator.new_pair(loc, &f, &r)
            } else {
                let ref_name = gensym("destructuring_capture".as_bytes().to_vec());
                let at_atom = allocator.new_atom(loc.clone(), "@".as_bytes())?;
                let name_atom = allocator.new_atom(loc, &ref_name)?;
                let new_arg_list = enlist(allocator, &[at_atom, name_atom, arg_sexp.clone()])?;
                formulate_path_selections_for_destructuring_arg(
                    allocator,
                    &new_arg_list,
                    bi_zero(),
                    bi_one(),
                    None,
                    selections,
                )
            }
        }
        ASExp::Atom => {
            // Note: can't co-borrow with allocator below.
            let buf_atom = allocator.atom(arg_sexp);
            let buf = buf_atom.as_ref().to_vec();
            if !buf.is_empty() {
                if let Some(capture) = referenced_from {
                    let tail = wrap_path_selection(allocator, arg_path + arg_depth, &capture)?;
                    selections.insert(buf, tail);
                    return Ok(arg_sexp.clone());
                }
            }
            Ok(arg_sexp.clone())
        }
    }
}

// These generate a new argument list that will use at-captures to identify
// roots to pick data out of in the eventual macro code that's emitted.  This
// is needed because macros and functions work differently.  While functions
// conceptually receive an environment and choose values out of it, macros
// bind parameters to the source code the user used to invoke them; therefore
// destructuring can be problematic
//
// Consider this example:
//
//   (defun-inline F ((A B C)) (+ A B C))
//
// Without supporting destructuring consciously, this will be turned by
// classic chialisp into a macro like this:
//
//   (defmacro F ((A B C)) (+ A B C))
//
// Which destructures the source text of the program:
//
//   (F (4 1 (list 2 3))) would be expected to output 6
//
// But instead, the destructuring gives:
//
//   (+ 4 1 (list 2 3))
//
// We insert a capture for any top level argument that is non-proper:
//
//   (defun-inline F ((@ destructuring_capture_$_1 (A B C))) (+ A B C))
//
// And "selections" contains the code that should be used in place of simply
// unquoting a named argument:
//
//   { "A": (f (unquote destructuring_capture_$_1)),
//     "B": (f (r (unquote destructuring_capture_$_1))
//     ...
//
// There is a unique case to deal with:
//
//   (defun-inline offset-of-pt (@ pt (X Y)) (+ X (* 8 Y)))
//
// Because pt represents the entire argument list, it will be in this form when
// unquoted:
//
//   (offset-of-pt 3 2) -> pt = (3 2)
//
// When substituted:
//
//   (offset-of-pt 3 2) -> (+ (f (3 2)) (* 8 (f (r (3 2)))))
//
// Simply quoting won't solve it, because the code may do something
//
//   (offset-of-pt (+ 1 Q) (- W 2)) -> (+ (f ((+ 1 Q) (- W 2))) ...)
//
// So we need a macro like "list" that starts not from the entire input
// environment but that destructures just its first argument as a list,
// so i adapted list into __chia__enlist.
// When so wrapped, the user may then destructure the capture argument.
pub fn formulate_path_selections_for_destructuring<A: ClassicAllocator>(
    allocator: &mut A,
    args_sexp: &A::NodePtr,
    selections: &mut HashMap<Vec<u8>, A::NodePtr>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    if let ASExp::Pair(a, b) = allocator.sexp(args_sexp) {
        if let Some((capture, substructure)) = is_at_capture(allocator, &a, &b) {
            if let ASExp::Atom = allocator.sexp(&capture) {
                let quoted_arg_list = wrap_in_unquote(allocator, &capture)?;
                let tail = wrap_in_compile_time_list(allocator, &quoted_arg_list)?;
                // Was: cbuf from capture.
                let buf_atom = allocator.atom(&capture);
                selections.insert(buf_atom.as_ref().to_vec(), tail.clone());
                let newsub = formulate_path_selections_for_destructuring_arg(
                    allocator,
                    &substructure,
                    bi_zero(),
                    bi_one(),
                    Some(tail),
                    selections,
                )?;
                return enlist(allocator, &[a, capture, newsub]);
            }
        }
        let f = formulate_path_selections_for_destructuring_arg(
            allocator,
            &a,
            bi_zero(),
            bi_one(),
            None,
            selections,
        )?;
        let r = formulate_path_selections_for_destructuring(allocator, &b, selections)?;
        let loc = allocator.loc(&b);
        allocator.new_pair(loc, &f, &r)
    } else {
        Ok(args_sexp.clone())
    }
}

// If true, these arguments represent a destructuring of some kind.
// In the case of inlines in classic chialisp, we must adjust how arguments
// are passed down to the macro body that gets created for the inline function.
pub fn is_inline_destructure<A: ClassicAllocator>(
    allocator: &mut A,
    args_sexp: &A::NodePtr,
) -> bool {
    if let ASExp::Pair(a, b) = allocator.sexp(args_sexp) {
        if let ASExp::Pair(_, _) = allocator.sexp(&a) {
            return true;
        }

        return is_inline_destructure(allocator, &b);
    }

    false
}
