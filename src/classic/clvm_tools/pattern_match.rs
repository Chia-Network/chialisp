use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm::sexp::equal_to;
use crate::classic::clvm_tools::stages::stage_2::abstraction::{ASExp, BufCarrier, ClassicAllocator};

use std::collections::HashMap;

pub const ATOM_MATCH: [u8; 1] = *b"$";
pub const SEXP_MATCH: [u8; 1] = *b":";

pub fn unify_bindings<A: ClassicAllocator>(
    allocator: &mut A,
    bindings: HashMap<String, A::NodePtr>,
    new_key: &[u8],
    new_value: &A::NodePtr,
) -> Option<HashMap<String, A::NodePtr>>
where
    A::NodePtr: Clone
{
    /*
     * Try to add a new binding to the list, rejecting it if it conflicts
     * with an existing binding.
     */
    let new_key_str = Bytes::new(Some(BytesFromType::Raw(new_key.to_vec()))).decode();
    match bindings.get(&new_key_str) {
        Some(binding) => {
            if !equal_to(allocator, binding, &new_value) {
                return None;
            }
            Some(bindings)
        }
        _ => {
            let mut new_bindings = bindings.clone();
            new_bindings.insert(new_key_str, new_value.clone());
            Some(new_bindings)
        }
    }
}

pub fn match_sexp<A: ClassicAllocator>(
    allocator: &mut A,
    pattern: &A::NodePtr,
    sexp: &A::NodePtr,
    known_bindings: HashMap<String, A::NodePtr>,
) -> Option<HashMap<String, A::NodePtr>>
where
    A::NodePtr: Clone
{
    /*
     * Determine if sexp matches the pattern, with the given known bindings already applied.
     * Returns None if no match, or a (possibly empty) dictionary of bindings if there is a match
     * Patterns look like this:
     * ($ . $) matches the literal "$", no bindings (mostly useless)
     * (: . :) matches the literal ":", no bindings (mostly useless)
     * ($ . A) matches B if B is an atom; and A is bound to B
     * (: . A) matches B always; and A is bound to B
     * (A . B) matches (C . D) if A matches C and B matches D
     *         and bindings are the unification (as long as unification is possible)
     */

    match (allocator.sexp(pattern), allocator.sexp(sexp)) {
        (ASExp::Atom, ASExp::Atom) => {
            // Two nodes in scope, both used.
            if allocator.atom(pattern) == allocator.atom(sexp) {
                Some(known_bindings)
            } else {
                None
            }
        }
        (ASExp::Pair(pleft, pright), _) => match (allocator.sexp(&pleft), allocator.sexp(&pright)) {
            (ASExp::Atom, ASExp::Atom) => {
                let left_atom = allocator.atom(&pleft);
                let right_atom = allocator.atom(&pright);

                // This is a false positive due to Allocator lifetime.
                #[allow(clippy::unnecessary_to_owned)]
                match allocator.sexp(sexp) {
                    ASExp::Atom => {
                        // Expression is ($ . $), sexp is '$', result: no capture.
                        // Avoid double borrow.
                        let sexp_atom = allocator.atom(sexp);
                        if left_atom.as_ref() == ATOM_MATCH {
                            if right_atom.as_ref() == ATOM_MATCH {
                                if sexp_atom.as_ref() == ATOM_MATCH {
                                    return Some(HashMap::new());
                                }
                                return None;
                            }

                            return unify_bindings(
                                allocator,
                                known_bindings,
                                &right_atom.as_ref().to_vec(),
                                sexp,
                            );
                        }
                        if left_atom.as_ref() == SEXP_MATCH {
                            if right_atom.as_ref() == SEXP_MATCH && sexp_atom.as_ref() == SEXP_MATCH
                            {
                                return Some(HashMap::new());
                            }

                            return unify_bindings(
                                allocator,
                                known_bindings,
                                // pat_right_bytes
                                &right_atom.as_ref().to_vec(),
                                sexp,
                            );
                        }

                        None
                    }
                    ASExp::Pair(sleft, sright) => {
                        if left_atom.as_ref() == SEXP_MATCH && right_atom.as_ref() != SEXP_MATCH {
                            return unify_bindings(
                                allocator,
                                known_bindings,
                                // pat_right_bytes
                                &right_atom.as_ref().to_vec(),
                                sexp,
                            );
                        }

                        match_sexp(allocator, &pleft, &sleft, known_bindings).and_then(
                            |new_bindings| match_sexp(allocator, &pright, &sright, new_bindings),
                        )
                    }
                }
            }
            _ => match allocator.sexp(sexp) {
                ASExp::Atom => None,
                ASExp::Pair(sleft, sright) => match_sexp(allocator, &pleft, &sleft, known_bindings)
                    .and_then(|new_bindings| match_sexp(allocator, &pright, &sright, new_bindings)),
            },
        },
        (ASExp::Atom, _) => None,
    }
}
