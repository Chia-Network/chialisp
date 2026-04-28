use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{sha256, Bytes, BytesFromType};
use crate::classic::clvm::casts::bigint_to_bytes_clvm;

use crate::compiler::sexp::{decode_string, parse_sexp, SExp};
pub use crate::compiler::srcloc::Srcloc;
use crate::util::{u8_from_number, Number};

use sha2::{Digest, Sha256};

pub mod armjit;

#[derive(Clone, Debug)]
pub enum DebugSExpValue<T> {
    Nil(Srcloc),
    Cons(Srcloc, T, T),
    Integer(Srcloc, Number),
    QuotedString(Srcloc, u8, Vec<u8>),
    Atom(Srcloc, Vec<u8>),
}

pub trait DebugSExp: Clone + Display {
    fn atom(loc: Srcloc, bytes: &[u8]) -> Self;
    fn loc(&self) -> Srcloc;
    fn atomize(&self) -> Self;
    fn to_number(&self) -> Option<Number>;
    fn proper_list(&self) -> Option<Vec<Self>>;
    fn explode(&self) -> DebugSExpValue<Self>;

    fn nilp(&self) -> bool {
        matches!(self.atom_bytes(), Some((_, bytes)) if bytes.is_empty())
    }

    fn atom_bytes(&self) -> Option<(Srcloc, Vec<u8>)> {
        match self.explode() {
            DebugSExpValue::Cons(_, _, _) => None,
            DebugSExpValue::Nil(loc) => Some((loc, Vec::new())),
            DebugSExpValue::Atom(loc, bytes) => Some((loc, bytes)),
            DebugSExpValue::QuotedString(loc, _, bytes) => Some((loc, bytes)),
            DebugSExpValue::Integer(loc, number) => {
                Some((loc, bigint_to_bytes_clvm(&number).data().clone()))
            }
        }
    }
}

impl DebugSExp for Rc<SExp> {
    fn atom(loc: Srcloc, bytes: &[u8]) -> Self {
        Rc::new(SExp::Atom(loc, bytes.to_vec()))
    }

    fn loc(&self) -> Srcloc {
        self.as_ref().loc()
    }

    fn atomize(&self) -> Self {
        Rc::new(self.as_ref().atomize())
    }

    fn to_number(&self) -> Option<Number> {
        self.as_ref().get_number().ok()
    }

    fn proper_list(&self) -> Option<Vec<Self>> {
        let mut res = Vec::new();
        let mut track = self.clone();

        loop {
            if track.nilp() {
                return Some(res);
            }

            match track.explode() {
                DebugSExpValue::Cons(_, left, right) => {
                    res.push(left);
                    track = right;
                }
                _ => return None,
            }
        }
    }

    fn explode(&self) -> DebugSExpValue<Self> {
        match self.as_ref() {
            SExp::Nil(loc) => DebugSExpValue::Nil(loc.clone()),
            SExp::Cons(loc, left, right) => {
                DebugSExpValue::Cons(loc.clone(), left.clone(), right.clone())
            }
            SExp::Integer(loc, number) => DebugSExpValue::Integer(loc.clone(), number.clone()),
            SExp::QuotedString(loc, quote, bytes) => {
                DebugSExpValue::QuotedString(loc.clone(), *quote, bytes.clone())
            }
            SExp::Atom(loc, bytes) => DebugSExpValue::Atom(loc.clone(), bytes.clone()),
        }
    }
}

pub type ConcreteSExp = Rc<SExp>;

pub fn debug_decode_string(v: &[u8]) -> String {
    decode_string(v)
}

pub fn debug_parse_sexp<I>(start: Srcloc, input: I) -> Result<Vec<Rc<SExp>>, (Srcloc, String)>
where
    I: Iterator<Item = u8>,
{
    parse_sexp(start, input)
}

pub fn debug_start_loc(file: &str) -> Srcloc {
    Srcloc::start(file)
}

pub fn debug_atom(loc: Srcloc, bytes: &[u8]) -> Rc<SExp> {
    <Rc<SExp> as DebugSExp>::atom(loc, bytes)
}

pub fn debug_sha256tree<T: DebugSExp>(sexp: T) -> Vec<u8> {
    match sexp.explode() {
        DebugSExpValue::Cons(_, left, right) => {
            let hash_left = debug_sha256tree(left);
            let hash_right = debug_sha256tree(right);
            let mut hasher = Sha256::new();
            hasher.update([2]);
            hasher.update(hash_left);
            hasher.update(hash_right);
            hasher.finalize().to_vec()
        }
        _ => {
            let (_, bytes) = sexp
                .atom_bytes()
                .expect("non-cons debug sexp should atomize");
            let mut hasher = Sha256::new();
            hasher.update([1]);
            hasher.update(bytes);
            hasher.finalize().to_vec()
        }
    }
}

pub fn debug_truthy<T: DebugSExp>(sexp: T) -> bool {
    !sexp.nilp()
}

pub fn debug_is_atom<T: DebugSExp>(sexp: T) -> Option<(Srcloc, Vec<u8>)> {
    sexp.atom_bytes()
}

pub fn debug_is_wrapped_atom<T: DebugSExp>(sexp: T) -> Option<(Srcloc, Vec<u8>)> {
    match sexp.explode() {
        DebugSExpValue::Cons(_, left, right) => {
            let (loc, atom) = match left.explode() {
                DebugSExpValue::Atom(loc, atom) => (loc, atom),
                _ => return None,
            };
            if debug_truthy(right) {
                None
            } else {
                Some((loc, atom))
            }
        }
        _ => None,
    }
}

pub fn debug_dequote<T: DebugSExp>(sexp: T) -> Option<T> {
    match sexp.explode() {
        DebugSExpValue::Cons(_, left, right) => match left.explode() {
            DebugSExpValue::Atom(_, atom) if atom == b"\x01" => Some(right),
            _ => None,
        },
        _ => None,
    }
}

fn debug_collect_by_hash<T: DebugSExp>(hash: &[u8], sexp: T, matches: &mut Vec<T>) -> Vec<u8> {
    if let DebugSExpValue::Cons(_, left, right) = sexp.explode() {
        let hash_left = debug_collect_by_hash(hash, left, matches);
        let hash_right = debug_collect_by_hash(hash, right, matches);
        let mut hasher = Sha256::new();
        hasher.update([2]);
        hasher.update(hash_left);
        hasher.update(hash_right);
        let my_hash = hasher.finalize().to_vec();
        if my_hash == hash {
            matches.push(sexp);
        }
        my_hash
    } else {
        let the_hash = debug_sha256tree(sexp.clone());
        if the_hash == hash {
            matches.push(sexp);
        }
        the_hash
    }
}

pub fn debug_find_all_by_hash<T: DebugSExp>(hash: &[u8], sexp: T) -> Vec<T> {
    let mut matches = Vec::new();
    debug_collect_by_hash(hash, sexp, &mut matches);
    matches
}

/// Given an SExp and a transformation, make a map of the transformed subtrees of
/// the given SExp in code that's indexed by treehash.  This will merge equivalent
/// subtrees but the uses to which it's put will generally work well enough.
///
/// Given how it's used downstream, there'd be no way to disambiguate anyhow.
///
/// A fuller explanation:e
///
/// This is purely syntactic so there's no environment in play here, basically
/// just about the CLVM value space and how program source code is represented in
/// CLVM values.
///
/// These are all equivalent in CLVM:
///
/// ##    "Y" Y 89 0x59
///
/// So a user writing:
///
/// ##    (list Y "Y" 89 0x59) ;; 1
///
/// Gives the compiler back a CLVM expression that could mean any of these
/// things:
///
/// ##    (c Y (c Y (c Y (c Y ()))))
/// ##    (c "Y" (c "Y" (c "Y" (c "Y" ()))))
/// ##    (c 89 (c 89 (c 89 (c 89 ()))))
/// ##    (c 0x59 (c 0x59 (c 0x59 (c 0x59 ()))))
///
/// So the compiler rehydrates this result by taking the largest matching subtrees
/// from the user's input and replacing it. The above is a pathological case for
/// this, and in general, doing something like:
///
/// ##    (if
/// ##      (some-condition X)
/// ##      (do-something-a X)
/// ##      (let ((Y (something X))) (do-something-else Y))
/// ##      )
///
/// Expands into a macro invocation for if, and comes back with 3 subtrees
/// identical to the user's input, so those whole trees return with their source
/// locations and the form of the user's input (Ys not rewritten as the number 89,
/// but as identifiers).
pub fn build_table_mut<X>(
    code_map: &mut HashMap<String, X>,
    tx: &dyn Fn(&SExp) -> X,
    code: &SExp,
) -> Bytes {
    match code {
        SExp::Cons(_l, a, b) => {
            let left = build_table_mut(code_map, tx, a.borrow());
            let right = build_table_mut(code_map, tx, b.borrow());
            let treehash = sha256(
                Bytes::new(Some(BytesFromType::Raw(vec![2])))
                    .concat(&left)
                    .concat(&right),
            );
            code_map.entry(treehash.hex()).or_insert_with(|| tx(code));
            treehash
        }
        SExp::Atom(_, a) => {
            let treehash = sha256(
                Bytes::new(Some(BytesFromType::Raw(vec![1])))
                    .concat(&Bytes::new(Some(BytesFromType::Raw(a.clone())))),
            );
            code_map.insert(treehash.hex(), tx(code));
            treehash
        }
        SExp::QuotedString(l, _, a) => {
            build_table_mut(code_map, tx, &SExp::Atom(l.clone(), a.clone()))
        }
        SExp::Integer(l, i) => build_table_mut(
            code_map,
            tx,
            &SExp::Atom(l.clone(), u8_from_number(i.clone())),
        ),
        SExp::Nil(l) => build_table_mut(code_map, tx, &SExp::Atom(l.clone(), Vec::new())),
    }
}

pub fn build_symbol_table_mut(code_map: &mut HashMap<String, String>, code: &SExp) -> Bytes {
    build_table_mut(code_map, &|sexp| sexp.loc().to_string(), code)
}

pub fn build_swap_table_mut(code_map: &mut HashMap<String, SExp>, code: &SExp) -> Bytes {
    build_table_mut(code_map, &|sexp| sexp.clone(), code)
}

fn relabel_inner_(
    code_map: &HashMap<String, SExp>,
    swap_table: &HashMap<SExp, String>,
    code: &SExp,
) -> SExp {
    swap_table
        .get(code)
        .and_then(|res| code_map.get(res))
        .cloned()
        .unwrap_or_else(|| match code {
            SExp::Cons(l, a, b) => {
                let new_a = relabel_inner_(code_map, swap_table, a.borrow());
                let new_b = relabel_inner_(code_map, swap_table, b.borrow());
                SExp::Cons(l.clone(), Rc::new(new_a), Rc::new(new_b))
            }
            _ => code.clone(),
        })
}

/// Given a map generated from preexisting code, replace value identical subtrees
/// with their rich valued equivalents.
///
/// Consider code that has run through a macro:
///
/// (defmacro M (VAR) VAR)
///
/// vs
///
/// (defmacro M (VAR) (q . 87))
///
/// As originally envisioned, chialisp macros compile to CLVM programs and consume
/// the program as CLVM code.  When the language is maximally permissive this isn't
/// inconsistent; a "W" string is the same representation as a W atom (an
/// identifier) and the number 87.  The problem is when users want the language to
/// distinguish between legal and illegal uses of identifiers, this poses a
/// problem.
///
/// In the above code, the macro produces a CLVM value.  That value has a valid
/// interpretation as the number 87, the string constant "W" or the identifier W.
/// If I make the rule that 'identifiers must be bound' under these conditions
/// then I've also made the rule that "one cannot return a number from a macro that
/// doesn't correspond coincidentally to the name of a bound variable, which
/// likely isn't expected given that the chialisp language gives the user the
/// ability to input this value in the distinct forms of integer, identifier,
/// string and such.  Therefore, the 87 here and the W in the next paragraph refer
/// to the same ambigious value in the CLVM value space.  A fix for this has been
/// held off for a while while a good long term solution was thought through, which
/// will appear in the form of macros that execute in the value space of chialisp
/// SExp (with distinctions between string, integer, identifier etc) and that
/// improvement is in process.
///
/// The raw result of either the integer 87, which doesn't give much clue as
/// to what's intended.  In one case, it *might* be true that VAR was untransformed
/// and the user intends the compiler to check whether downstream uses of W are
/// bound, in the second case, it's clear that won't be intended.
///
/// In classic chialisp, unclaimed identifiers are always treated as constant
/// numbers, but when we're being asked to make things strict, deciding which
/// to do makes things difficult.  Existing macro code assumes it can use unbound
/// words to name functions in the parent frame, among other things and they'll
/// be passed through as atom constants if not bound.
///
/// Relabel here takes a map made from the input of the macro invocation and
/// substitutes any equivalent subtree from before the application, which will
/// retain the form the user gave it.  This is fragile but works for now.
///
/// A way to do this better is planned.
pub fn relabel(code_map: &HashMap<String, SExp>, code: &SExp) -> SExp {
    let mut inv_swap_table = HashMap::new();
    build_swap_table_mut(&mut inv_swap_table, code);
    let mut swap_table = HashMap::new();
    for ent in inv_swap_table.iter() {
        swap_table.insert(ent.1.clone(), ent.0.clone());
    }
    relabel_inner_(code_map, &swap_table, code)
}

#[cfg(test)]
mod tests {
    use super::{
        debug_dequote, debug_find_all_by_hash, debug_is_wrapped_atom, debug_sha256tree, DebugSExp,
    };
    use crate::compiler::sexp::{enlist, SExp};
    use crate::compiler::srcloc::Srcloc;
    use std::rc::Rc;

    fn atom(bytes: &[u8]) -> Rc<SExp> {
        Rc::new(SExp::Atom(Srcloc::start("*test*"), bytes.to_vec()))
    }

    #[test]
    fn debug_sexp_proper_list_returns_original_nodes() {
        let first = atom(b"a");
        let second = atom(b"b");
        let list = Rc::new(enlist(
            Srcloc::start("*test*"),
            &[first.clone(), second.clone()],
        ));

        let proper = DebugSExp::proper_list(&list).expect("proper list");

        assert_eq!(proper, vec![first, second]);
    }

    #[test]
    fn debug_sexp_helpers_match_expected_clvm_shapes() {
        let quoted = Rc::new(SExp::Cons(
            Srcloc::start("*test*"),
            atom(&[1]),
            atom(b"value"),
        ));
        assert_eq!(debug_dequote(quoted.clone()), Some(atom(b"value")));

        let wrapped_atom = Rc::new(SExp::Cons(Srcloc::start("*test*"), atom(b"op"), atom(&[])));
        assert_eq!(
            debug_is_wrapped_atom(wrapped_atom),
            Some((Srcloc::start("*test*"), b"op".to_vec()))
        );
    }

    #[test]
    fn debug_treehash_search_finds_matching_subtrees() {
        let needle = atom(b"needle");
        let haystack = Rc::new(enlist(
            Srcloc::start("*test*"),
            &[atom(b"left"), needle.clone()],
        ));
        let hash = debug_sha256tree(needle.clone());

        let matches = debug_find_all_by_hash(&hash, haystack);

        assert_eq!(matches, vec![needle]);
    }
}
