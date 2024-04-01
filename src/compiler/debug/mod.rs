use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{sha256, Bytes, BytesFromType};

use crate::compiler::sexp::SExp;
use crate::util::u8_from_number;

pub mod armjit;

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
