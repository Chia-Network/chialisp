use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::{sha256, Bytes, BytesFromType};

use crate::compiler::clvm::integer_atom_bytes;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;

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
        SExp::Integer(l, i) => {
            build_table_mut(code_map, tx, &SExp::Atom(l.clone(), integer_atom_bytes(i)))
        }
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

/// Retain the macro evaluator's input origins independently of the value-based
/// restoration of Chialisp atom types. Equal input values can have different
/// locations; relabeling by hash alone would assign them the same origin.
/// Syntax created by the macro has no input origin, so use the invocation.
pub(crate) fn restore_macro_locations(
    args: &SExp,
    result: &SExp,
    relabeled: &SExp,
    invocation: &Srcloc,
) -> SExp {
    type Location = (Rc<String>, usize, usize, Option<(usize, usize)>);

    fn key(loc: &Srcloc) -> Location {
        (
            loc.file.clone(),
            loc.line,
            loc.col,
            loc.until.as_ref().map(|end| (end.line, end.col)),
        )
    }

    fn collect(node: &SExp, locations: &mut HashSet<Location>) {
        locations.insert(key(&node.loc()));
        if let SExp::Cons(_, left, right) = node {
            collect(left, locations);
            collect(right, locations);
        }
    }

    fn restore(
        result: &SExp,
        relabeled: &SExp,
        locations: &HashSet<Location>,
        invocation: &Srcloc,
    ) -> SExp {
        let origin = result.loc();
        let loc = if locations.contains(&key(&origin)) {
            origin
        } else {
            invocation.clone()
        };
        match (result, relabeled) {
            (SExp::Cons(_, left, right), SExp::Cons(_, typed_left, typed_right)) => SExp::Cons(
                loc,
                Rc::new(restore(left, typed_left, locations, invocation)),
                Rc::new(restore(right, typed_right, locations, invocation)),
            ),
            _ => relabeled.with_loc(loc),
        }
    }

    let mut locations = HashSet::new();
    collect(args, &mut locations);
    restore(result, relabeled, &locations, invocation)
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use clvm_rs::chia_dialect::{ChiaDialect, ClvmFlags};
    use clvm_rs::serde::{node_from_bytes, node_to_bytes};
    use clvm_rs::{run_program_with_diagnostics, Allocator};

    use crate::classic::clvm_tools::binutils::assemble;
    use crate::compiler::compiler::DefaultCompilerOpts;
    use crate::compiler::debug_metadata::{
        compile_with_debug, DebugMetadataCollection, SerializedFrame, StackFrameStyle,
    };

    fn macro_trace(source: &str, environment: &str) -> (Vec<u8>, String) {
        let artifact = compile_with_debug(
            Rc::new(DefaultCompilerOpts::new("macro-origin.clsp")),
            source,
        )
        .expect("compile macro fixture")
        .remove(0);
        let mut metadata = DebugMetadataCollection::default();
        metadata.insert(&artifact.metadata).expect("load sidecar");
        let mut allocator = Allocator::new();
        let program = node_from_bytes(&mut allocator, &artifact.program).unwrap();
        let environment = assemble(&mut allocator, environment).unwrap();
        let failure = run_program_with_diagnostics(
            &mut allocator,
            &ChiaDialect::new(ClvmFlags::empty()),
            program,
            environment,
            1_000_000,
            64,
        )
        .expect_err("macro fixture must fail");
        let frames = failure
            .frames
            .iter()
            .map(|frame| SerializedFrame {
                program: node_to_bytes(&allocator, frame.program).unwrap(),
                environment: node_to_bytes(&allocator, frame.environment).unwrap(),
            })
            .collect::<Vec<_>>();
        let rendered = metadata
            .format_captured_stack(&frames, failure.truncated, StackFrameStyle::Python)
            .unwrap();
        (artifact.program, rendered)
    }

    #[test]
    fn macro_trace_uses_invocation_and_argument_locations() {
        let source = indoc! {"
            (mod (X)
              (include *standard-cl-23*)
              (defun check (A)
                (if (> A 0)
                  (* A A)
                  (x)))
              (check X))
        "};
        let (program, rendered) = macro_trace(source, "(-1)");
        // Recorded before source attribution changed: debug origins must not
        // change the compiled program or the deployed puzzle's identity.
        let mut allocator = Allocator::new();
        let expected = assemble(
            &mut allocator,
            "(2 (1 2 2 (4 2 (4 5 ()))) (4 (1 2 (1 2 (3 (21 5 (1)) (1 2 (1 18 5 5) 1) (1 2 (1 8) 1)) 1) 1) 1))",
        )
        .unwrap();
        assert_eq!(program, node_to_bytes(&allocator, expected).unwrap());
        assert!(
            rendered.contains("line 4, column 6, in check"),
            "{rendered}"
        );
        assert!(rendered.contains("line 6, column 8"), "{rendered}");
        assert!(!rendered.contains("*macros*"), "{rendered}");
    }

    #[test]
    fn macro_trace_distinguishes_equal_input_occurrences() {
        let source = indoc! {"
            (mod ()
              (include *standard-cl-23*)
              (defmacro second (A B) B)
              (second
                (x)
                (x)))
        "};
        let (_, rendered) = macro_trace(source, "()");
        assert!(rendered.contains("line 6, column 6"), "{rendered}");
        assert!(!rendered.contains("line 5,"), "{rendered}");
    }

    #[test]
    fn macro_trace_generated_nested_forms_use_outer_invocation() {
        let (expected_program, _) =
            macro_trace("(mod (X) (include *standard-cl-23*) (x X))", "(7)");
        for kind in ["defmacro", "defmac"] {
            let source = format!(
                "(mod (X)\n  (include *standard-cl-23*)\n  ({kind} inner (A) (qq (x (unquote A))))\n  ({kind} outer (A) (qq (inner (unquote A))))\n  (outer X))\n"
            );
            let (program, rendered) = macro_trace(&source, "(7)");
            assert_eq!(program, expected_program, "{kind}");
            assert!(rendered.contains("line 5,"), "{kind}: {rendered}");
            assert!(!rendered.contains("line 3,"), "{kind}: {rendered}");
            assert!(!rendered.contains("line 4,"), "{kind}: {rendered}");
            assert!(!rendered.contains("*macros*"), "{kind}: {rendered}");
        }
    }

    #[test]
    fn macro_trace_generated_value_does_not_inherit_equal_input_location() {
        let source = indoc! {"
            (mod ()
              (include *standard-cl-23*)
              (defmacro generated (A) (q . (x)))
              (generated
                (x)))
        "};
        let (_, rendered) = macro_trace(source, "()");
        assert!(rendered.contains("line 4, column 4"), "{rendered}");
        assert!(!rendered.contains("line 5,"), "{rendered}");
    }
}
