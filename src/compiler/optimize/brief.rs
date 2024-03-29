use num_bigint::ToBigInt;
use std::borrow::Borrow;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::classic::clvm_tools::node_path::compose_paths;

#[cfg(test)]
use crate::compiler::sexp::parse_sexp;
use crate::compiler::sexp::SExp;

#[cfg(test)]
use crate::compiler::srcloc::Srcloc;

fn is_quote_atom(a: &SExp) -> bool {
    if let SExp::Atom(_, n) = a.atomize() {
        return n.len() == 1 && n[0] == 1;
    }

    false
}

fn is_first_atom(a: &SExp) -> bool {
    if let SExp::Atom(_, n) = a.atomize() {
        return n.len() == 1 && n[0] == 5;
    }

    false
}

fn is_rest_atom(a: &SExp) -> bool {
    if let SExp::Atom(_, n) = a.atomize() {
        return n.len() == 1 && n[0] == 6;
    }

    false
}

// Collapse sequences of (f (r (f ... X))) representing (a P1 X)
// into (a P1 X) if X is not a path
// or P1 || X    if X is a path
fn brief_path_selection_single(mut body: Rc<SExp>) -> (bool, Rc<SExp>) {
    let orig_body = body.clone();
    let mut found_stack = 0;
    let mut target_path = bi_one();

    while let Some(lst) = body.proper_list() {
        if let [cmd, arg] = &lst[..] {
            if is_quote_atom(cmd) {
                break;
            }

            let is_first = is_first_atom(cmd);
            if is_first || is_rest_atom(cmd) {
                found_stack += 1;
                target_path *= 2_u32.to_bigint().unwrap();
                if !is_first {
                    target_path += bi_one();
                }
                body = Rc::new(arg.clone());
                continue;
            }
        }

        break;
    }

    if found_stack > 0 {
        let intval = if let SExp::Integer(_l, i) = body.borrow() {
            Some(i.clone())
        } else {
            None
        };

        if let Some(i) = intval {
            // The number to the left is "closer to the root".
            // These bits are selected first.
            let final_path = compose_paths(&i, &target_path);
            return (true, Rc::new(SExp::Integer(body.loc(), final_path)));
        }

        /*
        if found_stack > 2 {
            let apply_atom = Rc::new(SExp::Atom(body.loc(), vec![2]));
            let at_atom = Rc::new(SExp::Atom(body.loc(), vec![1]));

            return (true, Rc::new(SExp::Cons(
                body.loc(),
                apply_atom.clone(),
                Rc::new(SExp::Cons(
                    body.loc(),
                    Rc::new(SExp::Cons(
                        body.loc(),
                        at_atom.clone(),
                        Rc::new(SExp::Integer(body.loc(), target_path))
                    )),
                    Rc::new(SExp::Cons(
                        body.loc(),
                        Rc::new(SExp::Cons(
                            body.loc(),
                            apply_atom,
                            Rc::new(SExp::Cons(
                                body.loc(),
                                body.clone(),
                                Rc::new(SExp::Cons(
                                    body.loc(),
                                    at_atom.clone(),
                                    Rc::new(SExp::Nil(body.loc()))
                                ))
                            ))
                        )),
                        Rc::new(SExp::Nil(body.loc()))
                    ))
                ))
            )));
        }
        */
    }

    (false, orig_body)
}

pub fn brief_path_selection(body: Rc<SExp>) -> (bool, Rc<SExp>) {
    let (changed, new_body) = brief_path_selection_single(body.clone());
    if changed {
        return (true, new_body);
    }

    if let Some(lst) = body.proper_list() {
        if lst.len() < 2 || is_quote_atom(&lst[0]) {
            return (false, body);
        }

        let mut end = Rc::new(SExp::Nil(body.loc()));
        let mut updated = false;
        for f in lst.iter().rev() {
            let (mod_a, a) = brief_path_selection(Rc::new(f.clone()));
            updated = updated || mod_a;
            end = Rc::new(SExp::Cons(body.loc(), a, end));
        }
        return (updated, end);
    }

    (false, body)
}

#[test]
fn test_brief_path_selection_none() {
    let filename = "*test*";
    let loc = Srcloc::start(filename);
    let parsed = parse_sexp(loc.clone(), "(2 (1 (1) (1) (1)) (3))".bytes()).expect("should parse");
    let (did_work, optimized) = brief_path_selection(parsed[0].clone());
    assert!(!did_work);
    assert_eq!(optimized.to_string(), "(2 (1 (1) (1) (1)) (3))");
}

/*
#[test]
fn test_brief_path_selection_one_first_unknown_body() {
    let filename = "*test*";
    let loc = Srcloc::start(filename);
    let parsed = parse_sexp(loc.clone(), "(5 (5 (5 (+ 3 2))))".bytes()).expect("should parse");
    let (did_work, optimized) = brief_path_selection(parsed[0].clone());
    assert!(did_work);
    assert_eq!(optimized.to_string(), "(2 8 (2 (+ 3 2) 1))");
}
*/

#[test]
fn test_brief_path_selection_one_first_path_body_1() {
    let filename = "*test*";
    let loc = Srcloc::start(filename);
    let parsed = parse_sexp(loc.clone(), "(5 (5 (6 (5 11))))".bytes()).expect("should parse");
    let (did_work, optimized) = brief_path_selection(parsed[0].clone());
    assert!(did_work);
    // f f r f -> path 18 (0100[1])
    // 11      ->         (110[1])
    // Joined so that 11's path occupies the inner (low) bits: 1100100[1]
    assert_eq!(optimized.to_string(), "147");
}
