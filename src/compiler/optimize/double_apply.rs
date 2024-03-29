use crate::compiler::clvm::truthy;
use crate::compiler::prims::primquote;
use crate::compiler::sexp::{AtomValue, NodeSel, SExp, SelectNode, ThisNode};
use std::borrow::Borrow;
use std::rc::Rc;

// Turn:
//
// (a (q any) 1)
//
// into
//
// any
//
// I now realize this is exactly cons_q_a_optimizer from classic :-)
pub fn change_double_to_single_apply(sexp: Rc<SExp>) -> (bool, Rc<SExp>) {
    if let Ok(NodeSel::Cons(
        _,
        NodeSel::Cons(
            NodeSel::Cons(
                // quoted program
                _,
                inner_program,
            ),
            NodeSel::Cons(_, _),
        ),
    )) = NodeSel::Cons(
        AtomValue::Here(&[2]),
        NodeSel::Cons(
            NodeSel::Cons(
                // quoted program
                AtomValue::Here(&[1]),
                ThisNode,
            ),
            NodeSel::Cons(AtomValue::Here(&[1]), ThisNode),
        ),
    )
    .select_nodes(sexp.clone())
    {
        return (true, inner_program);
    }

    (false, sexp)
}

fn change_apply_double_quote(sexp: Rc<SExp>) -> (bool, Rc<SExp>) {
    if let Ok(NodeSel::Cons(
        _, // apply
        NodeSel::Cons(
            NodeSel::Cons(
                _, // q
                NodeSel::Cons(
                    _, // 1
                    body,
                ),
            ),
            _,
        ),
    )) = NodeSel::Cons(
        AtomValue::Here(&[2]),
        NodeSel::Cons(
            NodeSel::Cons(
                AtomValue::Here(&[1]),
                NodeSel::Cons(AtomValue::Here(&[1]), ThisNode),
            ),
            ThisNode,
        ),
    )
    .select_nodes(sexp.clone())
    {
        return (true, Rc::new(primquote(body.loc(), body.clone())));
    }

    (false, sexp)
}

fn collapse_constant_condition(sexp: Rc<SExp>) -> (bool, Rc<SExp>) {
    if let Ok(NodeSel::Cons(
        _, // i
        NodeSel::Cons(cond, NodeSel::Cons(a, NodeSel::Cons(b, _))),
    )) = NodeSel::Cons(
        AtomValue::Here(&[3]),
        NodeSel::Cons(
            ThisNode,
            NodeSel::Cons(ThisNode, NodeSel::Cons(ThisNode, ThisNode)),
        ),
    )
    .select_nodes(sexp.clone())
    {
        // There are two cases here we care about:
        // Either cond is () or it's (1 . something)
        // The following filters away a non-const condition and leaves
        // the remaining as either Some(true) or Some(false), then
        // chooses a wing based on that.
        return NodeSel::Cons(AtomValue::Here(&[1]), ThisNode)
            .select_nodes(cond.clone())
            .ok()
            .map(|NodeSel::Cons(_, cond_quoted)| Some(truthy(cond_quoted)))
            .unwrap_or_else(|| if !truthy(cond) { Some(false) } else { None })
            .map(|use_cond| if use_cond { (true, a) } else { (true, b) })
            .unwrap_or_else(|| (false, sexp));
    }

    (false, sexp)
}

// Recognize some optimizations:
//
// specific: (a (q 1 . x) _) => (q . x)
// classic optimizer: (a (op SEXP) ENV) => (op (a SEXP ENV)) <- wip
// classic optimizer: (a (q SEXP) 1) => SEXP
pub fn remove_double_apply(mut sexp: Rc<SExp>, spine: bool) -> (bool, Rc<SExp>) {
    // Don't descend into quoted expressions.
    if spine {
        if let Ok(NodeSel::Cons(_, _)) =
            NodeSel::Cons(AtomValue::Here(&[1]), ThisNode).select_nodes(sexp.clone())
        {
            return (false, sexp);
        }
    }

    let mut any_transformation = true;
    let mut was_transformed = false;

    while any_transformation {
        if let SExp::Cons(l, a, b) = sexp.borrow() {
            // These transformations play on each other but finalize together.
            let (a_changed, new_a) = remove_double_apply(a.clone(), true);
            let (b_changed, new_b) = remove_double_apply(b.clone(), false);

            let result = Rc::new(SExp::Cons(l.clone(), new_a, new_b));
            if spine {
                let (root_transformed_dq, result_dq) = change_apply_double_quote(result);
                let (root_transformed_unapply, result_single_apply) =
                    change_double_to_single_apply(result_dq);

                let (constant_collapse, result_end) =
                    collapse_constant_condition(result_single_apply);

                any_transformation = a_changed
                    || b_changed
                    || root_transformed_dq
                    || root_transformed_unapply
                    || constant_collapse;
                was_transformed |= any_transformation;
                sexp = result_end;
            } else {
                any_transformation = a_changed || b_changed;
                was_transformed |= any_transformation;
                sexp = result;
            }
        } else {
            break;
        }
    }

    (was_transformed, sexp)
}
