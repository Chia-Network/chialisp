use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::mem::swap;
use std::rc::Rc;

use clvm_rs::error::EvalErr;
use num_bigint::ToBigInt;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::cost::Cost;
use clvm_rs::reduction::{Reduction, Response};

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm::sexp::{
    atom, enlist, equal_to, first, fold_m, map_m, non_nil, proper_list,
};
use crate::classic::clvm_tools::binutils::disassemble;
use crate::classic::clvm_tools::node_path::NodePath;
use crate::classic::clvm_tools::pattern_match::match_sexp;
use crate::classic::clvm_tools::stages::assemble;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::classic::clvm_tools::stages::stage_2::helpers::quote;
use crate::classic::clvm_tools::stages::stage_2::operators::AllocatorRefOrTreeHash;

use crate::util::{number_from_u8, u8_from_number};

#[derive(Clone)]
pub struct DoOptProg {}

const DEBUG_OPTIMIZATIONS: bool = false;
const DIAG_OPTIMIZATIONS: bool = false;

pub fn seems_constant_tail(allocator: &mut Allocator, sexp_: NodePtr) -> bool {
    let mut sexp = sexp_;

    loop {
        match allocator.sexp(sexp) {
            SExp::Pair(l, r) => {
                if !seems_constant(allocator, l) {
                    return false;
                }

                sexp = r;
            }
            SExp::Atom => {
                return sexp == NodePtr::NIL;
            }
        }
    }
}

pub fn seems_constant(allocator: &mut Allocator, sexp: NodePtr) -> bool {
    match allocator.sexp(sexp) {
        SExp::Atom => {
            return sexp == NodePtr::NIL;
        }
        SExp::Pair(operator, r) => {
            match allocator.sexp(operator) {
                SExp::Atom => {
                    // Was buf of operator.
                    let atom = allocator.atom(operator);
                    if atom.as_ref().len() == 1 && atom.as_ref()[0] == 1 {
                        return true;
                    } else if atom.as_ref().len() == 1 && atom.as_ref()[0] == 8 {
                        return false;
                    }
                }
                SExp::Pair(_, _) => {
                    if !seems_constant(allocator, operator) {
                        return false;
                    }
                }
            }

            if !seems_constant_tail(allocator, r) {
                return false;
            }
        }
    }
    true
}

pub fn constant_optimizer(
    allocator: &mut Allocator,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    _max_cost: Cost,
    runner: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    /*
     * If the expression does not depend upon @ anywhere,
     * it's a constant. So we can simply evaluate it and
     * return the quoted result.
     */
    if let SExp::Pair(first, _) = allocator.sexp(r) {
        // first relevant in scope.
        if let SExp::Atom = allocator.sexp(first) {
            let buf = allocator.atom(first);
            if buf.as_ref().len() == 1 && buf.as_ref()[0] == 1 {
                // Short circuit already quoted expression.
                return Ok(r);
            }
        }
    }

    let sc_r = seems_constant(allocator, r);
    let nn_r = non_nil(allocator, r);
    if DIAG_OPTIMIZATIONS {
        println!(
            "COPT {} SC_R {} NN_R {}",
            disassemble(allocator, r, None),
            sc_r,
            nn_r
        );
    }
    if sc_r && nn_r {
        return m! {
            res <- runner.run_program(
                allocator,
                r,
                NodePtr::NIL,
                None
            );
            let r1 = res.1;
            let _ = if DIAG_OPTIMIZATIONS {
                println!(
                    "CONSTANT_OPTIMIZER {} TO {}",
                    disassemble(allocator, r, None),
                    disassemble(allocator, r1, None)
                );
            };
            quoted <- quote(allocator, r1);
            Ok(quoted)
        };
    }

    Ok(r)
}

pub fn is_args_call(allocator: &Allocator, r: NodePtr) -> bool {
    if let SExp::Atom = allocator.sexp(r) {
        // Only r in scope.
        let buf = allocator.atom(r);
        buf.as_ref().len() == 1 && buf.as_ref()[0] == 1
    } else {
        false
    }
}

pub fn cons_q_a_optimizer_pattern(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(a (q . (: . sexp)) (: . args))").unwrap()
}

pub fn cons_q_a_optimizer(
    allocator: &mut Allocator,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    let cons_q_a_optimizer_pattern = cons_q_a_optimizer_pattern(allocator);

    /*
     * This applies the transform
     * (a (q . SEXP) @) => SEXP
     */

    let matched = match_sexp(allocator, cons_q_a_optimizer_pattern, r, HashMap::new());

    match (
        matched.as_ref().and_then(|t1| t1.get("args").copied()),
        matched.as_ref().and_then(|t1| t1.get("sexp").copied()),
    ) {
        (Some(args), Some(sexp)) => {
            if is_args_call(allocator, args) {
                Ok(sexp)
            } else {
                Ok(r)
            }
        }
        _ => Ok(r),
    }
}

fn cons_pattern(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(c (: . first) (: . rest)))").unwrap()
}

fn cons_f(allocator: &mut Allocator, args: NodePtr) -> Result<NodePtr, EvalErr> {
    m! {
        let cons_pattern = cons_pattern(allocator);
        if let Some(first) = match_sexp(allocator, cons_pattern, args, HashMap::new()).and_then(|t| t.get("first").copied()) {
            Ok(first)
        } else {
            m! {
                first_atom <- allocator.new_atom(&[5]);
                tail <- allocator.new_pair(args, NodePtr::NIL);
                allocator.new_pair(first_atom, tail)
            }
        }
    }
}

fn cons_r(allocator: &mut Allocator, args: NodePtr) -> Result<NodePtr, EvalErr> {
    m! {
        let cons_pattern = cons_pattern(allocator);
        if let Some(rest) = match_sexp(allocator, cons_pattern, args, HashMap::new()).and_then(|t| t.get("rest").copied()) {
            Ok(rest)
        } else {
            m! {
                rest_atom <- allocator.new_atom(&[6]);
                tail <- allocator.new_pair(args, NodePtr::NIL);
                allocator.new_pair(rest_atom, tail)
            }
        }
    }
}

fn path_from_args(
    allocator: &mut Allocator,
    sexp: NodePtr,
    new_args: NodePtr,
) -> Result<NodePtr, EvalErr> {
    match allocator.sexp(sexp) {
        SExp::Atom => {
            // Only sexp in scope.
            let atom = allocator.atom(sexp);
            let v = number_from_u8(atom.as_ref());
            if v <= bi_one() {
                Ok(new_args)
            } else {
                let sexp = allocator.new_atom(&u8_from_number(v.clone() >> 1).to_vec())?;
                if (v & 1_u32.to_bigint().unwrap()) != bi_zero() {
                    let cons_r_res = cons_r(allocator, new_args)?;
                    path_from_args(allocator, sexp, cons_r_res)
                } else {
                    let cons_f_res = cons_f(allocator, new_args)?;
                    path_from_args(allocator, sexp, cons_f_res)
                }
            }
        }
        _ => Ok(new_args),
    }
}

pub fn sub_args(
    allocator: &mut Allocator,
    sexp: NodePtr,
    new_args: NodePtr,
) -> Result<NodePtr, EvalErr> {
    match allocator.sexp(sexp) {
        SExp::Atom => path_from_args(allocator, sexp, new_args),
        SExp::Pair(first_pre, rest) => {
            let first;

            match allocator.sexp(first_pre) {
                SExp::Pair(_, _) => {
                    first = sub_args(allocator, first_pre, new_args)?;
                }
                SExp::Atom => {
                    // Atom is a reflection of first_pre.
                    let atom = allocator.atom(first_pre);
                    if atom.as_ref().len() == 1 && atom.as_ref()[0] == 1 {
                        return Ok(sexp);
                    } else {
                        first = first_pre;
                    }
                }
            }

            match proper_list(allocator, rest, true) {
                Some(tail_args) => m! {
                    res <- map_m(
                        allocator,
                        &mut tail_args.iter(),
                        &|allocator, elt| {
                            sub_args(allocator, *elt, new_args)
                        }
                    );
                    tail_list <- enlist(allocator, &res);
                    allocator.new_pair(first, tail_list)
                },
                None => path_from_args(allocator, sexp, new_args),
            }
        }
    }
}

fn var_change_optimizer_cons_eval_pattern(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(a (q . (: . sexp)) (: . args))").unwrap()
}

pub fn var_change_optimizer_cons_eval(
    allocator: &mut Allocator,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    /*
     * This applies the transform
     * (a (q . (op SEXP1...)) (ARGS)) => (q . RET_VAL) where ARGS != @
     * via
     * (op (a SEXP1 (ARGS)) ...) (ARGS)) and then "children_optimizer" of this.
     * In some cases, this can result in a constant in some of the children.
     *
     * If we end up needing to push the "change of variables" to only one child, keep
     * the optimization. Otherwise discard it.
     */

    let pattern = var_change_optimizer_cons_eval_pattern(allocator);
    match match_sexp(allocator, pattern, r, HashMap::new()).as_ref() {
        None => Ok(r),
        Some(t1) => {
            let original_args = t1.get("args").ok_or_else(|| {
                EvalErr::InternalError(r, "bad pattern match on args".to_string())
            })?;

            if DIAG_OPTIMIZATIONS {
                println!(
                    "XXX ORIGINAL_ARGS {}",
                    disassemble(allocator, *original_args, None)
                );
            };
            let original_call = t1.get("sexp").ok_or_else(|| {
                EvalErr::InternalError(r, "bad pattern match on sexp".to_string())
            })?;

            if DIAG_OPTIMIZATIONS {
                println!(
                    "XXX ORIGINAL_CALL {}",
                    disassemble(allocator, *original_call, None)
                );
            };

            let new_eval_sexp_args = sub_args(allocator, *original_call, *original_args)?;

            if DIAG_OPTIMIZATIONS {
                println!(
                    "XXX new_eval_sexp_args {} ORIG {}",
                    disassemble(allocator, new_eval_sexp_args, None),
                    disassemble(allocator, *original_args, None)
                );
            };

            // Do not iterate into a quoted value as if it were a list
            if seems_constant(allocator, new_eval_sexp_args) {
                if DIAG_OPTIMIZATIONS {
                    println!("XXX seems_constant");
                }
                optimize_sexp_(allocator, memo, new_eval_sexp_args, eval_f)
            } else {
                if DIAG_OPTIMIZATIONS {
                    println!("XXX does not seems_constant");
                };

                proper_list(allocator, new_eval_sexp_args, true)
                    .map(|new_operands| {
                        let mut opt_operands = Vec::new();
                        for item in new_operands.iter() {
                            opt_operands.push(optimize_sexp_(
                                allocator,
                                memo,
                                *item,
                                eval_f.clone(),
                            )?);
                        }

                        let non_constant_count = fold_m(
                            allocator,
                            &|allocator, acc, val| {
                                if DIAG_OPTIMIZATIONS {
                                    println!(
                                        "XXX opt_operands {} {}",
                                        acc,
                                        disassemble(allocator, val, None)
                                    );
                                }
                                let increment = match allocator.sexp(val) {
                                    SExp::Pair(val_first, _) => match allocator.sexp(val_first) {
                                        SExp::Atom => {
                                            // Atom reflects val_first.
                                            let vf_buf = allocator.atom(val_first);
                                            (vf_buf.as_ref().len() != 1 || vf_buf.as_ref()[0] != 1)
                                                as i32
                                        }
                                        _ => 0,
                                    },
                                    _ => 0,
                                };

                                Ok::<_, EvalErr>(acc + increment)
                            },
                            0,
                            &mut opt_operands.iter().copied(),
                        )?;

                        if DIAG_OPTIMIZATIONS {
                            println!("XXX non_constant_count {non_constant_count}");
                        };

                        if non_constant_count < 1 {
                            enlist(allocator, &opt_operands)
                        } else {
                            Ok(r)
                        }
                    })
                    .unwrap_or(Ok(r))
            }
        }
    }
}

pub fn children_optimizer(
    allocator: &mut Allocator,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    // Recursively apply optimizations to all non-quoted child nodes.
    match proper_list(allocator, r, true) {
        None => Ok(r),
        Some(list) => {
            if list.is_empty() {
                return Ok(r);
            }
            if let SExp::Atom = allocator.sexp(list[0]) {
                let atom = allocator.atom(list[0]);
                if atom.as_ref().to_vec() == vec![1] {
                    return Ok(r);
                }
            }

            let mut optimized = Vec::new();
            let mut different = false;
            for item in list.iter() {
                let res = optimize_sexp_(allocator, memo, *item, eval_f.clone())?;
                if different || !equal_to(allocator, *item, res) {
                    different = true;
                }
                optimized.push(res);
            }

            if different {
                enlist(allocator, &optimized)
            } else {
                // If we didn't produce any different children, skip producing
                // a new list and return r.  Take advantage of using a consistent
                // allocator to help the cache.
                Ok(r)
            }
        }
    }
}

fn cons_optimizer_pattern_first(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(f (c (: . first) (: . rest)))").unwrap()
}

fn cons_optimizer_pattern_rest(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(r (c (: . first) (: . rest)))").unwrap()
}

fn cons_optimizer(
    allocator: &mut Allocator,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    /*
     * This applies the transform
     *  (f (c A B)) => A
     *  and
     *  (r (c A B)) => B
     */
    let cons_optimizer_pattern_first = cons_optimizer_pattern_first(allocator);
    let cons_optimizer_pattern_rest = cons_optimizer_pattern_rest(allocator);

    m! {
        let t1 = match_sexp(
            allocator, cons_optimizer_pattern_first, r, HashMap::new()
        );
        match t1.and_then(|t| t.get("first").copied()) {
            Some(first) => Ok(first),
            _ => {
                m! {
                    let t2 = match_sexp(
                        allocator, cons_optimizer_pattern_rest, r, HashMap::new()
                    );
                    match t2.and_then(|t| t.get("rest").copied()) {
                        Some(rest) => Ok(rest),
                        _ => Ok(r)
                    }
                }
            }
        }
    }
}

fn first_atom_pattern(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(f ($ . atom))").unwrap()
}

fn rest_atom_pattern(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(r ($ . atom))").unwrap()
}

fn path_optimizer(
    allocator: &mut Allocator,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    let first_atom_pattern = first_atom_pattern(allocator);
    let rest_atom_pattern = rest_atom_pattern(allocator);

    /*
     * This applies the transform
     *   (f N) => A
     * and
     *   (r N) => B
     */

    let first_match = match_sexp(allocator, first_atom_pattern, r, HashMap::new());
    let rest_match = match_sexp(allocator, rest_atom_pattern, r, HashMap::new());

    match (first_match, rest_match) {
        (Some(first), _) => {
            match first
                .get("atom")
                .and_then(|a| atom(allocator, *a).ok())
                .map(|atom| number_from_u8(&atom))
            {
                Some(atom) => {
                    let node = NodePath::new(Some(atom)).add(NodePath::new(None).first());
                    allocator.new_atom(node.as_path().data())
                }
                _ => Ok(r),
            }
        }
        (_, Some(rest)) => {
            match rest
                .get("atom")
                .and_then(|a| atom(allocator, *a).ok())
                .map(|atom| number_from_u8(&atom))
            {
                Some(atom) => {
                    let node = NodePath::new(Some(atom)).add(NodePath::new(None).rest());
                    allocator.new_atom(node.as_path().data())
                }
                _ => Ok(r),
            }
        }
        _ => Ok(r),
    }
}

fn quote_pattern_1(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(q . 0)").unwrap()
}

fn quote_null_optimizer(
    allocator: &mut Allocator,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    let quote_pattern_1 = quote_pattern_1(allocator);

    // This applies the transform `(q . 0)` => `0`
    let t1 = match_sexp(allocator, quote_pattern_1, r, HashMap::new());
    Ok(t1.map(|_| NodePtr::NIL).unwrap_or_else(|| r))
}

fn apply_null_pattern_1(allocator: &mut Allocator) -> NodePtr {
    assemble(allocator, "(a 0 . (: . rest))").unwrap()
}

fn apply_null_optimizer(
    allocator: &mut Allocator,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    let apply_null_pattern_1 = apply_null_pattern_1(allocator);

    // This applies the transform `(a 0 ARGS)` => `0`
    let t1 = match_sexp(allocator, apply_null_pattern_1, r, HashMap::new());
    Ok(t1.map(|_| NodePtr::NIL).unwrap_or_else(|| r))
}

struct OptimizerRunner<'a> {
    pub name: String,
    #[allow(clippy::type_complexity)]
    to_run: &'a dyn Fn(
        &mut Allocator,
        &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
        NodePtr,
        Rc<dyn TRunProgram>,
    ) -> Result<NodePtr, EvalErr>,
}

impl<'a> OptimizerRunner<'a> {
    pub fn invoke(
        &self,
        allocator: &mut Allocator,
        memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
        r: NodePtr,
        eval_f: Rc<dyn TRunProgram>,
    ) -> Result<NodePtr, EvalErr> {
        (self.to_run)(allocator, memo, r, eval_f)
    }

    #[allow(clippy::type_complexity)]
    pub fn new(
        name: &str,
        to_run: &'a dyn Fn(
            &mut Allocator,
            &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
            NodePtr,
            Rc<dyn TRunProgram>,
        ) -> Result<NodePtr, EvalErr>,
    ) -> Self {
        OptimizerRunner {
            name: name.to_string(),
            to_run,
        }
    }
}

pub fn optimize_sexp_(
    allocator: &mut Allocator,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r_: NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    // First compare the NodePtr to see if we've cached this exact one.
    // Note that this scoping is here to prevent the borrowed mutable ref from
    // preventing us from using memo downstream when we've done one optimize
    // pass and need to cache the result.
    {
        let memo_ref: Ref<HashMap<AllocatorRefOrTreeHash, NodePtr>> = memo.borrow();
        let memo: &HashMap<AllocatorRefOrTreeHash, NodePtr> = &memo_ref;
        if let Some(res) = memo.get(&AllocatorRefOrTreeHash::new_from_nodeptr(r_)) {
            return Ok(*res);
        }
    }

    // Fall back to treehash comparison since we didn't get an exact pointer hit.
    let footprint = AllocatorRefOrTreeHash::new_from_sexp(allocator, r_);
    {
        let memo_ref: Ref<HashMap<AllocatorRefOrTreeHash, NodePtr>> = memo.borrow();
        let memo: &HashMap<AllocatorRefOrTreeHash, NodePtr> = &memo_ref;
        if let Some(res) = memo.get(&footprint) {
            return Ok(*res);
        }
    }

    /*
     * Optimize an s-expression R written for clvm to R_opt where
     * (a R args) == (a R_opt args) for ANY args.
     */
    let optimizers: Vec<OptimizerRunner> = vec![
        OptimizerRunner::new("cons_optimizer", &cons_optimizer),
        OptimizerRunner::new("constant_optimizer", &|allocator, memo, r, eval_f| {
            constant_optimizer(allocator, memo, r, 0, eval_f.clone())
        }),
        OptimizerRunner::new("cons_q_a_optimizer", &cons_q_a_optimizer),
        OptimizerRunner::new(
            "var_change_optimizer_cons_eval",
            &var_change_optimizer_cons_eval,
        ),
        OptimizerRunner::new("children_optimizer", &children_optimizer),
        OptimizerRunner::new("path_optimizer", &path_optimizer),
        OptimizerRunner::new("quote_null_optimizer", &quote_null_optimizer),
        OptimizerRunner::new("apply_null_optimizer", &apply_null_optimizer),
    ];

    let mut r = r_;

    loop {
        let start_r = r;
        let mut name = "".to_string();

        match allocator.sexp(r) {
            SExp::Atom => {
                return Ok(r);
            }
            SExp::Pair(_, _) => {
                for opt in optimizers.iter() {
                    name.clone_from(&opt.name);
                    match opt.invoke(allocator, memo, r, eval_f.clone()) {
                        Err(e) => {
                            return Err(e);
                        }
                        Ok(res) => {
                            if !equal_to(allocator, r, res) {
                                r = res;
                                break;
                            }
                        }
                    }
                }

                if equal_to(allocator, start_r, r) {
                    memo.replace_with(|mr| {
                        let mut work = HashMap::new();
                        swap(&mut work, mr);
                        work.insert(footprint.clone(), start_r);
                        work.insert(AllocatorRefOrTreeHash::new_from_nodeptr(r), start_r);
                        work
                    });

                    return Ok(start_r);
                }

                if DEBUG_OPTIMIZATIONS {
                    println!(
                        "OPT-{:?}[{}] => {}",
                        name,
                        disassemble(allocator, start_r, None),
                        disassemble(allocator, r, None)
                    );
                }
            }
        }
    }
}

pub fn optimize_sexp(
    allocator: &mut Allocator,
    r: NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    let optimized = RefCell::new(HashMap::new());

    if DIAG_OPTIMIZATIONS {
        println!("START OPTIMIZE {}", disassemble(allocator, r, None));
    }
    optimize_sexp_(allocator, &optimized, r, eval_f).inspect(|x| {
        if DIAG_OPTIMIZATIONS {
            println!(
                "OPTIMIZE_SEXP {} GIVING {}",
                disassemble(allocator, r, None),
                disassemble(allocator, *x, None)
            );
        }
    })
}

pub fn do_optimize(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut Allocator,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: NodePtr,
) -> Response {
    let r_first = first(allocator, r)?;
    optimize_sexp_(allocator, memo, r_first, runner.clone())
        .map(|optimized| Reduction(1, optimized))
}
