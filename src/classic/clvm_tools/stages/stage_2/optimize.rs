use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::mem::swap;
use std::rc::Rc;

use clvm_rs::error::EvalErr;
use num_bigint::ToBigInt;

use clvm_rs::allocator::NodePtr;
use clvm_rs::cost::Cost;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm::sexp::{enlist, equal_to, first, fold_m, map_m, proper_list};
use crate::classic::clvm_tools::node_path::NodePath;
use crate::classic::clvm_tools::pattern_match::match_sexp;
use crate::classic::clvm_tools::stages::assemble;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::classic::clvm_tools::stages::stage_2::abstraction::{
    ASExp, BufCarrier, ClError, ClassicAllocator,
};
use crate::classic::clvm_tools::stages::stage_2::helpers::quote;
use crate::classic::clvm_tools::stages::stage_2::operators::AllocatorRefOrTreeHash;
use crate::compiler::srcloc::Srcloc;

use crate::util::{number_from_u8, u8_from_number};

#[derive(Clone)]
pub struct DoOptProg {}

const DEBUG_OPTIMIZATIONS: bool = false;
const DIAG_OPTIMIZATIONS: bool = false;

fn seems_constant_tail<A: ClassicAllocator>(allocator: &mut A, sexp_: &A::NodePtr) -> bool
where
    A::NodePtr: Clone,
{
    let mut sexp = sexp_.clone();

    loop {
        match allocator.sexp(&sexp) {
            ASExp::Pair(l, r) => {
                if !seems_constant(allocator, &l) {
                    return false;
                }

                sexp = r;
            }
            ASExp::Atom => {
                return allocator.is_nil(&sexp);
            }
        }
    }
}

pub fn seems_constant<A: ClassicAllocator>(allocator: &mut A, sexp: &A::NodePtr) -> bool
where
    A::NodePtr: Clone,
{
    match allocator.sexp(sexp) {
        ASExp::Atom => {
            return allocator.is_nil(sexp);
        }
        ASExp::Pair(operator, r) => {
            match allocator.sexp(&operator) {
                ASExp::Atom => {
                    // Was buf of operator.
                    let atom = allocator.atom(&operator);
                    if atom.as_ref().len() == 1 && atom.as_ref()[0] == 1 {
                        return true;
                    } else if atom.as_ref().len() == 1 && atom.as_ref()[0] == 8 {
                        return false;
                    }
                }
                ASExp::Pair(_, _) => {
                    if !seems_constant(allocator, &operator) {
                        return false;
                    }
                }
            }

            if !seems_constant_tail(allocator, &r) {
                return false;
            }
        }
    }
    true
}

pub fn constant_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    _max_cost: Cost,
    runner: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    /*
     * If the expression does not depend upon @ anywhere,
     * it's a constant. So we can simply evaluate it and
     * return the quoted result.
     */
    if let ASExp::Pair(first, _) = allocator.sexp(r) {
        // first relevant in scope.
        if let ASExp::Atom = allocator.sexp(&first) {
            let buf = allocator.atom(&first);
            if buf.as_ref().len() == 1 && buf.as_ref()[0] == 1 {
                // Short circuit already quoted expression.
                return Ok(r.clone());
            }
        }
    }

    let sc_r = seems_constant(allocator, r);
    let nn_r = !allocator.is_nil(r);
    if DIAG_OPTIMIZATIONS {
        println!(
            "COPT SC_R {} NN_R {} {}",
            sc_r,
            nn_r,
            allocator.disassemble(r, None),
        );
    }
    if sc_r && nn_r {
        let r_export = allocator.export(r);
        let res = runner
            .run_program(allocator.allocator(), r_export, NodePtr::NIL, None)
            .map_err(|e| ClError(allocator.loc(r), e))?;
        let r1 = allocator.import(allocator.loc(r), res.1)?;
        if DIAG_OPTIMIZATIONS {
            println!(
                "CONSTANT_OPTIMIZER {} TO {}",
                allocator.disassemble(r, None),
                allocator.disassemble(&r1, None)
            );
        };
        return quote(allocator, &r1);
    }

    Ok(r.clone())
}

pub fn is_args_call<A: ClassicAllocator>(allocator: &A, r: &A::NodePtr) -> bool {
    if let ASExp::Atom = allocator.sexp(r) {
        // Only r in scope.
        let buf = allocator.atom(r);
        buf.as_ref().len() == 1 && buf.as_ref()[0] == 1
    } else {
        false
    }
}

pub fn cons_q_a_optimizer_pattern<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let assembled = assemble(allocator.allocator(), "(a (q . (: . sexp)) (: . args))").unwrap();
    allocator
        .import(Srcloc::start("*cons_q_a_optimizer_pattern*"), assembled)
        .unwrap()
}

pub fn cons_q_a_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let cons_q_a_optimizer_pattern = cons_q_a_optimizer_pattern(allocator);

    /*
     * This applies the transform
     * (a (q . SEXP) @) => SEXP
     */

    let matched = match_sexp(allocator, &cons_q_a_optimizer_pattern, r, HashMap::new());

    match (
        matched.as_ref().and_then(|t1| t1.get("args").cloned()),
        matched.as_ref().and_then(|t1| t1.get("sexp").cloned()),
    ) {
        (Some(args), Some(sexp)) => {
            if is_args_call(allocator, &args) {
                Ok(sexp)
            } else {
                Ok(r.clone())
            }
        }
        _ => Ok(r.clone()),
    }
}

fn cons_pattern<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let assembled = assemble(allocator.allocator(), "(c (: . first) (: . rest)))").unwrap();
    allocator
        .import(Srcloc::start("*cons_pattern*"), assembled)
        .unwrap()
}

fn cons_f<A: ClassicAllocator>(allocator: &mut A, args: &A::NodePtr) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let cons_pattern = cons_pattern(allocator);
    let pair_loc = allocator.loc(args);
    if let Some(first) = match_sexp(allocator, &cons_pattern, args, HashMap::new())
        .and_then(|t| t.get("first").cloned())
    {
        Ok(first)
    } else {
        let first_loc = allocator.loc(args);
        let first_atom = allocator.new_atom(first_loc.clone(), &[5])?;
        let nil = allocator.import(first_loc, NodePtr::NIL)?;
        let tail = allocator.new_pair(pair_loc.clone(), args, &nil)?;
        allocator.new_pair(pair_loc, &first_atom, &tail)
    }
}

fn cons_r<A: ClassicAllocator>(allocator: &mut A, args: &A::NodePtr) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let cons_pattern = cons_pattern(allocator);
    let pair_loc = allocator.loc(args);
    if let Some(rest) = match_sexp(allocator, &cons_pattern, args, HashMap::new())
        .and_then(|t| t.get("rest").cloned())
    {
        Ok(rest)
    } else {
        let rest_loc = allocator.loc(args);
        let rest_atom = allocator.new_atom(rest_loc.clone(), &[6])?;
        let nil = allocator.import(rest_loc, NodePtr::NIL)?;
        let tail = allocator.new_pair(pair_loc.clone(), args, &nil)?;
        allocator.new_pair(pair_loc, &rest_atom, &tail)
    }
}

fn path_from_args<A: ClassicAllocator>(
    allocator: &mut A,
    sexp: &A::NodePtr,
    new_args: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    match allocator.sexp(sexp) {
        ASExp::Atom => {
            // Only sexp in scope.
            let atom = allocator.atom(sexp);
            let v = number_from_u8(atom.as_ref());
            if v <= bi_one() {
                Ok(new_args.clone())
            } else {
                let loc = allocator.loc(sexp);
                let sexp = allocator.new_atom(loc, &u8_from_number(v.clone() >> 1).to_vec())?;
                if (v & 1_u32.to_bigint().unwrap()) != bi_zero() {
                    let cons_r_res = cons_r(allocator, new_args)?;
                    path_from_args(allocator, &sexp, &cons_r_res)
                } else {
                    let cons_f_res = cons_f(allocator, new_args)?;
                    path_from_args(allocator, &sexp, &cons_f_res)
                }
            }
        }
        _ => Ok(new_args.clone()),
    }
}

pub fn sub_args<A: ClassicAllocator>(
    allocator: &mut A,
    sexp: &A::NodePtr,
    new_args: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    match allocator.sexp(sexp) {
        ASExp::Atom => path_from_args(allocator, sexp, new_args),
        ASExp::Pair(first_pre, rest) => {
            let first;

            match allocator.sexp(&first_pre) {
                ASExp::Pair(_, _) => {
                    first = sub_args(allocator, &first_pre, new_args)?;
                }
                ASExp::Atom => {
                    // Atom is a reflection of first_pre.
                    let atom = allocator.atom(&first_pre);
                    if atom.as_ref().len() == 1 && atom.as_ref()[0] == 1 {
                        return Ok(sexp.clone());
                    } else {
                        first = first_pre;
                    }
                }
            }

            match proper_list(allocator, &rest, true) {
                Some(tail_args) => {
                    let res = map_m(allocator, &mut tail_args.iter(), &|allocator, elt| {
                        sub_args(allocator, elt, new_args)
                    })?;
                    let tail_list = enlist(allocator, &res)?;
                    let first_loc = allocator.loc(&first);
                    allocator.new_pair(first_loc, &first, &tail_list)
                }
                None => path_from_args(allocator, sexp, new_args),
            }
        }
    }
}

fn var_change_optimizer_cons_eval_pattern<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(a (q . (: . sexp)) (: . args))").unwrap();
    allocator
        .import(Srcloc::start("*var_change_optimizer_cons_eval_pattern*"), a)
        .unwrap()
}

pub fn var_change_optimizer_cons_eval<A: ClassicAllocator>(
    allocator: &mut A,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
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
    let export_r = allocator.export(r);
    match match_sexp(allocator, &pattern, r, HashMap::new()).as_ref() {
        None => Ok(r.clone()),
        Some(t1) => {
            let original_args = t1.get("args").ok_or_else(|| {
                ClError(
                    allocator.loc(r),
                    EvalErr::InternalError(export_r, "bad pattern match on args".to_string()),
                )
            })?;

            if DIAG_OPTIMIZATIONS {
                println!(
                    "XXX ORIGINAL_ARGS {}",
                    allocator.disassemble(original_args, None)
                );
            };
            let original_call = t1.get("sexp").ok_or_else(|| {
                ClError(
                    allocator.loc(r),
                    EvalErr::InternalError(export_r, "bad pattern match on sexp".to_string()),
                )
            })?;

            if DIAG_OPTIMIZATIONS {
                println!(
                    "XXX ORIGINAL_CALL {}",
                    allocator.disassemble(original_call, None)
                );
            };

            let new_eval_sexp_args = sub_args(allocator, original_call, original_args)?;

            if DIAG_OPTIMIZATIONS {
                println!(
                    "XXX new_eval_sexp_args {} ORIG {}",
                    allocator.disassemble(&new_eval_sexp_args, None),
                    allocator.disassemble(original_args, None)
                );
            };

            // Do not iterate into a quoted value as if it were a list
            if seems_constant(allocator, &new_eval_sexp_args) {
                if DIAG_OPTIMIZATIONS {
                    println!("XXX seems_constant");
                }
                optimize_sexp_(allocator, memo, &new_eval_sexp_args, eval_f)
            } else {
                if DIAG_OPTIMIZATIONS {
                    println!("XXX does not seems_constant");
                };

                proper_list(allocator, &new_eval_sexp_args, true)
                    .map(|new_operands| {
                        let mut opt_operands = Vec::new();
                        for item in new_operands.iter() {
                            opt_operands.push(optimize_sexp_(
                                allocator,
                                memo,
                                item,
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
                                        allocator.disassemble(&val, None)
                                    );
                                }
                                let increment = match allocator.sexp(&val) {
                                    ASExp::Pair(val_first, _) => match allocator.sexp(&val_first) {
                                        ASExp::Atom => {
                                            // Atom reflects val_first.
                                            let vf_buf = allocator.atom(&val_first);
                                            (vf_buf.as_ref().len() != 1 || vf_buf.as_ref()[0] != 1)
                                                as i32
                                        }
                                        _ => 0,
                                    },
                                    _ => 0,
                                };

                                Ok::<_, ClError>(acc + increment)
                            },
                            0,
                            &mut opt_operands.iter().cloned(),
                        )?;

                        if DIAG_OPTIMIZATIONS {
                            println!("XXX non_constant_count {non_constant_count}");
                        };

                        if non_constant_count < 1 {
                            enlist(allocator, &opt_operands)
                        } else {
                            Ok(r.clone())
                        }
                    })
                    .unwrap_or(Ok(r.clone()))
            }
        }
    }
}

pub fn children_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    // Recursively apply optimizations to all non-quoted child nodes.
    match proper_list(allocator, r, true) {
        None => Ok(r.clone()),
        Some(list) => {
            if list.is_empty() {
                return Ok(r.clone());
            }
            if let ASExp::Atom = allocator.sexp(&list[0]) {
                let atom = allocator.atom(&list[0]);
                if atom.as_ref().to_vec() == vec![1] {
                    return Ok(r.clone());
                }
            }

            let mut optimized = Vec::new();
            let mut different = false;
            for item in list.iter() {
                let res = optimize_sexp_(allocator, memo, item, eval_f.clone())?;
                if different || !equal_to(allocator, item, &res) {
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
                Ok(r.clone())
            }
        }
    }
}

fn cons_optimizer_pattern_first<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(f (c (: . first) (: . rest)))").unwrap();
    allocator
        .import(Srcloc::start("*cons_optimizer_pattern_first*"), a)
        .unwrap()
}

fn cons_optimizer_pattern_rest<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(r (c (: . first) (: . rest)))").unwrap();
    allocator
        .import(Srcloc::start("*cons_optimizer_pattern_rest*"), a)
        .unwrap()
}

fn cons_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
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
            allocator, &cons_optimizer_pattern_first, r, HashMap::new()
        );
        match t1.and_then(|t| t.get("first").cloned()) {
            Some(first) => Ok(first),
            _ => {
                m! {
                    let t2 = match_sexp(
                        allocator, &cons_optimizer_pattern_rest, r, HashMap::new()
                    );
                    match t2.and_then(|t| t.get("rest").cloned()) {
                        Some(rest) => Ok(rest),
                        _ => Ok(r.clone())
                    }
                }
            }
        }
    }
}

fn first_atom_pattern<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(f ($ . atom))").unwrap();
    allocator
        .import(Srcloc::start("*first_atom_pattern*"), a)
        .unwrap()
}

fn rest_atom_pattern<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(r ($ . atom))").unwrap();
    allocator
        .import(Srcloc::start("*first_atom_pattern*"), a)
        .unwrap()
}

fn path_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let first_atom_pattern = first_atom_pattern(allocator);
    let rest_atom_pattern = rest_atom_pattern(allocator);

    /*
     * This applies the transform
     *   (f N) => A
     * and
     *   (r N) => B
     */

    let first_match = match_sexp(allocator, &first_atom_pattern, r, HashMap::new());
    let rest_match = match_sexp(allocator, &rest_atom_pattern, r, HashMap::new());

    match (first_match, rest_match) {
        (Some(first), _) => {
            match first
                .get("atom")
                .filter(|a| matches!(allocator.sexp(a), ASExp::Atom))
                .map(|a| (allocator.loc(a), allocator.atom(a)))
                .map(|(loc, atom)| (loc, number_from_u8(atom.as_ref())))
            {
                Some((loc, atom)) => {
                    let node = NodePath::new(Some(atom)).add(NodePath::new(None).first());
                    allocator.new_atom(loc, node.as_path().data())
                }
                _ => Ok(r.clone()),
            }
        }
        (_, Some(rest)) => {
            match rest
                .get("atom")
                .filter(|a| matches!(allocator.sexp(a), ASExp::Atom))
                .map(|a| (allocator.loc(a), allocator.atom(a)))
                .map(|(loc, atom)| (loc, number_from_u8(atom.as_ref())))
            {
                Some((loc, atom)) => {
                    let node = NodePath::new(Some(atom)).add(NodePath::new(None).rest());
                    allocator.new_atom(loc, node.as_path().data())
                }
                _ => Ok(r.clone()),
            }
        }
        _ => Ok(r.clone()),
    }
}

fn quote_pattern_1<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(q . 0)").unwrap();
    allocator
        .import(Srcloc::start("*quote_pattern_1*"), a)
        .unwrap()
}

fn quote_null_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let quote_pattern_1 = quote_pattern_1(allocator);

    // This applies the transform `(q . 0)` => `0`
    let t1 = match_sexp(allocator, &quote_pattern_1, r, HashMap::new());
    let loc = allocator.loc(r);
    let imported_nil = allocator.import(loc, NodePtr::NIL)?;
    Ok(t1.map(|_| imported_nil).unwrap_or_else(|| r.clone()))
}

fn apply_null_pattern_1<A: ClassicAllocator>(allocator: &mut A) -> A::NodePtr {
    let a = assemble(allocator.allocator(), "(a 0 . (: . rest))").unwrap();
    allocator
        .import(Srcloc::start("*apply_null_pattern_1*"), a)
        .unwrap()
}

fn apply_null_optimizer<A: ClassicAllocator>(
    allocator: &mut A,
    _memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
    _eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let apply_null_pattern_1 = apply_null_pattern_1(allocator);

    // This applies the transform `(a 0 ARGS)` => `0`
    let t1 = match_sexp(allocator, &apply_null_pattern_1, r, HashMap::new());
    let loc = allocator.loc(r);
    let imported_nil = allocator.import(loc, NodePtr::NIL)?;
    Ok(t1.map(|_| imported_nil).unwrap_or_else(|| r.clone()))
}

struct OptimizerRunner<'a, A: ClassicAllocator>
where
    A::NodePtr: Clone,
{
    pub name: String,
    #[allow(clippy::type_complexity)]
    to_run: &'a dyn Fn(
        &mut A,
        &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
        &A::NodePtr,
        Rc<dyn TRunProgram>,
    ) -> Result<A::NodePtr, ClError>,
}

impl<'a, A: ClassicAllocator> OptimizerRunner<'a, A>
where
    A::NodePtr: Clone,
{
    pub fn invoke(
        &self,
        allocator: &mut A,
        memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
        r: &A::NodePtr,
        eval_f: Rc<dyn TRunProgram>,
    ) -> Result<A::NodePtr, ClError> {
        (self.to_run)(allocator, memo, r, eval_f)
    }

    #[allow(clippy::type_complexity)]
    pub fn new(
        name: &str,
        to_run: &'a dyn Fn(
            &mut A,
            &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
            &A::NodePtr,
            Rc<dyn TRunProgram>,
        ) -> Result<A::NodePtr, ClError>,
    ) -> Self {
        OptimizerRunner {
            name: name.to_string(),
            to_run,
        }
    }
}

pub fn optimize_sexp_<A: ClassicAllocator>(
    allocator: &mut A,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r_: &A::NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    // First compare the NodePtr to see if we've cached this exact one.
    // Note that this scoping is here to prevent the borrowed mutable ref from
    // preventing us from using memo downstream when we've done one optimize
    // pass and need to cache the result.
    let exported_r = allocator.export(r_);
    {
        let memo_ref: Ref<HashMap<AllocatorRefOrTreeHash, NodePtr>> = memo.borrow();
        let memo: &HashMap<AllocatorRefOrTreeHash, NodePtr> = &memo_ref;
        if let Some(res) = memo.get(&AllocatorRefOrTreeHash::new_from_nodeptr(exported_r)) {
            let imported_cached = allocator.import(allocator.loc(r_), *res)?;
            return Ok(imported_cached);
        }
    }

    // Fall back to treehash comparison since we didn't get an exact pointer hit.
    let footprint = AllocatorRefOrTreeHash::new_from_sexp(allocator.allocator(), exported_r);
    {
        let memo_ref: Ref<HashMap<AllocatorRefOrTreeHash, NodePtr>> = memo.borrow();
        let memo: &HashMap<AllocatorRefOrTreeHash, NodePtr> = &memo_ref;
        if let Some(res) = memo.get(&footprint) {
            let imported_cached = allocator.import(allocator.loc(r_), *res)?;
            return Ok(imported_cached);
        }
    }

    /*
     * Optimize an s-expression R written for clvm to R_opt where
     * (a R args) == (a R_opt args) for ANY args.
     */
    let optimizers: Vec<OptimizerRunner<A>> = vec![
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

    let mut r = r_.clone();

    loop {
        let start_r = r.clone();
        let start_r_export = allocator.export(&r);
        let mut name = "".to_string();

        match allocator.sexp(&r) {
            ASExp::Atom => {
                return Ok(r.clone());
            }
            ASExp::Pair(_, _) => {
                for opt in optimizers.iter() {
                    name.clone_from(&opt.name);
                    match opt.invoke(allocator, memo, &r, eval_f.clone()) {
                        Err(e) => {
                            return Err(e);
                        }
                        Ok(res) => {
                            if !equal_to(allocator, &r, &res) {
                                r = res;
                                break;
                            }
                        }
                    }
                }

                if equal_to(allocator, &start_r, &r) {
                    memo.replace_with(|mr| {
                        let mut work = HashMap::new();
                        swap(&mut work, mr);
                        work.insert(footprint.clone(), start_r_export);
                        let r_export = allocator.export(&r);
                        work.insert(
                            AllocatorRefOrTreeHash::new_from_nodeptr(r_export),
                            start_r_export,
                        );
                        work
                    });

                    return Ok(start_r);
                }

                if DEBUG_OPTIMIZATIONS {
                    println!(
                        "OPT-{:?}[{}] => {}",
                        name,
                        allocator.disassemble(&start_r, None),
                        allocator.disassemble(&r, None)
                    );
                }
            }
        }
    }
}

pub fn optimize_sexp<A: ClassicAllocator>(
    allocator: &mut A,
    r: &A::NodePtr,
    eval_f: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let optimized = RefCell::new(HashMap::new());

    if DIAG_OPTIMIZATIONS {
        println!("START OPTIMIZE {}", allocator.disassemble(r, None));
    }
    optimize_sexp_(allocator, &optimized, r, eval_f).inspect(|x| {
        if DIAG_OPTIMIZATIONS {
            println!(
                "OPTIMIZE_SEXP {} GIVING {}",
                allocator.disassemble(r, None),
                allocator.disassemble(x, None)
            );
        }
    })
}

pub fn do_optimize<A: ClassicAllocator>(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut A,
    memo: &RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    r: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let r_first = first(allocator, r)?;
    optimize_sexp_(allocator, memo, &r_first, runner.clone())
}
