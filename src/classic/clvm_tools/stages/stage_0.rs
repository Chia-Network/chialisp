use clvm_rs::allocator::{Allocator, NodePtr};
use clvm_rs::chia_dialect::{ChiaDialect, ENABLE_KECCAK_OPS_OUTSIDE_GUARD, NO_UNKNOWN_OPS};
use clvm_rs::core_ops::{op_cons, op_eq, op_first, op_if, op_listp, op_raise, op_rest};
use clvm_rs::cost::Cost;
use clvm_rs::dialect::{Dialect, OperatorSet};
use clvm_rs::error::EvalErr;
use clvm_rs::more_ops::{
    op_add, op_all, op_any, op_ash, op_concat, op_div, op_divmod, op_gr, op_gr_bytes, op_logand,
    op_logior, op_lognot, op_logxor, op_lsh, op_multiply, op_not, op_point_add, op_pubkey_for_exp,
    op_sha256, op_strlen, op_substr, op_subtract, op_unknown,
};
use clvm_rs::reduction::{Reduction, Response};

use clvm_rs::run_program::{run_program_with_pre_eval, PreEval};

use crate::classic::clvm::OPERATORS_LATEST_VERSION;

#[derive(Default)]
pub struct RunProgramOption {
    pub max_cost: Option<Cost>,
    pub pre_eval_f: Option<PreEval>,
    pub strict: bool,
    pub operators_version: usize,
}

pub trait TRunProgram {
    fn run_program(
        &self,
        allocator: &mut Allocator,
        program: NodePtr,
        args: NodePtr,
        option: Option<RunProgramOption>,
    ) -> Response;
}

pub struct DefaultProgramRunner {}

impl DefaultProgramRunner {
    pub fn new() -> Self {
        DefaultProgramRunner {}
    }
}

impl Default for DefaultProgramRunner {
    fn default() -> Self {
        DefaultProgramRunner::new()
    }
}

pub struct OriginalDialect {
    flags: u32,
}

impl OriginalDialect {
    pub fn new(flags: u32) -> Self {
        OriginalDialect { flags }
    }
}

// Copied from clvm_rs because it isn't public.
fn unknown_operator(
    allocator: &mut Allocator,
    o: NodePtr,
    args: NodePtr,
    flags: u32,
    max_cost: Cost,
) -> Response {
    if (flags & NO_UNKNOWN_OPS) != 0 {
        Err(EvalErr::InternalError(
            o,
            "unimplemented operator".to_string(),
        ))
    } else {
        op_unknown(allocator, o, args, max_cost)
    }
}

impl Dialect for OriginalDialect {
    fn op(
        &self,
        allocator: &mut Allocator,
        o: NodePtr,
        argument_list: NodePtr,
        max_cost: Cost,
        _extension: OperatorSet,
    ) -> Response {
        let op_len = allocator.atom_len(o);
        if op_len > 1 {
            return unknown_operator(allocator, o, argument_list, self.flags, max_cost);
        }
        let Some(op) = allocator.small_number(o) else {
            return unknown_operator(allocator, o, argument_list, self.flags, max_cost);
        };
        let f = match op {
            // 1 = quote
            // 2 = apply
            3 => op_if,
            4 => op_cons,
            5 => op_first,
            6 => op_rest,
            7 => op_listp,
            8 => op_raise,
            9 => op_eq,
            10 => op_gr_bytes,
            11 => op_sha256,
            12 => op_substr,
            13 => op_strlen,
            14 => op_concat,
            // 15 ---
            16 => op_add,
            17 => op_subtract,
            18 => op_multiply,
            19 => op_div,
            20 => op_divmod,
            21 => op_gr,
            22 => op_ash,
            23 => op_lsh,
            24 => op_logand,
            25 => op_logior,
            26 => op_logxor,
            27 => op_lognot,
            // 28 ---
            29 => op_point_add,
            30 => op_pubkey_for_exp,
            // 31 ---
            32 => op_not,
            33 => op_any,
            34 => op_all,
            // 35 ---
            // 36 = softfork
            _ => {
                return unknown_operator(allocator, o, argument_list, self.flags, max_cost);
            }
        };
        f(allocator, argument_list, max_cost)
    }

    fn quote_kw(&self) -> u32 {
        1
    }
    fn apply_kw(&self) -> u32 {
        2
    }
    fn softfork_kw(&self) -> u32 {
        36
    }

    // interpret the extension argument passed to the softfork operator, and
    // return the Operators it enables (or None) if we don't know what it means
    fn softfork_extension(&self, _ext: u32) -> OperatorSet {
        OperatorSet::Default
    }

    fn allow_unknown_ops(&self) -> bool {
        (self.flags & NO_UNKNOWN_OPS) == 0
    }
}

type PostEvalT = Box<dyn Fn(&mut Allocator, Option<NodePtr>)>;
type PreEvalT = Box<dyn Fn(&mut Allocator, NodePtr, NodePtr) -> Result<Option<PostEvalT>, EvalErr>>;

fn run_program_with_pre_eval_dialect<D: Dialect>(
    allocator: &mut Allocator,
    dialect: &D,
    program: NodePtr,
    args: NodePtr,
    max_cost: Cost,
    pre_eval_f: Option<PreEvalT>,
) -> Result<Reduction, EvalErr> {
    run_program_with_pre_eval(allocator, dialect, program, args, max_cost, pre_eval_f)
}

impl TRunProgram for DefaultProgramRunner {
    fn run_program(
        &self,
        allocator: &mut Allocator,
        program: NodePtr,
        args: NodePtr,
        option: Option<RunProgramOption>,
    ) -> Response {
        let max_cost = option.as_ref().and_then(|o| o.max_cost).unwrap_or(0);
        let operators_version = option
            .as_ref()
            .map(|o| o.operators_version)
            .unwrap_or(OPERATORS_LATEST_VERSION);

        match operators_version {
            0 => run_program_with_pre_eval_dialect(
                allocator,
                &OriginalDialect::new(NO_UNKNOWN_OPS),
                program,
                args,
                max_cost,
                option.and_then(|o| o.pre_eval_f),
            ),
            1 => run_program_with_pre_eval_dialect(
                allocator,
                &ChiaDialect::new(NO_UNKNOWN_OPS),
                program,
                args,
                max_cost,
                option.and_then(|o| o.pre_eval_f),
            ),
            _ => run_program_with_pre_eval_dialect(
                allocator,
                &ChiaDialect::new(NO_UNKNOWN_OPS | ENABLE_KECCAK_OPS_OUTSIDE_GUARD),
                program,
                args,
                max_cost,
                option.and_then(|o| o.pre_eval_f),
            ),
        }
    }
}
