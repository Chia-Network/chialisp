use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use num_bigint::ToBigInt;

use clvm_rs::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::clvm::{run, truthy};
use crate::compiler::codegen::{codegen, hoist_assign_form};
use crate::compiler::compiler::is_at_capture;
use crate::compiler::comptypes::{
    Binding, BindingPattern, BodyForm, CompileErr, CompileForm, CompilerOpts, DefunData,
    HelperForm, LambdaData, LetData, LetFormInlineHint, LetFormKind,
};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::BasicCompileContext;
use crate::compiler::CompileContextWrapper;
use crate::util::{number_from_u8, u8_from_number, Number};

const PRIM_RUN_LIMIT: usize = 1000000;
pub const EVAL_STACK_LIMIT: usize = 200;

#[derive(Clone, Debug, Default)]
pub struct VisitedInfo {
    functions: HashMap<Vec<u8>, Rc<BodyForm>>,
    max_depth: Option<usize>,
}

#[derive(Clone)]
pub struct LambdaApply {
    lambda: LambdaData,
    body: Rc<BodyForm>,
    env: Rc<BodyForm>,
}

type EvalEnv = Rc<HashMap<Vec<u8>, Rc<BodyForm>>>;

#[derive(Clone)]
struct OwnedCallSpec {
    loc: Srcloc,
    name: Vec<u8>,
    args: Vec<Rc<BodyForm>>,
    tail: Option<Rc<BodyForm>>,
    original: Rc<BodyForm>,
}

struct ShrinkRequest {
    prog_args: Rc<SExp>,
    env: EvalEnv,
    body: Rc<BodyForm>,
    only_inline: bool,
    depth: usize,
}

struct IsLambdaRequest {
    prog_args: Rc<SExp>,
    env: EvalEnv,
    parts: Vec<Rc<BodyForm>>,
    only_inline: bool,
    depth: usize,
}

struct PrimitiveRequest {
    call: OwnedCallSpec,
    prog_args: Rc<SExp>,
    arguments: Vec<Rc<BodyForm>>,
    env: EvalEnv,
    only_inline: bool,
    depth: usize,
}

struct LambdaRequest {
    prog_args: Rc<SExp>,
    env: EvalEnv,
    lapply: LambdaApply,
    only_inline: bool,
    depth: usize,
}

struct InvokeRequest {
    call: OwnedCallSpec,
    prog_args: Rc<SExp>,
    arguments: Vec<Rc<BodyForm>>,
    env: EvalEnv,
    only_inline: bool,
    depth: usize,
}

struct ChaseRequest {
    body: Rc<BodyForm>,
    depth: usize,
}

struct MashRequest {
    maybe_condition: Rc<BodyForm>,
    env: Rc<BodyForm>,
    depth: usize,
}

struct EnrichRequest {
    prog_args: Rc<SExp>,
    env: EvalEnv,
    ldata: LambdaData,
    only_inline: bool,
    depth: usize,
}

enum EvalRequest {
    Shrink(ShrinkRequest),
    IsLambda(IsLambdaRequest),
    Primitive(PrimitiveRequest),
    Lambda(LambdaRequest),
    Invoke(InvokeRequest),
    Chase(ChaseRequest),
    Mash(MashRequest),
    Enrich(EnrichRequest),
}

enum EvalValue {
    Body(Rc<BodyForm>),
    Lambda(Option<LambdaApply>),
}

type EvalResult = Result<EvalValue, CompileErr>;

struct PrimitiveState {
    call: OwnedCallSpec,
    prog_args: Rc<SExp>,
    arguments: Vec<Rc<BodyForm>>,
    env: EvalEnv,
    only_inline: bool,
    depth: usize,
    prim: Rc<SExp>,
    target: Vec<Rc<BodyForm>>,
    converted: Vec<Option<Rc<SExp>>>,
    next: usize,
    all_primitive: bool,
}

struct DefunState {
    defun: Box<DefunData>,
    prog_args: Rc<SExp>,
    arguments: Vec<Rc<BodyForm>>,
    env: EvalEnv,
    only_inline: bool,
    depth: usize,
    call_loc: Srcloc,
}

struct CaptureState {
    defun: Box<DefunData>,
    prog_args: Rc<SExp>,
    env: EvalEnv,
    only_inline: bool,
    depth: usize,
    captures: Vec<(Vec<u8>, Rc<BodyForm>)>,
    translated: HashMap<Vec<u8>, Rc<BodyForm>>,
    next: usize,
}

enum Continuation {
    Identity,
    IsLambdaProgram(IsLambdaRequest),
    IsLambdaEnv {
        evaluated_prog: Rc<BodyForm>,
        request: IsLambdaRequest,
    },
    LambdaCaptures(LambdaRequest),
    PrimitiveArg(PrimitiveState),
    PrimitiveLambda(PrimitiveState),
    Chase {
        depth: usize,
    },
    ContinueApply {
        depth: usize,
    },
    MashOrOriginal {
        original: Rc<BodyForm>,
    },
    MashTrue {
        x_head: Rc<BodyForm>,
        cond: Rc<BodyForm>,
        iffalse: Rc<BodyForm>,
        apply_head: Rc<BodyForm>,
        env: Rc<BodyForm>,
        location: Srcloc,
        depth: usize,
    },
    MashFalse {
        x_head: Rc<BodyForm>,
        cond: Rc<BodyForm>,
        true_result: Rc<BodyForm>,
        location: Srcloc,
    },
    DefunTail(DefunState),
    DefunCapture(CaptureState),
    EnrichCaptures(EnrichRequest),
    EnrichBody {
        ldata: LambdaData,
        new_captures: Rc<BodyForm>,
        interpretable: HashMap<Vec<u8>, Rc<BodyForm>>,
    },
}

enum EvalStep {
    Request(Box<EvalRequest>, Box<Continuation>),
    Complete(EvalResult),
}

// Frontend evaluator based on my fuzzer representation and direct interpreter of
// that.
#[derive(Debug)]
pub enum ArgInputs {
    Whole(Rc<BodyForm>),
    Pair(Rc<ArgInputs>, Rc<ArgInputs>),
}

/// Evaluator is an object that simplifies expressions, given the helpers
/// (helpers are forms that are reusable parts of programs, such as defconst,
/// defun or defmacro) from a program.  In the simplest form, it can be used to
/// power a chialisp repl, but also to simplify expressions to their components.
///
/// The emitted expressions are simpler and sometimes smaller, depending on what the
/// evaulator was able to do.  It performs all obvious substitutions and some
/// obvious simplifications based on CLVM operations (such as combining
/// picking operations with conses in some cases).  If the expression can't
/// be simplified to a constant, any remaining variable references and the
/// operations on them are left.
///
/// Because of what it can do, it's also used for "use checking" to determine
/// whether input parameters to the program as a whole are used in the program's
/// eventual results.  The simplification it does is general eta conversion with
/// some other local transformations thrown in.
pub struct Evaluator {
    opts: Rc<dyn CompilerOpts>,
    runner: Rc<dyn TRunProgram>,
    prims: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    helpers: Vec<HelperForm>,
    mash_conditions: bool,
    ignore_exn: bool,
}

fn select_helper(bindings: &[HelperForm], name: &[u8]) -> Option<HelperForm> {
    for b in bindings.iter() {
        if b.name() == name {
            return Some(b.clone());
        }
    }

    None
}

fn compute_paths_of_destructure(
    bindings: &mut Vec<(Vec<u8>, Rc<BodyForm>)>,
    structure: Rc<SExp>,
    path: Number,
    mask: Number,
    bodyform: Rc<BodyForm>,
) {
    match structure.atomize() {
        SExp::Cons(_, a, b) => {
            let next_mask = mask.clone() * 2_u32.to_bigint().unwrap();
            let next_right_path = mask + path.clone();
            compute_paths_of_destructure(bindings, a, path, next_mask.clone(), bodyform.clone());
            compute_paths_of_destructure(bindings, b, next_right_path, next_mask, bodyform);
        }
        SExp::Atom(_, name) => {
            let mut produce_path = path.clone() | mask;
            let mut output_form = bodyform.clone();

            while produce_path > bi_one() {
                if path.clone() & produce_path.clone() != bi_zero() {
                    // Right path
                    output_form = Rc::new(make_operator1(
                        &bodyform.loc(),
                        "r".to_string(),
                        output_form,
                    ));
                } else {
                    // Left path
                    output_form = Rc::new(make_operator1(
                        &bodyform.loc(),
                        "f".to_string(),
                        output_form,
                    ));
                }

                produce_path /= 2_u32.to_bigint().unwrap();
            }

            bindings.push((name, output_form));
        }
        _ => {}
    }
}

fn update_parallel_bindings(
    bindings: &HashMap<Vec<u8>, Rc<BodyForm>>,
    have_bindings: &[Rc<Binding>],
) -> HashMap<Vec<u8>, Rc<BodyForm>> {
    let mut new_bindings = bindings.clone();
    for b in have_bindings.iter() {
        match &b.pattern {
            BindingPattern::Name(name) => {
                new_bindings.insert(name.clone(), b.body.clone());
            }
            BindingPattern::Complex(structure) => {
                let mut computed_getters = Vec::new();
                compute_paths_of_destructure(
                    &mut computed_getters,
                    structure.clone(),
                    bi_zero(),
                    bi_one(),
                    b.body.clone(),
                );
                for (name, p) in computed_getters.iter() {
                    new_bindings.insert(name.clone(), p.clone());
                }
            }
        }
    }
    new_bindings
}

// Tell whether the bodyform is a simple primitive.
pub fn is_primitive(expr: &BodyForm) -> bool {
    matches!(
        expr,
        BodyForm::Quoted(_)
            | BodyForm::Value(SExp::Nil(_))
            | BodyForm::Value(SExp::Integer(_, _))
            | BodyForm::Value(SExp::QuotedString(_, _, _))
    )
}

fn make_operator1(l: &Srcloc, op: String, arg: Rc<BodyForm>) -> BodyForm {
    BodyForm::Call(
        l.clone(),
        vec![
            Rc::new(BodyForm::Value(SExp::atom_from_string(l.clone(), &op))),
            arg,
        ],
        None,
    )
}

fn make_operator2(l: &Srcloc, op: String, arg1: Rc<BodyForm>, arg2: Rc<BodyForm>) -> BodyForm {
    BodyForm::Call(
        l.clone(),
        vec![
            Rc::new(BodyForm::Value(SExp::atom_from_string(l.clone(), &op))),
            arg1,
            arg2,
        ],
        None,
    )
}

// For any arginput, give a bodyform that computes it.  In most cases, the
// bodyform is extracted, in a few cases, we may need to form a cons operation.
fn get_bodyform_from_arginput(l: &Srcloc, arginput: &ArgInputs) -> Rc<BodyForm> {
    match arginput {
        ArgInputs::Whole(bf) => bf.clone(),
        ArgInputs::Pair(a, b) => {
            let bfa = get_bodyform_from_arginput(l, a);
            let bfb = get_bodyform_from_arginput(l, b);
            Rc::new(make_operator2(l, "c".to_string(), bfa, bfb))
        }
    }
}

// Given an SExp argument capture structure and SExp containing the arguments
// constructed for the function, populate a HashMap with minimized expressions
// which match the requested argument destructuring.
//
// It's possible this will result in irreducible (unknown at compile time)
// argument expressions.
pub fn create_argument_captures(
    argument_captures: &mut HashMap<Vec<u8>, Rc<BodyForm>>,
    formed_arguments: &ArgInputs,
    function_arg_spec: Rc<SExp>,
) -> Result<(), CompileErr> {
    match (formed_arguments, function_arg_spec.borrow()) {
        (_, SExp::Nil(_)) => Ok(()),
        (ArgInputs::Whole(bf), SExp::Cons(l, f, r)) => {
            match (is_at_capture(f.clone(), r.clone()), bf.borrow()) {
                (Some((capture, substructure)), BodyForm::Quoted(SExp::Cons(_, _, _))) => {
                    argument_captures.insert(capture, bf.clone());
                    create_argument_captures(argument_captures, formed_arguments, substructure)
                }
                (None, BodyForm::Quoted(SExp::Cons(_, fa, ra))) => {
                    // Argument destructuring splits a quoted sexp that can itself
                    // be destructured.
                    let fa_borrowed: &SExp = fa.borrow();
                    let ra_borrowed: &SExp = ra.borrow();
                    create_argument_captures(
                        argument_captures,
                        &ArgInputs::Whole(Rc::new(BodyForm::Quoted(fa_borrowed.clone()))),
                        f.clone(),
                    )?;
                    create_argument_captures(
                        argument_captures,
                        &ArgInputs::Whole(Rc::new(BodyForm::Quoted(ra_borrowed.clone()))),
                        r.clone(),
                    )
                }
                (Some((capture, substructure)), bf) => {
                    argument_captures.insert(capture, Rc::new(bf.clone()));
                    create_argument_captures(argument_captures, formed_arguments, substructure)
                }
                (None, bf) => {
                    // Argument destructuring splits a value that couldn't
                    // previously be reduced.  We'll punt it back unreduced by
                    // specifying how the right part is reached.
                    create_argument_captures(
                        argument_captures,
                        &ArgInputs::Whole(Rc::new(make_operator1(
                            l,
                            "f".to_string(),
                            Rc::new(bf.clone()),
                        ))),
                        f.clone(),
                    )?;
                    create_argument_captures(
                        argument_captures,
                        &ArgInputs::Whole(Rc::new(make_operator1(
                            l,
                            "r".to_string(),
                            Rc::new(bf.clone()),
                        ))),
                        r.clone(),
                    )
                }
            }
        }
        (ArgInputs::Pair(af, ar), SExp::Cons(l, f, r)) => {
            if let Some((capture, substructure)) = is_at_capture(f.clone(), r.clone()) {
                let bfa = get_bodyform_from_arginput(l, af);
                let bfb = get_bodyform_from_arginput(l, ar);
                let fused_arguments = Rc::new(make_operator2(l, "c".to_string(), bfa, bfb));
                argument_captures.insert(capture, fused_arguments);
                create_argument_captures(argument_captures, formed_arguments, substructure)
            } else {
                create_argument_captures(argument_captures, af, f.clone())?;
                create_argument_captures(argument_captures, ar, r.clone())
            }
        }
        (ArgInputs::Whole(x), SExp::Atom(_, name)) => {
            argument_captures.insert(name.clone(), x.clone());
            Ok(())
        }
        (ArgInputs::Pair(_, _), SExp::Atom(l, name)) => {
            argument_captures.insert(
                name.clone(),
                get_bodyform_from_arginput(l, formed_arguments),
            );
            Ok(())
        }
        (_, _) => Err(CompileErr(
            function_arg_spec.loc(),
            format!(
                "not yet supported argument alternative: ArgInput {formed_arguments:?} SExp {function_arg_spec}"
            ),
        )),
    }
}

fn arg_inputs_primitive(arginputs: Rc<ArgInputs>) -> bool {
    match arginputs.borrow() {
        ArgInputs::Whole(bf) => is_primitive(bf),
        ArgInputs::Pair(a, b) => arg_inputs_primitive(a.clone()) && arg_inputs_primitive(b.clone()),
    }
}

fn decons_args(formed_tail: Rc<BodyForm>) -> ArgInputs {
    if let Some((head, tail)) = match_cons(formed_tail.clone()) {
        let arg_head = decons_args(head.clone());
        let arg_tail = decons_args(tail.clone());
        ArgInputs::Pair(Rc::new(arg_head), Rc::new(arg_tail))
    } else {
        ArgInputs::Whole(formed_tail)
    }
}

fn build_argument_captures(
    l: &Srcloc,
    arguments_to_convert: &[Rc<BodyForm>],
    tail: Option<Rc<BodyForm>>,
    args: Rc<SExp>,
) -> Result<HashMap<Vec<u8>, Rc<BodyForm>>, CompileErr> {
    let formed_tail = tail.unwrap_or_else(|| Rc::new(BodyForm::Quoted(SExp::Nil(l.clone()))));
    let mut formed_arguments = decons_args(formed_tail);

    for i_reverse in 0..arguments_to_convert.len() {
        let i = arguments_to_convert.len() - i_reverse - 1;
        formed_arguments = ArgInputs::Pair(
            Rc::new(ArgInputs::Whole(arguments_to_convert[i].clone())),
            Rc::new(formed_arguments),
        );
    }

    let mut argument_captures = HashMap::new();
    create_argument_captures(&mut argument_captures, &formed_arguments, args)?;
    Ok(argument_captures)
}

fn make_prim_call(l: Srcloc, prim: Rc<SExp>, args: Rc<SExp>) -> Rc<SExp> {
    Rc::new(SExp::Cons(l, prim, args))
}

pub fn build_reflex_captures(captures: &mut HashMap<Vec<u8>, Rc<BodyForm>>, args: Rc<SExp>) {
    match args.borrow() {
        SExp::Atom(l, name) => {
            captures.insert(
                name.clone(),
                Rc::new(BodyForm::Value(SExp::Atom(l.clone(), name.clone()))),
            );
        }
        SExp::Cons(l, a, b) => {
            if let Some((capture, substructure)) = is_at_capture(a.clone(), b.clone()) {
                captures.insert(
                    capture.clone(),
                    Rc::new(BodyForm::Value(SExp::Atom(l.clone(), capture))),
                );
                build_reflex_captures(captures, substructure);
            } else {
                build_reflex_captures(captures, a.clone());
                build_reflex_captures(captures, b.clone());
            }
        }
        _ => {}
    }
}

pub fn dequote(l: Srcloc, exp: Rc<BodyForm>) -> Result<Rc<SExp>, CompileErr> {
    match exp.borrow() {
        BodyForm::Quoted(v) => Ok(Rc::new(v.clone())),
        _ => Err(CompileErr(
            l,
            format!(
                "not a quoted result in macro expansion: {} {:?}",
                exp.to_sexp(),
                exp
            ),
        )),
    }
}

/*
fn show_env(env: &HashMap<Vec<u8>, Rc<BodyForm>>) {
    let loc = Srcloc::start(&"*env*".to_string());
    for kv in env.iter() {
        println!(
            " - {}: {}",
            SExp::Atom(loc.clone(), kv.0.clone()).to_string(),
            kv.1.to_sexp().to_string()
        );
    }
}
*/

pub fn first_of_alist(lst: Rc<SExp>) -> Result<Rc<SExp>, CompileErr> {
    match lst.borrow() {
        SExp::Cons(_, f, _) => Ok(f.clone()),
        _ => Err(CompileErr(lst.loc(), format!("No first element of {lst}"))),
    }
}

pub fn second_of_alist(lst: Rc<SExp>) -> Result<Rc<SExp>, CompileErr> {
    match lst.borrow() {
        SExp::Cons(_, _, r) => first_of_alist(r.clone()),
        _ => Err(CompileErr(lst.loc(), format!("No second element of {lst}"))),
    }
}

fn synthesize_args(
    template: Rc<SExp>,
    env: &HashMap<Vec<u8>, Rc<BodyForm>>,
) -> Result<Rc<BodyForm>, CompileErr> {
    match template.borrow() {
        SExp::Atom(_, name) => env.get(name).map(|x| Ok(x.clone())).unwrap_or_else(|| {
            Err(CompileErr(
                template.loc(),
                format!("Argument {template} referenced but not in env"),
            ))
        }),
        SExp::Cons(l, f, r) => {
            if let Some((capture, _substructure)) = is_at_capture(f.clone(), r.clone()) {
                synthesize_args(Rc::new(SExp::Atom(l.clone(), capture)), env)
            } else {
                Ok(Rc::new(BodyForm::Call(
                    l.clone(),
                    vec![
                        Rc::new(BodyForm::Value(SExp::atom_from_string(template.loc(), "c"))),
                        synthesize_args(f.clone(), env)?,
                        synthesize_args(r.clone(), env)?,
                    ],
                    None,
                )))
            }
        }
        SExp::Nil(l) => Ok(Rc::new(BodyForm::Quoted(SExp::Nil(l.clone())))),
        _ => Err(CompileErr(
            template.loc(),
            format!("unknown argument template {template}"),
        )),
    }
}

fn reflex_capture(name: &[u8], capture: Rc<BodyForm>) -> bool {
    match capture.borrow() {
        BodyForm::Value(SExp::Atom(_, n)) => n == name,
        _ => false,
    }
}

fn match_atom_to_prim(name: Vec<u8>, p: u8, h: Rc<SExp>) -> bool {
    match h.borrow() {
        SExp::Atom(_, v) => v == &name || (v.len() == 1 && v[0] == p),
        SExp::Integer(_, v) => *v == p.to_bigint().unwrap(),
        _ => false,
    }
}

fn is_quote_atom(h: Rc<SExp>) -> bool {
    match_atom_to_prim(vec![b'q'], 1, h)
}

pub fn is_apply_atom(h: Rc<SExp>) -> bool {
    match_atom_to_prim(vec![b'a'], 2, h)
}

pub fn is_i_atom(h: Rc<SExp>) -> bool {
    match_atom_to_prim(vec![b'i'], 3, h)
}

pub fn is_not_atom(h: Rc<SExp>) -> bool {
    match_atom_to_prim(b"not".to_vec(), 32, h)
}

fn is_cons_atom(h: Rc<SExp>) -> bool {
    match_atom_to_prim(vec![b'c'], 4, h)
}

fn match_cons(args: Rc<BodyForm>) -> Option<(Rc<BodyForm>, Rc<BodyForm>)> {
    // Since this matches a primitve, there's no alternative for a tail.
    if let BodyForm::Call(_, v, None) = args.borrow() {
        if v.len() < 3 {
            return None;
        }
        let have_cons_atom = is_cons_atom(v[0].to_sexp());
        if have_cons_atom {
            return Some((v[1].clone(), v[2].clone()));
        }
    }

    None
}

fn promote_args_to_bodyform(
    head: Rc<SExp>,
    arg: Rc<SExp>,
    whole_args: Rc<BodyForm>,
) -> Result<Vec<Rc<BodyForm>>, CompileErr> {
    if let Some(v) = arg.proper_list() {
        let head_borrowed: &SExp = head.borrow();
        let mut result = vec![Rc::new(BodyForm::Value(head_borrowed.clone()))];
        for a in v.iter() {
            result.push(promote_program_to_bodyform(
                Rc::new(a.clone()),
                whole_args.clone(),
            )?);
        }
        return Ok(result);
    }

    Err(CompileErr(
        arg.loc(),
        "improper argument list for primitive".to_string(),
    ))
}

fn choose_from_env_by_path(path_: Number, args_program: Rc<BodyForm>) -> Rc<BodyForm> {
    let mut path = path_;
    let mut op_list = Vec::new();
    let two = 2_i32.to_bigint().unwrap();

    if path == bi_zero() {
        return Rc::new(BodyForm::Quoted(SExp::Nil(args_program.loc())));
    }

    while path != bi_one() {
        op_list.push(path.clone() % two.clone() == bi_one());
        path = path.clone() / two.clone();
    }

    let mut result_form = args_program.clone();
    for op in op_list.iter() {
        if let Some((head, tail)) = match_cons(result_form.clone()) {
            if *op {
                result_form = tail.clone();
            } else {
                result_form = head.clone();
            }
        } else {
            let apply_op = if *op { 6 } else { 5 };
            result_form = Rc::new(BodyForm::Call(
                args_program.loc(),
                vec![
                    Rc::new(BodyForm::Value(SExp::Atom(
                        args_program.loc(),
                        vec![apply_op],
                    ))),
                    result_form,
                ],
                None,
            ));
        }
    }
    result_form
}

fn promote_program_to_bodyform(
    program: Rc<SExp>,
    env: Rc<BodyForm>,
) -> Result<Rc<BodyForm>, CompileErr> {
    match program.borrow() {
        SExp::Cons(_, h, t) => {
            if is_quote_atom(h.clone()) {
                let t_borrowed: &SExp = t.borrow();
                return Ok(Rc::new(BodyForm::Quoted(t_borrowed.clone())));
            }

            // Process tails to change bare numbers to (@ n)
            let args = promote_args_to_bodyform(h.clone(), t.clone(), env)?;
            Ok(Rc::new(BodyForm::Call(program.loc(), args, None)))
        }
        SExp::Integer(_, n) => {
            // A program that is an atom refers to a position
            // in the environment.
            Ok(choose_from_env_by_path(n.clone(), env))
        }
        SExp::QuotedString(_, _, v) => {
            // Treated as integer path.
            let integer = number_from_u8(v);
            Ok(choose_from_env_by_path(integer, env))
        }
        SExp::Atom(_, v) => {
            // Treated as integer path.
            let integer = number_from_u8(v);
            Ok(choose_from_env_by_path(integer, env))
        }
        _ => {
            let borrowed_program: &SExp = program.borrow();
            Ok(Rc::new(BodyForm::Quoted(borrowed_program.clone())))
        }
    }
}

fn match_i_op(candidate: Rc<BodyForm>) -> Option<(Rc<BodyForm>, Rc<BodyForm>, Rc<BodyForm>)> {
    // Matches a primitve, no possibility of a tail item.
    if let BodyForm::Call(_, cvec, None) = candidate.borrow() {
        if cvec.len() != 4 {
            return None;
        }
        if let BodyForm::Value(atom) = cvec[0].borrow() {
            if is_i_atom(Rc::new(atom.clone())) {
                return Some((cvec[1].clone(), cvec[2].clone(), cvec[3].clone()));
            }
        }
    }

    None
}

fn flatten_expression_to_names_inner(collection: &mut HashSet<Vec<u8>>, expr: Rc<SExp>) {
    match expr.borrow() {
        SExp::Cons(_, a, b) => {
            flatten_expression_to_names_inner(collection, a.clone());
            flatten_expression_to_names_inner(collection, b.clone());
        }
        SExp::Atom(_, a) => {
            collection.insert(a.clone());
        }
        _ => {}
    }
}

fn flatten_expression_to_names(expr: Rc<SExp>) -> Rc<BodyForm> {
    let mut collection = HashSet::new();
    flatten_expression_to_names_inner(&mut collection, expr.clone());
    let mut transformed = Vec::new();
    for a in collection.iter() {
        transformed.push(a.clone());
    }
    transformed.sort();
    let mut call_vec: Vec<Rc<BodyForm>> = transformed
        .iter()
        .map(|x| Rc::new(BodyForm::Value(SExp::Atom(expr.loc(), x.clone()))))
        .collect();
    call_vec.insert(
        0,
        Rc::new(BodyForm::Value(SExp::Atom(expr.loc(), vec![b'+']))),
    );
    Rc::new(BodyForm::Call(expr.loc(), call_vec, None))
}

pub fn eval_dont_expand_let(inline_hint: &Option<LetFormInlineHint>) -> bool {
    matches!(inline_hint, Some(LetFormInlineHint::NonInline(_)))
}

pub fn filter_capture_args(args: Rc<SExp>, name_map: &HashMap<Vec<u8>, Rc<BodyForm>>) -> Rc<SExp> {
    match args.borrow() {
        SExp::Cons(l, a, b) => {
            let a_filtered = filter_capture_args(a.clone(), name_map);
            let b_filtered = filter_capture_args(b.clone(), name_map);
            if !truthy(a_filtered.clone()) && !truthy(b_filtered.clone()) {
                return Rc::new(SExp::Nil(l.clone()));
            }
            Rc::new(SExp::Cons(l.clone(), a_filtered, b_filtered))
        }
        SExp::Atom(l, n) => {
            if name_map.contains_key(n) {
                Rc::new(SExp::Nil(l.clone()))
            } else {
                args
            }
        }
        _ => Rc::new(SExp::Nil(args.loc())),
    }
}

impl Evaluator {
    pub fn new(
        opts: Rc<dyn CompilerOpts>,
        runner: Rc<dyn TRunProgram>,
        helpers: Vec<HelperForm>,
    ) -> Self {
        Evaluator {
            opts: opts.clone(),
            runner,
            prims: opts.prim_map(),
            helpers,
            mash_conditions: false,
            ignore_exn: false,
        }
    }

    pub fn mash_conditions(&self) -> Self {
        Evaluator {
            opts: self.opts.clone(),
            runner: self.runner.clone(),
            prims: self.prims.clone(),
            helpers: self.helpers.clone(),
            mash_conditions: true,
            ignore_exn: true,
        }
    }

    fn body_result(result: EvalResult, loc: Srcloc) -> Result<Rc<BodyForm>, CompileErr> {
        match result? {
            EvalValue::Body(body) => Ok(body),
            EvalValue::Lambda(_) => Err(CompileErr(
                loc,
                "internal evaluator return type mismatch".to_string(),
            )),
        }
    }

    fn body_done(result: Result<Rc<BodyForm>, CompileErr>) -> EvalStep {
        EvalStep::Complete(result.map(EvalValue::Body))
    }

    fn increment_depth(
        state: &VisitedInfo,
        depth: usize,
        loc: Srcloc,
    ) -> Result<usize, CompileErr> {
        if state.max_depth.is_some_and(|limit| depth >= limit) {
            Err(CompileErr(loc, "stack limit exceeded".to_string()))
        } else {
            Ok(depth + 1)
        }
    }

    fn request_body(request: ShrinkRequest, continuation: Continuation) -> EvalStep {
        Self::request(EvalRequest::Shrink(request), continuation)
    }

    fn request(request: EvalRequest, continuation: Continuation) -> EvalStep {
        EvalStep::Request(Box::new(request), Box::new(continuation))
    }

    #[allow(clippy::too_many_arguments)]
    fn invoke_macro_expansion(
        &self,
        context: &mut BasicCompileContext,
        l: Srcloc,
        call_loc: Srcloc,
        program: Rc<CompileForm>,
        prog_args: Rc<SExp>,
        arguments: Vec<Rc<BodyForm>>,
        env: EvalEnv,
        depth: usize,
    ) -> EvalStep {
        let mut macro_args = Rc::new(SExp::Nil(l.clone()));
        for argument in arguments.iter().rev() {
            let arg_repr = argument.to_sexp();
            macro_args = Rc::new(SExp::Cons(l.clone(), arg_repr, macro_args));
        }

        let macro_expansion = match self.expand_macro(context, l.clone(), program, macro_args) {
            Ok(expansion) => expansion,
            Err(error) => return Self::body_done(Err(error)),
        };

        if let Ok(input) = dequote(call_loc, macro_expansion.clone()) {
            let frontend_macro_input = Rc::new(SExp::Cons(
                l.clone(),
                Rc::new(SExp::atom_from_string(l.clone(), "mod")),
                Rc::new(SExp::Cons(
                    l.clone(),
                    prog_args.clone(),
                    Rc::new(SExp::Cons(l.clone(), input, Rc::new(SExp::Nil(l)))),
                )),
            ));

            match frontend(self.opts.clone(), &[frontend_macro_input]) {
                Ok(program) => Self::request_body(
                    ShrinkRequest {
                        prog_args,
                        env,
                        body: program.compileform().exp.clone(),
                        only_inline: false,
                        depth,
                    },
                    Continuation::Identity,
                ),
                Err(error) => Self::body_done(Err(error)),
            }
        } else {
            Self::body_done(promote_program_to_bodyform(
                macro_expansion.to_sexp(),
                Rc::new(BodyForm::Value(SExp::Atom(
                    macro_expansion.loc(),
                    vec![b'@'],
                ))),
            ))
        }
    }

    fn is_lambda_apply(&self, request: IsLambdaRequest) -> EvalStep {
        if request.parts.len() != 3 || !is_apply_atom(request.parts[0].to_sexp()) {
            return EvalStep::Complete(Ok(EvalValue::Lambda(None)));
        }

        Self::request_body(
            ShrinkRequest {
                prog_args: request.prog_args.clone(),
                env: request.env.clone(),
                body: request.parts[1].clone(),
                only_inline: request.only_inline,
                depth: request.depth,
            },
            Continuation::IsLambdaProgram(request),
        )
    }

    fn do_lambda_apply(&self, request: LambdaRequest) -> EvalStep {
        Self::request_body(
            ShrinkRequest {
                prog_args: request.prog_args.clone(),
                env: request.env.clone(),
                body: request.lapply.lambda.captures.clone(),
                only_inline: request.only_inline,
                depth: request.depth,
            },
            Continuation::LambdaCaptures(request),
        )
    }

    fn invoke_primitive(
        &self,
        context: &mut BasicCompileContext,
        request: PrimitiveRequest,
    ) -> EvalStep {
        if request.call.name == b"@" {
            return Self::body_done(Ok(Rc::new(BodyForm::Quoted(SExp::Cons(
                request.call.loc.clone(),
                Rc::new(SExp::Nil(request.call.loc)),
                request.prog_args,
            )))));
        }

        if request.call.name == b"com" {
            let mut end_of_list = Rc::new(SExp::Cons(
                request.call.loc.clone(),
                request.arguments[0].to_sexp(),
                Rc::new(SExp::Nil(request.call.loc.clone())),
            ));
            for h in self.helpers.iter() {
                end_of_list = Rc::new(SExp::Cons(
                    request.call.loc.clone(),
                    h.to_sexp(),
                    end_of_list,
                ))
            }
            let use_body = SExp::Cons(
                request.call.loc.clone(),
                Rc::new(SExp::Atom(request.call.loc.clone(), b"mod".to_vec())),
                Rc::new(SExp::Cons(
                    request.call.loc.clone(),
                    request.prog_args,
                    end_of_list,
                )),
            );
            return match self.compile_code(context, false, Rc::new(use_body)) {
                Ok(compiled) => {
                    Self::body_done(Ok(Rc::new(BodyForm::Quoted(compiled.as_ref().clone()))))
                }
                Err(error) => Self::body_done(Err(error)),
            };
        }

        let Some(prim) = self.lookup_prim(request.call.loc.clone(), &request.call.name) else {
            return Self::body_done(Err(CompileErr(
                request.call.loc,
                format!(
                    "Don't yet support this call type {} {:?}",
                    request.call.original.to_sexp(),
                    request.call.original
                ),
            )));
        };

        let count = request.arguments.len();
        let state = PrimitiveState {
            call: request.call,
            prog_args: request.prog_args,
            arguments: request.arguments,
            env: request.env,
            only_inline: request.only_inline,
            depth: request.depth,
            prim,
            target: Vec::new(),
            converted: vec![None; count],
            next: count,
            all_primitive: true,
        };
        self.next_primitive_argument(context, state)
    }

    fn next_primitive_argument(
        &self,
        context: &mut BasicCompileContext,
        mut state: PrimitiveState,
    ) -> EvalStep {
        if state.target.is_empty() {
            state.target = state.call.args.clone();
        }
        if state.next > 0 {
            let index = state.next - 1;
            state.next = index;
            return Self::request_body(
                ShrinkRequest {
                    prog_args: state.prog_args.clone(),
                    env: state.env.clone(),
                    body: state.arguments[index].clone(),
                    only_inline: state.only_inline,
                    depth: state.depth,
                },
                Continuation::PrimitiveArg(state),
            );
        }

        let mut converted_args = SExp::Nil(state.call.loc.clone());
        for converted in state.converted.iter().rev() {
            converted_args = SExp::Cons(
                state.call.loc.clone(),
                converted.as_ref().expect("converted argument").clone(),
                Rc::new(converted_args),
            );
        }
        if state.all_primitive {
            let result = self.run_prim(
                context.allocator(),
                state.call.loc.clone(),
                make_prim_call(state.call.loc.clone(), state.prim, Rc::new(converted_args)),
                Rc::new(SExp::Nil(state.call.loc.clone())),
            );
            return match result {
                Ok(body) => Self::body_done(Ok(body)),
                Err(_) if state.only_inline || self.ignore_exn => Self::body_done(Ok(Rc::new(
                    BodyForm::Call(state.call.loc, state.target, None),
                ))),
                Err(error) => Self::body_done(Err(error)),
            };
        }

        Self::request(
            EvalRequest::IsLambda(IsLambdaRequest {
                prog_args: state.prog_args.clone(),
                env: state.env.clone(),
                parts: state.target.clone(),
                only_inline: state.only_inline,
                depth: state.depth,
            }),
            Continuation::PrimitiveLambda(state),
        )
    }

    fn continue_apply(&self, env: Rc<BodyForm>, run_program: Rc<SExp>, depth: usize) -> EvalStep {
        match promote_program_to_bodyform(run_program.clone(), env) {
            Ok(program) => Self::request_body(
                ShrinkRequest {
                    prog_args: Rc::new(SExp::Nil(run_program.loc())),
                    env: Rc::new(HashMap::new()),
                    body: program,
                    only_inline: false,
                    depth,
                },
                Continuation::ContinueApply { depth },
            ),
            Err(error) => Self::body_done(Err(error)),
        }
    }

    fn do_mash_condition(&self, request: MashRequest, state: &mut VisitedInfo) -> EvalStep {
        if let Some((cond, iftrue, iffalse)) = match_i_op(request.maybe_condition.clone()) {
            let x_head = Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), vec![b'x'])));
            let apply_head = Rc::new(BodyForm::Value(SExp::Atom(iftrue.loc(), vec![2])));
            let where_from = cond.loc().to_string();
            let where_from_vec = where_from.as_bytes().to_vec();

            if let Some(present) = state.functions.get(&where_from_vec) {
                return Self::body_done(Ok(present.clone()));
            }
            state.functions.insert(
                where_from_vec,
                Rc::new(BodyForm::Call(
                    request.maybe_condition.loc(),
                    vec![x_head.clone(), cond.clone()],
                    None,
                )),
            );
            return Self::request(
                EvalRequest::Chase(ChaseRequest {
                    body: Rc::new(BodyForm::Call(
                        iftrue.loc(),
                        vec![apply_head.clone(), iftrue.clone(), request.env.clone()],
                        None,
                    )),
                    depth: request.depth,
                }),
                Continuation::MashTrue {
                    x_head,
                    cond,
                    iffalse,
                    apply_head,
                    env: request.env,
                    location: request.maybe_condition.loc(),
                    depth: request.depth,
                },
            );
        }
        Self::body_done(Err(CompileErr(
            request.maybe_condition.loc(),
            "not i op".to_string(),
        )))
    }

    fn chase_apply(&self, request: ChaseRequest) -> EvalStep {
        if let BodyForm::Call(l, vec, None) = request.body.borrow() {
            if !vec.is_empty() && is_apply_atom(vec[0].to_sexp()) {
                if let Ok(run_program) = dequote(l.clone(), vec[1].clone()) {
                    return self.continue_apply(vec[2].clone(), run_program, request.depth);
                }
                if self.mash_conditions {
                    return Self::request(
                        EvalRequest::Mash(MashRequest {
                            maybe_condition: vec[1].clone(),
                            env: vec[2].clone(),
                            depth: request.depth,
                        }),
                        Continuation::MashOrOriginal {
                            original: request.body.clone(),
                        },
                    );
                }
            }
        }
        Self::body_done(Ok(request.body))
    }

    fn handle_invoke(&self, context: &mut BasicCompileContext, request: InvokeRequest) -> EvalStep {
        let helper = select_helper(&self.helpers, &request.call.name);
        match helper {
            Some(HelperForm::Defmacro(mac)) => {
                if request.call.tail.is_some() {
                    return Self::body_done(Err(CompileErr(
                        request.call.loc,
                        "Macros cannot use runtime rest arguments".to_string(),
                    )));
                }
                self.invoke_macro_expansion(
                    context,
                    mac.loc.clone(),
                    request.call.loc,
                    mac.program,
                    request.prog_args,
                    request.arguments,
                    request.env,
                    request.depth,
                )
            }
            Some(HelperForm::Defun(inline, defun)) => {
                if !inline && request.only_inline {
                    return Self::body_done(Ok(request.call.original));
                }
                let state = DefunState {
                    defun,
                    prog_args: request.prog_args,
                    arguments: request.arguments,
                    env: request.env,
                    only_inline: request.only_inline,
                    depth: request.depth,
                    call_loc: request.call.loc,
                };
                if let Some(tail) = request.call.tail {
                    Self::request_body(
                        ShrinkRequest {
                            prog_args: state.prog_args.clone(),
                            env: state.env.clone(),
                            body: tail,
                            only_inline: state.only_inline,
                            depth: state.depth,
                        },
                        Continuation::DefunTail(state),
                    )
                } else {
                    self.start_defun_captures(state, None)
                }
            }
            _ => Self::request(
                EvalRequest::Primitive(PrimitiveRequest {
                    call: request.call,
                    prog_args: request.prog_args,
                    arguments: request.arguments,
                    env: request.env,
                    only_inline: request.only_inline,
                    depth: request.depth,
                }),
                Continuation::Chase {
                    depth: request.depth,
                },
            ),
        }
    }

    fn start_defun_captures(&self, state: DefunState, tail: Option<Rc<BodyForm>>) -> EvalStep {
        let captures = match build_argument_captures(
            &state.call_loc,
            &state.arguments,
            tail,
            state.defun.args.clone(),
        ) {
            Ok(captures) => captures.into_iter().collect(),
            Err(error) => return Self::body_done(Err(error)),
        };
        self.next_defun_capture(CaptureState {
            defun: state.defun,
            prog_args: state.prog_args,
            env: state.env,
            only_inline: state.only_inline,
            depth: state.depth,
            captures,
            translated: HashMap::new(),
            next: 0,
        })
    }

    fn next_defun_capture(&self, mut state: CaptureState) -> EvalStep {
        if state.next < state.captures.len() {
            let body = state.captures[state.next].1.clone();
            state.next += 1;
            return Self::request_body(
                ShrinkRequest {
                    prog_args: state.prog_args.clone(),
                    env: state.env.clone(),
                    body,
                    only_inline: state.only_inline,
                    depth: state.depth,
                },
                Continuation::DefunCapture(state),
            );
        }
        Self::request_body(
            ShrinkRequest {
                prog_args: state.defun.args.clone(),
                env: Rc::new(state.translated),
                body: state.defun.body,
                only_inline: state.only_inline,
                depth: state.depth,
            },
            Continuation::Identity,
        )
    }

    fn enrich_lambda_site_info(&self, request: EnrichRequest) -> EvalStep {
        if !truthy(request.ldata.capture_args.clone()) {
            return Self::body_done(Ok(Rc::new(BodyForm::Lambda(Box::new(request.ldata)))));
        }
        Self::request_body(
            ShrinkRequest {
                prog_args: request.prog_args.clone(),
                env: request.env.clone(),
                body: request.ldata.captures.clone(),
                only_inline: request.only_inline,
                depth: request.depth,
            },
            Continuation::EnrichCaptures(request),
        )
    }

    fn get_function(&self, name: &[u8]) -> Option<Box<DefunData>> {
        for h in self.helpers.iter() {
            if let HelperForm::Defun(false, dd) = &h {
                if name == h.name() {
                    return Some(dd.clone());
                }
            }
        }

        None
    }

    fn create_mod_for_fun(&self, l: &Srcloc, function: &DefunData) -> Rc<BodyForm> {
        Rc::new(BodyForm::Mod(
            l.clone(),
            CompileForm {
                loc: l.clone(),
                include_forms: Vec::new(),
                args: function.args.clone(),
                helpers: self.helpers.clone(),
                exp: function.body.clone(),
            },
        ))
    }

    fn dispatch(
        &self,
        context: &mut BasicCompileContext,
        request: EvalRequest,
        state: &mut VisitedInfo,
    ) -> EvalStep {
        match request {
            EvalRequest::Shrink(mut request) => {
                request.depth =
                    match Self::increment_depth(state, request.depth, request.body.loc()) {
                        Ok(depth) => depth,
                        Err(error) => return Self::body_done(Err(error)),
                    };
                self.shrink_bodyform_visited(context, request)
            }
            EvalRequest::IsLambda(mut request) => {
                let loc = request
                    .parts
                    .first()
                    .map(|part| part.loc())
                    .unwrap_or_else(|| Srcloc::start("*evaluator*"));
                request.depth = match Self::increment_depth(state, request.depth, loc) {
                    Ok(depth) => depth,
                    Err(error) => {
                        return EvalStep::Complete(Err(error));
                    }
                };
                self.is_lambda_apply(request)
            }
            EvalRequest::Primitive(mut request) => {
                request.depth =
                    match Self::increment_depth(state, request.depth, request.call.loc.clone()) {
                        Ok(depth) => depth,
                        Err(error) => return Self::body_done(Err(error)),
                    };
                self.invoke_primitive(context, request)
            }
            EvalRequest::Lambda(request) => self.do_lambda_apply(request),
            EvalRequest::Invoke(request) => self.handle_invoke(context, request),
            EvalRequest::Chase(request) => self.chase_apply(request),
            EvalRequest::Mash(request) => self.do_mash_condition(request, state),
            EvalRequest::Enrich(request) => self.enrich_lambda_site_info(request),
        }
    }

    fn shrink_bodyform_visited(
        &self,
        _context: &mut BasicCompileContext,
        request: ShrinkRequest,
    ) -> EvalStep {
        let ShrinkRequest {
            prog_args,
            env,
            body,
            only_inline,
            depth,
        } = request;
        match body.borrow() {
            BodyForm::Let(LetFormKind::Parallel, letdata) => {
                if eval_dont_expand_let(&letdata.inline_hint) && only_inline {
                    return Self::body_done(Ok(body.clone()));
                }

                let updated_bindings = update_parallel_bindings(&env, &letdata.bindings);
                Self::request_body(
                    ShrinkRequest {
                        prog_args,
                        env: Rc::new(updated_bindings),
                        body: letdata.body.clone(),
                        only_inline,
                        depth,
                    },
                    Continuation::Identity,
                )
            }
            BodyForm::Let(LetFormKind::Sequential, letdata) => {
                if eval_dont_expand_let(&letdata.inline_hint) && only_inline {
                    return Self::body_done(Ok(body.clone()));
                }

                if letdata.bindings.is_empty() {
                    Self::request_body(
                        ShrinkRequest {
                            prog_args,
                            env,
                            body: letdata.body.clone(),
                            only_inline,
                            depth,
                        },
                        Continuation::Identity,
                    )
                } else {
                    let first_binding_as_list: Vec<Rc<Binding>> =
                        letdata.bindings.iter().take(1).cloned().collect();
                    let rest_of_bindings: Vec<Rc<Binding>> =
                        letdata.bindings.iter().skip(1).cloned().collect();

                    let updated_bindings = update_parallel_bindings(&env, &first_binding_as_list);
                    Self::request_body(
                        ShrinkRequest {
                            prog_args,
                            env: Rc::new(updated_bindings),
                            body: Rc::new(BodyForm::Let(
                                LetFormKind::Sequential,
                                Box::new(LetData {
                                    bindings: rest_of_bindings,
                                    ..*letdata.clone()
                                }),
                            )),
                            only_inline,
                            depth,
                        },
                        Continuation::Identity,
                    )
                }
            }
            BodyForm::Let(LetFormKind::Assign, letdata) => {
                if eval_dont_expand_let(&letdata.inline_hint) && only_inline {
                    return Self::body_done(Ok(body.clone()));
                }

                match hoist_assign_form(letdata) {
                    Ok(hoisted) => Self::request_body(
                        ShrinkRequest {
                            prog_args,
                            env,
                            body: Rc::new(hoisted),
                            only_inline,
                            depth,
                        },
                        Continuation::Identity,
                    ),
                    Err(error) => Self::body_done(Err(error)),
                }
            }
            BodyForm::Quoted(_) => Self::body_done(Ok(body.clone())),
            BodyForm::Value(SExp::Atom(l, name)) => {
                if name == b"@" {
                    match synthesize_args(prog_args.clone(), &env) {
                        Ok(literal_args) => Self::request_body(
                            ShrinkRequest {
                                prog_args,
                                env,
                                body: literal_args,
                                only_inline,
                                depth,
                            },
                            Continuation::Identity,
                        ),
                        Err(error) => Self::body_done(Err(error)),
                    }
                } else if let Some(function) = self.get_function(name) {
                    Self::request_body(
                        ShrinkRequest {
                            prog_args,
                            env,
                            body: self.create_mod_for_fun(l, function.borrow()),
                            only_inline,
                            depth,
                        },
                        Continuation::Identity,
                    )
                } else if let Some(value) = env.get(name) {
                    let value = value.clone();
                    if reflex_capture(name, value.clone()) {
                        Self::body_done(Ok(value))
                    } else {
                        Self::request_body(
                            ShrinkRequest {
                                prog_args,
                                env,
                                body: value,
                                only_inline,
                                depth,
                            },
                            Continuation::Identity,
                        )
                    }
                } else if let Some(constant) = self.get_constant(name) {
                    Self::request_body(
                        ShrinkRequest {
                            prog_args,
                            env,
                            body: constant,
                            only_inline,
                            depth,
                        },
                        Continuation::Identity,
                    )
                } else {
                    Self::body_done(Ok(Rc::new(BodyForm::Value(SExp::Atom(
                        l.clone(),
                        name.clone(),
                    )))))
                }
            }
            BodyForm::Value(v) => Self::body_done(Ok(Rc::new(BodyForm::Quoted(v.clone())))),
            BodyForm::Call(l, parts, tail) => {
                if parts.is_empty() {
                    return Self::body_done(Err(CompileErr(
                        l.clone(),
                        "Impossible empty call list".to_string(),
                    )));
                }

                let head_expr = parts[0].clone();
                let arguments: Vec<Rc<BodyForm>> = parts.iter().skip(1).cloned().collect();

                let call = match head_expr.borrow() {
                    BodyForm::Value(SExp::Atom(_, call_name)) => OwnedCallSpec {
                        loc: l.clone(),
                        name: call_name.clone(),
                        args: parts.clone(),
                        original: body.clone(),
                        tail: tail.clone(),
                    },
                    BodyForm::Value(SExp::Integer(_, call_int)) => OwnedCallSpec {
                        loc: l.clone(),
                        name: u8_from_number(call_int.clone()),
                        args: parts.clone(),
                        original: body.clone(),
                        tail: None,
                    },
                    _ => {
                        return Self::body_done(Err(CompileErr(
                            l.clone(),
                            format!("Don't know how to call {}", head_expr.to_sexp()),
                        )))
                    }
                };
                Self::request(
                    EvalRequest::Invoke(InvokeRequest {
                        call,
                        prog_args,
                        arguments,
                        env,
                        only_inline,
                        depth,
                    }),
                    Continuation::Identity,
                )
            }
            BodyForm::Mod(l, program) => {
                let mut symbols = HashMap::new();
                let optimizer = match get_optimizer(l, self.opts.clone()) {
                    Ok(optimizer) => optimizer,
                    Err(error) => return Self::body_done(Err(error)),
                };
                let mut context_wrapper =
                    CompileContextWrapper::new(self.runner.clone(), &mut symbols, optimizer);
                Self::body_done(
                    codegen(context_wrapper.context(), self.opts.clone(), program)
                        .map(|code| Rc::new(BodyForm::Quoted(code))),
                )
            }
            BodyForm::Lambda(ldata) => Self::request(
                EvalRequest::Enrich(EnrichRequest {
                    prog_args,
                    env,
                    ldata: *ldata.clone(),
                    only_inline,
                    depth,
                }),
                Continuation::Identity,
            ),
        }
    }

    fn resume(
        &self,
        context: &mut BasicCompileContext,
        continuation: Continuation,
        result: EvalResult,
    ) -> EvalStep {
        match continuation {
            Continuation::Identity => EvalStep::Complete(result),
            Continuation::IsLambdaProgram(request) => {
                let evaluated_prog = match Self::body_result(result, request.parts[1].loc()) {
                    Ok(body) => body,
                    Err(error) => return EvalStep::Complete(Err(error)),
                };
                Self::request_body(
                    ShrinkRequest {
                        prog_args: request.prog_args.clone(),
                        env: request.env.clone(),
                        body: request.parts[2].clone(),
                        only_inline: request.only_inline,
                        depth: request.depth,
                    },
                    Continuation::IsLambdaEnv {
                        evaluated_prog,
                        request,
                    },
                )
            }
            Continuation::IsLambdaEnv {
                evaluated_prog,
                request,
            } => {
                let evaluated_env = match Self::body_result(result, request.parts[2].loc()) {
                    Ok(body) => body,
                    Err(error) => return EvalStep::Complete(Err(error)),
                };
                let applied = match evaluated_prog.borrow() {
                    BodyForm::Lambda(ldata) => Some(LambdaApply {
                        lambda: *ldata.clone(),
                        body: ldata.body.clone(),
                        env: evaluated_env,
                    }),
                    _ => None,
                };
                EvalStep::Complete(Ok(EvalValue::Lambda(applied)))
            }
            Continuation::LambdaCaptures(request) => {
                let reified = match Self::body_result(result, request.lapply.lambda.captures.loc())
                {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                let mut lambda_env = (*request.env).clone();
                if let Err(error) = create_argument_captures(
                    &mut lambda_env,
                    &ArgInputs::Whole(reified),
                    request.lapply.lambda.capture_args.clone(),
                ) {
                    return Self::body_done(Err(error));
                }
                if let Err(error) = create_argument_captures(
                    &mut lambda_env,
                    &ArgInputs::Whole(request.lapply.env.clone()),
                    request.lapply.lambda.args.clone(),
                ) {
                    return Self::body_done(Err(error));
                }
                Self::request_body(
                    ShrinkRequest {
                        prog_args: request.lapply.lambda.args,
                        env: Rc::new(lambda_env),
                        body: request.lapply.body,
                        only_inline: request.only_inline,
                        depth: request.depth,
                    },
                    Continuation::Identity,
                )
            }
            Continuation::PrimitiveArg(mut state) => {
                let index = state.next;
                let shrunk = match Self::body_result(result, state.arguments[index].loc()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                state.target[index + 1] = shrunk.clone();
                state.converted[index] = Some(shrunk.to_sexp());
                state.all_primitive &= arg_inputs_primitive(Rc::new(ArgInputs::Whole(shrunk)));
                self.next_primitive_argument(context, state)
            }
            Continuation::PrimitiveLambda(state) => match result {
                Ok(EvalValue::Lambda(Some(applied))) => Self::request(
                    EvalRequest::Lambda(LambdaRequest {
                        prog_args: state.prog_args,
                        env: state.env,
                        lapply: applied,
                        only_inline: state.only_inline,
                        depth: state.depth,
                    }),
                    Continuation::Identity,
                ),
                Ok(EvalValue::Lambda(None)) => Self::request(
                    EvalRequest::Chase(ChaseRequest {
                        body: Rc::new(BodyForm::Call(
                            state.call.loc,
                            state.target,
                            state.call.tail,
                        )),
                        depth: state.depth,
                    }),
                    Continuation::Identity,
                ),
                Ok(EvalValue::Body(_)) => Self::body_done(Err(CompileErr(
                    state.call.loc,
                    "internal evaluator return type mismatch".to_string(),
                ))),
                Err(error) => Self::body_done(Err(error)),
            },
            Continuation::Chase { depth } => match result {
                Ok(EvalValue::Body(body)) => Self::request(
                    EvalRequest::Chase(ChaseRequest { body, depth }),
                    Continuation::Identity,
                ),
                other => EvalStep::Complete(other),
            },
            Continuation::ContinueApply { depth } => match result {
                Ok(EvalValue::Body(body)) => Self::request(
                    EvalRequest::Chase(ChaseRequest { body, depth }),
                    Continuation::Identity,
                ),
                other => EvalStep::Complete(other),
            },
            Continuation::MashOrOriginal { original } => match result {
                Ok(value) => EvalStep::Complete(Ok(value)),
                Err(_) => Self::body_done(Ok(original)),
            },
            Continuation::MashTrue {
                x_head,
                cond,
                iffalse,
                apply_head,
                env,
                location,
                depth,
            } => {
                let true_result = match Self::body_result(result, location.clone()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                Self::request(
                    EvalRequest::Chase(ChaseRequest {
                        body: Rc::new(BodyForm::Call(
                            iffalse.loc(),
                            vec![apply_head, iffalse, env],
                            None,
                        )),
                        depth,
                    }),
                    Continuation::MashFalse {
                        x_head,
                        cond,
                        true_result,
                        location,
                    },
                )
            }
            Continuation::MashFalse {
                x_head,
                cond,
                true_result,
                location,
            } => {
                let false_result = match Self::body_result(result, location.clone()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                Self::body_done(Ok(Rc::new(BodyForm::Call(
                    location,
                    vec![
                        x_head,
                        flatten_expression_to_names(cond.to_sexp()),
                        flatten_expression_to_names(true_result.to_sexp()),
                        flatten_expression_to_names(false_result.to_sexp()),
                    ],
                    None,
                ))))
            }
            Continuation::DefunTail(state) => {
                let tail = match Self::body_result(result, state.call_loc.clone()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                self.start_defun_captures(state, Some(tail))
            }
            Continuation::DefunCapture(mut state) => {
                let index = state.next - 1;
                let shrunk = match Self::body_result(result, state.captures[index].1.loc()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                state
                    .translated
                    .insert(state.captures[index].0.clone(), shrunk);
                self.next_defun_capture(state)
            }
            Continuation::EnrichCaptures(request) => {
                let new_captures = match Self::body_result(result, request.ldata.captures.loc()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                let mut arg_captures = HashMap::new();
                if let Err(error) = create_argument_captures(
                    &mut arg_captures,
                    &decons_args(new_captures.clone()),
                    request.ldata.capture_args.clone(),
                ) {
                    return Self::body_done(Err(error));
                }
                let interpretable: HashMap<_, _> = arg_captures
                    .into_iter()
                    .filter(|(_, value)| dequote(value.loc(), value.clone()).is_ok())
                    .collect();
                let combined_args = Rc::new(SExp::Cons(
                    request.ldata.loc.clone(),
                    request.ldata.capture_args.clone(),
                    request.ldata.args.clone(),
                ));
                Self::request_body(
                    ShrinkRequest {
                        prog_args: combined_args,
                        env: Rc::new(interpretable.clone()),
                        body: request.ldata.body.clone(),
                        only_inline: request.only_inline,
                        depth: request.depth,
                    },
                    Continuation::EnrichBody {
                        ldata: request.ldata,
                        new_captures,
                        interpretable,
                    },
                )
            }
            Continuation::EnrichBody {
                ldata,
                new_captures,
                interpretable,
            } => {
                let simplified = match Self::body_result(result, ldata.body.loc()) {
                    Ok(body) => body,
                    Err(error) => return Self::body_done(Err(error)),
                };
                let new_capture_args =
                    filter_capture_args(ldata.capture_args.clone(), &interpretable);
                Self::body_done(Ok(Rc::new(BodyForm::Lambda(Box::new(LambdaData {
                    args: ldata.args.clone(),
                    capture_args: new_capture_args,
                    captures: new_captures,
                    body: simplified,
                    ..ldata
                })))))
            }
        }
    }

    /// The main entrypoint for the evaluator, shrink_bodyform takes a notion of the
    /// current argument set (in case something depends on its shape), the
    /// bindings in force, and a frontend expression to evaluate and simplifies
    /// it as much as possible.  The result is the "least complex" version of the
    /// expression we can make with what we know; this includes taking any part that's
    /// constant and fully applying it to make a constant of the full subexpression
    /// as well as a few other small rewriting elements.
    ///
    /// There are a few simplification steps that may make code larger, such as
    /// fully substituting inline applications and eliminating let bindings.
    ///
    /// the only_inline flag controls whether only inline functions are expanded
    /// or whether it's allowed to expand all functions, depending on whehter it's
    /// intended to simply make a result that ends at inline expansion or generate
    /// as full a result as possible.
    pub fn shrink_bodyform(
        &self,
        context: &mut BasicCompileContext,
        prog_args: Rc<SExp>,
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        body: Rc<BodyForm>,
        only_inline: bool,
        stack_limit: Option<usize>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let mut state = VisitedInfo {
            max_depth: stack_limit,
            ..Default::default()
        };
        let mut continuations = Vec::new();
        let mut step = self.dispatch(
            context,
            EvalRequest::Shrink(ShrinkRequest {
                prog_args,
                env: Rc::new(env.clone()),
                body,
                only_inline,
                depth: 1,
            }),
            &mut state,
        );
        loop {
            step = match step {
                EvalStep::Request(request, continuation) => {
                    continuations.push(continuation);
                    self.dispatch(context, *request, &mut state)
                }
                EvalStep::Complete(result) => {
                    if let Some(continuation) = continuations.pop() {
                        self.resume(context, *continuation, result)
                    } else {
                        return Self::body_result(result, Srcloc::start("*evaluator*"));
                    }
                }
            };
        }
    }

    fn expand_macro(
        &self,
        context: &mut BasicCompileContext,
        call_loc: Srcloc,
        program: Rc<CompileForm>,
        args: Rc<SExp>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let mut new_helpers = Vec::new();
        let mut used_names = HashSet::new();

        let mut end_of_list = Rc::new(SExp::Cons(
            call_loc.clone(),
            program.exp.to_sexp(),
            Rc::new(SExp::Nil(call_loc.clone())),
        ));

        for h in program.helpers.iter() {
            new_helpers.push(h.clone());
            used_names.insert(h.name());
        }

        for h in self.helpers.iter() {
            if !used_names.contains(h.name()) {
                new_helpers.push(h.clone());
            }
        }

        for h in new_helpers.iter() {
            end_of_list = Rc::new(SExp::Cons(call_loc.clone(), h.to_sexp(), end_of_list))
        }

        let use_body = Rc::new(SExp::Cons(
            call_loc.clone(),
            Rc::new(SExp::Atom(call_loc.clone(), "mod".as_bytes().to_vec())),
            Rc::new(SExp::Cons(
                call_loc.clone(),
                program.args.clone(),
                end_of_list,
            )),
        ));

        let compiled = self.compile_code(context, false, use_body)?;
        self.run_prim(context.allocator(), call_loc, compiled, args)
    }

    fn lookup_prim(&self, l: Srcloc, name: &[u8]) -> Option<Rc<SExp>> {
        match self.prims.get(name) {
            Some(p) => Some(p.clone()),
            None => {
                if name.len() == 1 {
                    Some(Rc::new(SExp::Atom(l, name.to_owned())))
                } else {
                    None
                }
            }
        }
    }

    fn run_prim(
        &self,
        allocator: &mut Allocator,
        call_loc: Srcloc,
        prim: Rc<SExp>,
        args: Rc<SExp>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        run(
            allocator,
            self.runner.clone(),
            self.prims.clone(),
            prim,
            args,
            None,
            Some(PRIM_RUN_LIMIT),
        )
        .map_err(|e| match e {
            RunFailure::RunExn(_, s) => CompileErr(call_loc.clone(), format!("exception: {s}")),
            RunFailure::RunErr(_, s) => CompileErr(call_loc.clone(), s),
        })
        .map(|res| {
            let res_borrowed: &SExp = res.borrow();
            Rc::new(BodyForm::Quoted(res_borrowed.clone()))
        })
    }

    fn compile_code(
        &self,
        context: &mut BasicCompileContext,
        in_defun: bool,
        use_body: Rc<SExp>,
    ) -> Result<Rc<SExp>, CompileErr> {
        // Com takes place in the current environment.
        // We can only reduce com if all bindings are
        // primitive.
        let updated_opts = self
            .opts
            .set_stdenv(!in_defun)
            .set_in_defun(in_defun)
            .set_frontend_opt(false);

        let com_result = updated_opts.compile_program(context, use_body)?;

        Ok(Rc::new(com_result.to_sexp()))
    }

    pub fn add_helper(&mut self, h: &HelperForm) {
        for i in 0..self.helpers.len() {
            if self.helpers[i].name() == h.name() {
                self.helpers[i] = h.clone();
                return;
            }
        }
        self.helpers.push(h.clone());
    }

    // The evaluator treats the forms coming up from constants as live.
    fn get_constant(&self, name: &[u8]) -> Option<Rc<BodyForm>> {
        for h in self.helpers.iter() {
            if let HelperForm::Defconstant(defc) = h {
                if defc.name == name {
                    return Some(defc.body.clone());
                }
            }
        }
        None
    }
}
