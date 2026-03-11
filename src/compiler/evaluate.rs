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
    Binding, BindingPattern, BodyForm, CallSpec, CompileErr, CompileForm, CompilerOpts, DefunData,
    HelperForm, LambdaData, LetData, LetFormInlineHint, LetFormKind,
};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::stackvisit::{HasDepthLimit, VisitedMarker};
use crate::compiler::CompileContextWrapper;
use crate::util::{number_from_u8, u8_from_number, Number};

const PRIM_RUN_LIMIT: usize = 1000000;
pub const EVAL_STACK_LIMIT: usize = 200;

// Stack depth checker.
#[derive(Clone, Debug, Default)]
pub struct VisitedInfo {
    functions: HashMap<Vec<u8>, Rc<BodyForm>>,
    max_depth: Option<usize>,
}

impl HasDepthLimit<Srcloc, CompileErr> for VisitedInfo {
    fn depth_limit(&self) -> Option<usize> {
        self.max_depth
    }
    fn stack_err(&self, loc: Srcloc) -> CompileErr {
        CompileErr(loc, "stack limit exceeded".to_string())
    }
}

trait VisitedInfoAccess {
    fn get_function(&mut self, name: &[u8]) -> Option<Rc<BodyForm>>;
    fn insert_function(&mut self, name: Vec<u8>, body: Rc<BodyForm>);
}

impl VisitedInfoAccess for VisitedMarker<'_, VisitedInfo> {
    fn get_function(&mut self, name: &[u8]) -> Option<Rc<BodyForm>> {
        if let Some(ref mut info) = self.info {
            info.functions.get(name).cloned()
        } else {
            None
        }
    }

    fn insert_function(&mut self, name: Vec<u8>, body: Rc<BodyForm>) {
        if let Some(ref mut info) = self.info {
            info.functions.insert(name, body);
        }
    }
}

pub struct LambdaApply {
    lambda: LambdaData,
    body: Rc<BodyForm>,
    env: Rc<BodyForm>,
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

impl<'info> Evaluator {
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

    #[allow(clippy::too_many_arguments)]
    fn invoke_macro_expansion(
        &self,
        allocator: &mut Allocator,
        visited: &'_ mut VisitedMarker<'info, VisitedInfo>,
        l: Srcloc,
        call_loc: Srcloc,
        program: Rc<CompileForm>,
        prog_args: Rc<SExp>,
        arguments_to_convert: &[Rc<BodyForm>],
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        // Pass the SExp representation of the expressions into
        // the macro after forming an argument sexp and then
        let mut macro_args = Rc::new(SExp::Nil(l.clone()));
        for i_reverse in 0..arguments_to_convert.len() {
            let i = arguments_to_convert.len() - i_reverse - 1;
            let arg_repr = arguments_to_convert[i].to_sexp();
            macro_args = Rc::new(SExp::Cons(l.clone(), arg_repr, macro_args));
        }

        let macro_expansion = self.expand_macro(allocator, l.clone(), program, macro_args)?;

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

            frontend(self.opts.clone(), &[frontend_macro_input]).and_then(|program| {
                self.shrink_bodyform_visited(
                    allocator,
                    visited,
                    prog_args.clone(),
                    env,
                    program.exp,
                    false,
                )
            })
        } else {
            promote_program_to_bodyform(
                macro_expansion.to_sexp(),
                Rc::new(BodyForm::Value(SExp::Atom(
                    macro_expansion.loc(),
                    vec![b'@'],
                ))),
            )
        }
    }

    fn is_lambda_apply(
        &self,
        allocator: &mut Allocator,
        visited_: &'info mut VisitedMarker<'_, VisitedInfo>,
        prog_args: Rc<SExp>,
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        parts: &[Rc<BodyForm>],
        only_inline: bool,
    ) -> Result<Option<LambdaApply>, CompileErr> {
        if parts.len() == 3 && is_apply_atom(parts[0].to_sexp()) {
            let mut visited = VisitedMarker::again(parts[0].loc(), visited_)?;
            let evaluated_prog = self.shrink_bodyform_visited(
                allocator,
                &mut visited,
                prog_args.clone(),
                env,
                parts[1].clone(),
                only_inline,
            )?;
            let evaluated_env = self.shrink_bodyform_visited(
                allocator,
                &mut visited,
                prog_args,
                env,
                parts[2].clone(),
                only_inline,
            )?;
            if let BodyForm::Lambda(ldata) = evaluated_prog.borrow() {
                return Ok(Some(LambdaApply {
                    lambda: *ldata.clone(),
                    body: ldata.body.clone(),
                    env: evaluated_env,
                }));
            }
        }

        Ok(None)
    }

    fn do_lambda_apply(
        &self,
        allocator: &mut Allocator,
        visited: &mut VisitedMarker<'info, VisitedInfo>,
        prog_args: Rc<SExp>,
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        lapply: &LambdaApply,
        only_inline: bool,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let mut lambda_env = env.clone();

        // Finish eta-expansion.

        // We're carrying an enriched environment which we can use to enrich
        // the env map at this time.  Once we do that we can expand the body
        // fully because we're carring the info that goes with the primary
        // arguments.
        //
        // Generate the enriched environment.
        let reified_captures = self.shrink_bodyform_visited(
            allocator,
            visited,
            prog_args,
            env,
            lapply.lambda.captures.clone(),
            only_inline,
        )?;
        let formed_caps = ArgInputs::Whole(reified_captures);
        create_argument_captures(
            &mut lambda_env,
            &formed_caps,
            lapply.lambda.capture_args.clone(),
        )?;

        // Create captures with the actual parameters.
        let formed_args = ArgInputs::Whole(lapply.env.clone());
        create_argument_captures(&mut lambda_env, &formed_args, lapply.lambda.args.clone())?;

        self.shrink_bodyform_visited(
            allocator,
            visited,
            lapply.lambda.args.clone(),
            &lambda_env,
            lapply.body.clone(),
            only_inline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn invoke_primitive(
        &self,
        allocator: &mut Allocator,
        visited_: &'_ mut VisitedMarker<'info, VisitedInfo>,
        call: &CallSpec,
        prog_args: Rc<SExp>,
        arguments_to_convert: &[Rc<BodyForm>],
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        only_inline: bool,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let mut all_primitive = true;
        let mut target_vec: Vec<Rc<BodyForm>> = call.args.to_owned();
        let mut visited = VisitedMarker::again(call.loc.clone(), visited_)?;

        if call.name == "@".as_bytes() {
            // Synthesize the environment for this function
            Ok(Rc::new(BodyForm::Quoted(SExp::Cons(
                call.loc.clone(),
                Rc::new(SExp::Nil(call.loc.clone())),
                prog_args,
            ))))
        } else if call.name == "com".as_bytes() {
            let mut end_of_list = Rc::new(SExp::Cons(
                call.loc.clone(),
                arguments_to_convert[0].to_sexp(),
                Rc::new(SExp::Nil(call.loc.clone())),
            ));

            for h in self.helpers.iter() {
                end_of_list = Rc::new(SExp::Cons(call.loc.clone(), h.to_sexp(), end_of_list))
            }

            let use_body = SExp::Cons(
                call.loc.clone(),
                Rc::new(SExp::Atom(call.loc.clone(), "mod".as_bytes().to_vec())),
                Rc::new(SExp::Cons(call.loc.clone(), prog_args, end_of_list)),
            );

            let compiled = self.compile_code(allocator, false, Rc::new(use_body))?;
            let compiled_borrowed: &SExp = compiled.borrow();
            Ok(Rc::new(BodyForm::Quoted(compiled_borrowed.clone())))
        } else {
            let pres = self
                .lookup_prim(call.loc.clone(), call.name)
                .map(|prim| {
                    // Reduce all arguments.
                    let mut converted_args = SExp::Nil(call.loc.clone());

                    for i_reverse in 0..arguments_to_convert.len() {
                        let i = arguments_to_convert.len() - i_reverse - 1;
                        let shrunk = self.shrink_bodyform_visited(
                            allocator,
                            &mut visited,
                            prog_args.clone(),
                            env,
                            arguments_to_convert[i].clone(),
                            only_inline,
                        )?;

                        target_vec[i + 1] = shrunk.clone();

                        if !arg_inputs_primitive(Rc::new(ArgInputs::Whole(shrunk.clone()))) {
                            all_primitive = false;
                        }

                        converted_args =
                            SExp::Cons(call.loc.clone(), shrunk.to_sexp(), Rc::new(converted_args));
                    }

                    if all_primitive {
                        match self.run_prim(
                            allocator,
                            call.loc.clone(),
                            make_prim_call(call.loc.clone(), prim, Rc::new(converted_args)),
                            Rc::new(SExp::Nil(call.loc.clone())),
                        ) {
                            Ok(res) => Ok(res),
                            Err(e) => {
                                if only_inline || self.ignore_exn {
                                    Ok(Rc::new(BodyForm::Call(
                                        call.loc.clone(),
                                        target_vec.clone(),
                                        None,
                                    )))
                                } else {
                                    Err(e)
                                }
                            }
                        }
                    } else if let Some(applied_lambda) = self.is_lambda_apply(
                        allocator,
                        &mut visited,
                        prog_args.clone(),
                        env,
                        &target_vec,
                        only_inline,
                    )? {
                        self.do_lambda_apply(
                            allocator,
                            &mut visited,
                            prog_args.clone(),
                            env,
                            &applied_lambda,
                            only_inline,
                        )
                    } else {
                        // Since this is a primitive, there's no tail transform.
                        let reformed =
                            BodyForm::Call(call.loc.clone(), target_vec.clone(), call.tail.clone());
                        self.chase_apply(allocator, &mut visited, Rc::new(reformed))
                    }
                })
                .unwrap_or_else(|| {
                    // Build SExp arguments for external call or
                    // return the unevaluated chunk with minimized
                    // arguments.
                    Err(CompileErr(
                        call.loc.clone(),
                        format!(
                            "Don't yet support this call type {} {:?}",
                            call.original.to_sexp(),
                            call.original
                        ),
                    ))
                })?;
            Ok(pres)
        }
    }

    fn continue_apply(
        &self,
        allocator: &mut Allocator,
        visited: &'_ mut VisitedMarker<'info, VisitedInfo>,
        env: Rc<BodyForm>,
        run_program: Rc<SExp>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let bindings = HashMap::new();
        let program = promote_program_to_bodyform(run_program.clone(), env)?;
        let apply_result = self.shrink_bodyform_visited(
            allocator,
            visited,
            Rc::new(SExp::Nil(run_program.loc())),
            &bindings,
            program,
            false,
        )?;
        self.chase_apply(allocator, visited, apply_result)
    }

    fn do_mash_condition(
        &self,
        allocator: &mut Allocator,
        visited: &'_ mut VisitedMarker<'info, VisitedInfo>,
        maybe_condition: Rc<BodyForm>,
        env: Rc<BodyForm>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        // The inner part could be an 'i' which we know passes on
        // one of the two conditional arguments.  This was an apply so
        // we can distribute over the conditional arguments.
        if let Some((cond, iftrue, iffalse)) = match_i_op(maybe_condition.clone()) {
            let x_head = Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), vec![b'x'])));
            let apply_head = Rc::new(BodyForm::Value(SExp::Atom(iftrue.loc(), vec![2])));
            let where_from = cond.loc().to_string();
            let where_from_vec = where_from.as_bytes().to_vec();

            if let Some(present) = visited.get_function(&where_from_vec) {
                return Ok(present);
            }

            visited.insert_function(
                where_from_vec,
                Rc::new(BodyForm::Call(
                    maybe_condition.loc(),
                    vec![x_head.clone(), cond.clone()],
                    None,
                )),
            );

            let surrogate_apply_true = self.chase_apply(
                allocator,
                visited,
                Rc::new(BodyForm::Call(
                    iftrue.loc(),
                    vec![apply_head.clone(), iftrue.clone(), env.clone()],
                    None,
                )),
            );

            let surrogate_apply_false = self.chase_apply(
                allocator,
                visited,
                Rc::new(BodyForm::Call(
                    iffalse.loc(),
                    vec![apply_head, iffalse.clone(), env],
                    None,
                )),
            );

            // Reproduce the equivalent hull over the used values of
            // (a (i cond surrogate_apply_true surrogate_apply_false))
            // Flatten and short circuit any farther evaluation since we just
            // want the argument names passed through from the environment.
            let res = Rc::new(BodyForm::Call(
                maybe_condition.loc(),
                vec![
                    x_head,
                    flatten_expression_to_names(cond.to_sexp()),
                    flatten_expression_to_names(surrogate_apply_true?.to_sexp()),
                    flatten_expression_to_names(surrogate_apply_false?.to_sexp()),
                ],
                None,
            ));

            return Ok(res);
        }

        Err(CompileErr(maybe_condition.loc(), "not i op".to_string()))
    }

    fn chase_apply(
        &self,
        allocator: &mut Allocator,
        visited: &'_ mut VisitedMarker<'info, VisitedInfo>,
        body: Rc<BodyForm>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        if let BodyForm::Call(l, vec, None) = body.borrow() {
            if is_apply_atom(vec[0].to_sexp()) {
                if let Ok(run_program) = dequote(l.clone(), vec[1].clone()) {
                    return self.continue_apply(allocator, visited, vec[2].clone(), run_program);
                }

                if self.mash_conditions {
                    if let Ok(mashed) =
                        self.do_mash_condition(allocator, visited, vec[1].clone(), vec[2].clone())
                    {
                        return Ok(mashed);
                    }
                }
            }
        }

        Ok(body)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_invoke(
        &self,
        allocator: &mut Allocator,
        visited: &'_ mut VisitedMarker<'info, VisitedInfo>,
        call: &CallSpec,
        prog_args: Rc<SExp>,
        arguments_to_convert: &[Rc<BodyForm>],
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        only_inline: bool,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let helper = select_helper(&self.helpers, call.name);
        match helper {
            Some(HelperForm::Defmacro(mac)) => {
                if call.tail.is_some() {
                    return Err(CompileErr(
                        call.loc.clone(),
                        "Macros cannot use runtime rest arguments".to_string(),
                    ));
                }
                self.invoke_macro_expansion(
                    allocator,
                    visited,
                    mac.loc.clone(),
                    call.loc.clone(),
                    mac.program,
                    prog_args,
                    arguments_to_convert,
                    env,
                )
            }
            Some(HelperForm::Defun(inline, defun)) => {
                if !inline && only_inline {
                    return Ok(call.original.clone());
                }

                let translated_tail = if let Some(t) = call.tail.as_ref() {
                    Some(self.shrink_bodyform_visited(
                        allocator,
                        visited,
                        prog_args.clone(),
                        env,
                        t.clone(),
                        only_inline,
                    )?)
                } else {
                    None
                };

                let argument_captures_untranslated = build_argument_captures(
                    &call.loc.clone(),
                    arguments_to_convert,
                    translated_tail.clone(),
                    defun.args.clone(),
                )?;

                let mut argument_captures = HashMap::new();
                // Do this to protect against misalignment
                // between argument vec and destructuring.
                for kv in argument_captures_untranslated.iter() {
                    let shrunk = self.shrink_bodyform_visited(
                        allocator,
                        visited,
                        prog_args.clone(),
                        env,
                        kv.1.clone(),
                        only_inline,
                    )?;

                    argument_captures.insert(kv.0.clone(), shrunk.clone());
                }

                self.shrink_bodyform_visited(
                    allocator,
                    visited,
                    defun.args.clone(),
                    &argument_captures,
                    defun.body,
                    only_inline,
                )
            }
            _ => self
                .invoke_primitive(
                    allocator,
                    visited,
                    call,
                    prog_args,
                    arguments_to_convert,
                    env,
                    only_inline,
                )
                .and_then(|res| self.chase_apply(allocator, visited, res)),
        }
    }

    fn enrich_lambda_site_info(
        &self,
        allocator: &mut Allocator,
        visited: &'info mut VisitedMarker<'_, VisitedInfo>,
        prog_args: Rc<SExp>,
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        ldata: &LambdaData,
        only_inline: bool,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        if !truthy(ldata.capture_args.clone()) {
            return Ok(Rc::new(BodyForm::Lambda(Box::new(ldata.clone()))));
        }

        // Rewrite the captures based on what we know at the call site.
        let new_captures = self.shrink_bodyform_visited(
            allocator,
            visited,
            prog_args.clone(),
            env,
            ldata.captures.clone(),
            only_inline,
        )?;

        // Break up and make binding map.
        let deconsed_args = decons_args(new_captures.clone());
        let mut arg_captures = HashMap::new();
        create_argument_captures(
            &mut arg_captures,
            &deconsed_args,
            ldata.capture_args.clone(),
        )?;

        // Filter out elements that are not interpretable yet.
        let mut interpretable_captures = HashMap::new();
        for (n, v) in arg_captures.iter() {
            if dequote(v.loc(), v.clone()).is_ok() {
                // This capture has already been made into a literal.
                // We will substitute it in the lambda body and remove it
                // from the capture set.
                interpretable_captures.insert(n.clone(), v.clone());
            }
        }

        let combined_args = Rc::new(SExp::Cons(
            ldata.loc.clone(),
            ldata.capture_args.clone(),
            ldata.args.clone(),
        ));

        // Eliminate the captures via beta substituion.
        let simplified_body = self.shrink_bodyform_visited(
            allocator,
            visited,
            combined_args.clone(),
            &interpretable_captures,
            ldata.body.clone(),
            only_inline,
        )?;

        let new_capture_args =
            filter_capture_args(ldata.capture_args.clone(), &interpretable_captures);
        Ok(Rc::new(BodyForm::Lambda(Box::new(LambdaData {
            args: ldata.args.clone(),
            capture_args: new_capture_args,
            captures: new_captures,
            body: simplified_body,
            ..ldata.clone()
        }))))
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

    // A frontend language evaluator and minifier
    fn shrink_bodyform_visited(
        &self,
        allocator: &mut Allocator, // Support random prims via clvm_rs
        visited_: &'info mut VisitedMarker<'_, VisitedInfo>,
        prog_args: Rc<SExp>,
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        body: Rc<BodyForm>,
        only_inline: bool,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let mut visited = VisitedMarker::again(body.loc(), visited_)?;
        match body.borrow() {
            BodyForm::Let(LetFormKind::Parallel, letdata) => {
                if eval_dont_expand_let(&letdata.inline_hint) && only_inline {
                    return Ok(body.clone());
                }

                let updated_bindings = update_parallel_bindings(env, &letdata.bindings);
                self.shrink_bodyform_visited(
                    allocator,
                    &mut visited,
                    prog_args,
                    &updated_bindings,
                    letdata.body.clone(),
                    only_inline,
                )
            }
            BodyForm::Let(LetFormKind::Sequential, letdata) => {
                if eval_dont_expand_let(&letdata.inline_hint) && only_inline {
                    return Ok(body.clone());
                }

                if letdata.bindings.is_empty() {
                    self.shrink_bodyform_visited(
                        allocator,
                        &mut visited,
                        prog_args,
                        env,
                        letdata.body.clone(),
                        only_inline,
                    )
                } else {
                    let first_binding_as_list: Vec<Rc<Binding>> =
                        letdata.bindings.iter().take(1).cloned().collect();
                    let rest_of_bindings: Vec<Rc<Binding>> =
                        letdata.bindings.iter().skip(1).cloned().collect();

                    let updated_bindings = update_parallel_bindings(env, &first_binding_as_list);
                    self.shrink_bodyform_visited(
                        allocator,
                        &mut visited,
                        prog_args,
                        &updated_bindings,
                        Rc::new(BodyForm::Let(
                            LetFormKind::Sequential,
                            Box::new(LetData {
                                bindings: rest_of_bindings,
                                ..*letdata.clone()
                            }),
                        )),
                        only_inline,
                    )
                }
            }
            BodyForm::Let(LetFormKind::Assign, letdata) => {
                if eval_dont_expand_let(&letdata.inline_hint) && only_inline {
                    return Ok(body.clone());
                }

                self.shrink_bodyform_visited(
                    allocator,
                    &mut visited,
                    prog_args,
                    env,
                    Rc::new(hoist_assign_form(letdata)?),
                    only_inline,
                )
            }
            BodyForm::Quoted(_) => Ok(body.clone()),
            BodyForm::Value(SExp::Atom(l, name)) => {
                if name == &"@".as_bytes().to_vec() {
                    let literal_args = synthesize_args(prog_args.clone(), env)?;
                    self.shrink_bodyform_visited(
                        allocator,
                        &mut visited,
                        prog_args,
                        env,
                        literal_args,
                        only_inline,
                    )
                } else if let Some(function) = self.get_function(name) {
                    self.shrink_bodyform_visited(
                        allocator,
                        &mut visited,
                        prog_args,
                        env,
                        self.create_mod_for_fun(l, function.borrow()),
                        only_inline,
                    )
                } else {
                    env.get(name)
                        .map(|x| {
                            if reflex_capture(name, x.clone()) {
                                Ok(x.clone())
                            } else {
                                self.shrink_bodyform_visited(
                                    allocator,
                                    &mut visited,
                                    prog_args.clone(),
                                    env,
                                    x.clone(),
                                    only_inline,
                                )
                            }
                        })
                        .unwrap_or_else(|| {
                            self.get_constant(name)
                                .map(|x| {
                                    self.shrink_bodyform_visited(
                                        allocator,
                                        &mut visited,
                                        prog_args.clone(),
                                        env,
                                        x,
                                        only_inline,
                                    )
                                })
                                .unwrap_or_else(|| {
                                    Ok(Rc::new(BodyForm::Value(SExp::Atom(
                                        l.clone(),
                                        name.clone(),
                                    ))))
                                })
                        })
                }
            }
            BodyForm::Value(v) => Ok(Rc::new(BodyForm::Quoted(v.clone()))),
            BodyForm::Call(l, parts, tail) => {
                if parts.is_empty() {
                    return Err(CompileErr(
                        l.clone(),
                        "Impossible empty call list".to_string(),
                    ));
                }

                let head_expr = parts[0].clone();
                let arguments_to_convert: Vec<Rc<BodyForm>> =
                    parts.iter().skip(1).cloned().collect();

                match head_expr.borrow() {
                    BodyForm::Value(SExp::Atom(_call_loc, call_name)) => self.handle_invoke(
                        allocator,
                        &mut visited,
                        &CallSpec {
                            loc: l.clone(),
                            name: call_name,
                            args: parts,
                            original: body.clone(),
                            tail: tail.clone(),
                        },
                        prog_args,
                        &arguments_to_convert,
                        env,
                        only_inline,
                    ),
                    BodyForm::Value(SExp::Integer(_call_loc, call_int)) => self.handle_invoke(
                        allocator,
                        &mut visited,
                        &CallSpec {
                            loc: l.clone(),
                            name: &u8_from_number(call_int.clone()),
                            args: parts,
                            original: body.clone(),
                            tail: None,
                        },
                        prog_args,
                        &arguments_to_convert,
                        env,
                        only_inline,
                    ),
                    _ => Err(CompileErr(
                        l.clone(),
                        format!("Don't know how to call {}", head_expr.to_sexp()),
                    )),
                }
            }
            BodyForm::Mod(l, program) => {
                // A mod form yields the compiled code.
                let mut symbols = HashMap::new();
                let optimizer = get_optimizer(l, self.opts.clone())?;
                let mut context_wrapper = CompileContextWrapper::new(
                    allocator,
                    self.runner.clone(),
                    &mut symbols,
                    optimizer,
                );
                let code = codegen(&mut context_wrapper.context, self.opts.clone(), program)?;
                Ok(Rc::new(BodyForm::Quoted(code)))
            }
            BodyForm::Lambda(ldata) => self.enrich_lambda_site_info(
                allocator,
                &mut visited,
                prog_args,
                env,
                ldata,
                only_inline,
            ),
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
        allocator: &mut Allocator, // Support random prims via clvm_rs
        prog_args: Rc<SExp>,
        env: &HashMap<Vec<u8>, Rc<BodyForm>>,
        body: Rc<BodyForm>,
        only_inline: bool,
        stack_limit: Option<usize>,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let visited_info = VisitedInfo {
            max_depth: stack_limit,
            ..Default::default()
        };
        let mut visited_marker = VisitedMarker::new(visited_info);
        self.shrink_bodyform_visited(
            allocator, // Support random prims via clvm_rs
            &mut visited_marker,
            prog_args,
            env,
            body,
            only_inline,
        )
    }

    fn expand_macro(
        &self,
        allocator: &mut Allocator, // Support random prims via clvm_rs
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

        let compiled = self.compile_code(allocator, false, use_body)?;
        self.run_prim(allocator, call_loc, compiled, args)
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
        allocator: &mut Allocator,
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

        let com_result = updated_opts.compile_program(
            allocator,
            self.runner.clone(),
            use_body,
            &mut HashMap::new(),
        )?;

        Ok(Rc::new(com_result))
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
