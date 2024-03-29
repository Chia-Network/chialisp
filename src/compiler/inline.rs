use num_bigint::ToBigInt;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::codegen::{generate_expr_code, get_call_name, get_callable};
use crate::compiler::compiler::is_at_capture;
use crate::compiler::comptypes::{
    ArgsAndTail, BodyForm, CallSpec, Callable, CompileErr, CompiledCode, CompilerOpts,
    InlineFunction, LambdaData, PrimaryCodegen,
};
use crate::compiler::lambda::make_cons;
use crate::compiler::sexp::{decode_string, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::{BasicCompileContext, CompileContextWrapper};

use crate::util::Number;

fn apply_fn(loc: Srcloc, name: String, expr: Rc<BodyForm>) -> Rc<BodyForm> {
    Rc::new(BodyForm::Call(
        loc.clone(),
        vec![
            Rc::new(BodyForm::Value(SExp::atom_from_string(loc, &name))),
            expr,
        ],
        // Ok: applying a primitive or builtin requires no tail.
        None,
    ))
}

fn at_form(loc: Srcloc, path: Number) -> Rc<BodyForm> {
    apply_fn(
        loc.clone(),
        "@".to_string(),
        Rc::new(BodyForm::Value(SExp::Integer(loc, path))),
    )
}

/// Generates a list of paths that correspond to the toplevel positions in arg_
/// and if given, the improper tail.
pub fn synthesize_args(arg_: Rc<SExp>) -> (Vec<Rc<BodyForm>>, Option<Rc<BodyForm>>) {
    let two = 2_i32.to_bigint().unwrap();
    let mut start = 5_i32.to_bigint().unwrap();
    let mut tail = bi_one();
    let mut result = Vec::new();
    let mut arg = arg_;
    loop {
        match arg.borrow() {
            SExp::Cons(l, _, b) => {
                result.push(at_form(l.clone(), start.clone()));
                tail = bi_one() | (two.clone() * tail);
                start = bi_one() + start.clone() * two.clone();
                arg = b.clone();
            }
            SExp::Atom(l, _) => {
                return (result, Some(at_form(l.clone(), tail)));
            }
            _ => {
                return (result, None);
            }
        }
    }
}

/// Given arguments that didn't correspond to major list positions in the argument
/// list of the inline function being compiled, and the optional tail argument of
/// the call expression, form an expression which gives the list of unpaired
/// incoming arguments followed by the tail or nil.
///
/// Imagining that a function (defun-inline F (A B . C) ...) is called as
/// (F 3 5 7 11 &rest (list 13 17))
/// In the body, A = 3, B = 5 and C = (list 7 11 13 17)
/// The C argument is populated by creating a list whose tail is the contents
/// of the rest argument and which contains conses of each other proper position
/// argument (in this case, 7 and 11).
fn enlist_remaining_args(
    loc: Srcloc,
    arg_choice: usize,
    args: &[Rc<BodyForm>],
    tail: Option<Rc<BodyForm>>,
) -> Rc<BodyForm> {
    let mut result_body = tail.unwrap_or_else(|| Rc::new(BodyForm::Value(SExp::Nil(loc.clone()))));

    for arg in args.iter().skip(arg_choice).rev() {
        result_body = Rc::new(BodyForm::Call(
            loc.clone(),
            vec![
                Rc::new(BodyForm::Value(SExp::Integer(
                    loc.clone(),
                    4_u32.to_bigint().unwrap(),
                ))),
                arg.clone(),
                result_body,
            ],
            // Ok: applying cons.
            None,
        ));
    }

    result_body
}

fn pick_value_from_arg_element(
    match_args: Rc<SExp>,
    provided: Rc<BodyForm>,
    apply: &dyn Fn(Rc<BodyForm>) -> Rc<BodyForm>,
    name: Vec<u8>,
) -> Option<Rc<BodyForm>> {
    match match_args.borrow() {
        SExp::Cons(l, a, b) => {
            if let Some((capture, children)) = is_at_capture(a.clone(), b.clone()) {
                if capture == name {
                    return Some(apply(provided));
                }

                return pick_value_from_arg_element(children, provided, apply, name);
            }
            let matched_a = pick_value_from_arg_element(
                a.clone(),
                provided.clone(),
                &|x| apply_fn(l.clone(), "f".to_string(), apply(x)),
                name.clone(),
            );
            let matched_b = pick_value_from_arg_element(
                b.clone(),
                provided,
                &|x| apply_fn(l.clone(), "r".to_string(), apply(x)),
                name,
            );

            match (matched_a, matched_b) {
                (Some(a), _) => Some(a),
                (_, Some(b)) => Some(b),
                _ => None,
            }
        }
        SExp::Atom(_l, a) => {
            if *a == name {
                Some(apply(provided))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Given the arguments, a tail and an argument index, return the expression that
/// would in a normal function call correspond to the given argument number.  If
/// positional arguments run out and no tail is specified, a short circuit error
/// is given, otherwise the search generates a path into the tail argument's
/// value.
fn choose_arg_from_list_or_tail(
    callsite: &Srcloc,
    args: &[Rc<BodyForm>],
    tail: Option<Rc<BodyForm>>,
    index: usize,
) -> Result<Rc<BodyForm>, CompileErr> {
    let two = 2_i32.to_bigint().unwrap();

    if index >= args.len() {
        if let Some(t) = tail {
            let target_shift = index - args.len();
            let target_path =
                (two.clone() << target_shift) | (((two << target_shift) - bi_one()) >> 2);
            return Ok(Rc::new(BodyForm::Call(
                callsite.clone(),
                vec![
                    Rc::new(BodyForm::Value(SExp::Integer(
                        t.loc(),
                        2_u32.to_bigint().unwrap(),
                    ))),
                    Rc::new(BodyForm::Value(SExp::Integer(t.loc(), target_path))),
                    t.clone(),
                ],
                // Applying a primitive.
                None,
            )));
        }

        return Err(CompileErr(
            callsite.clone(),
            format!("Lookup for argument {} that wasn't passed", index + 1),
        ));
    }

    Ok(args[index].clone())
}

fn arg_lookup(
    callsite: Srcloc,
    mut match_args: Rc<SExp>,
    args: &[Rc<BodyForm>],
    mut tail: Option<Rc<BodyForm>>,
    name: Vec<u8>,
) -> Result<Option<Rc<BodyForm>>, CompileErr> {
    let two = 2_i32.to_bigint().unwrap();
    let mut arg_choice = 0;

    loop {
        match match_args.borrow() {
            SExp::Cons(_l, f, r) => {
                if let Some(x) = pick_value_from_arg_element(
                    f.clone(),
                    choose_arg_from_list_or_tail(&callsite, args, tail.clone(), arg_choice)?,
                    &|x| x,
                    name.clone(),
                ) {
                    return Ok(Some(x));
                } else {
                    arg_choice += 1;
                    match_args = r.clone();
                    continue;
                }
            }
            _ => {
                if arg_choice > args.len() {
                    let underflow = arg_choice - args.len();
                    let tail_path = (two.clone() << underflow) - bi_one();
                    tail = tail.map(|t| {
                        Rc::new(BodyForm::Call(
                            t.loc(),
                            vec![
                                Rc::new(BodyForm::Value(SExp::Integer(t.loc(), two.clone()))),
                                Rc::new(BodyForm::Value(SExp::Integer(t.loc(), tail_path))),
                                t.clone(),
                            ],
                            None,
                        ))
                    });
                }

                let tail_list = enlist_remaining_args(match_args.loc(), arg_choice, args, tail);
                return Ok(pick_value_from_arg_element(
                    match_args.clone(),
                    tail_list,
                    &|x: Rc<BodyForm>| x,
                    name,
                ));
            }
        }
    }
}

fn get_inline_callable(
    opts: Rc<dyn CompilerOpts>,
    compiler: &PrimaryCodegen,
    loc: Srcloc,
    list_head: Rc<BodyForm>,
) -> Result<Callable, CompileErr> {
    let list_head_borrowed: &BodyForm = list_head.borrow();
    let name = get_call_name(loc.clone(), list_head_borrowed.clone())?;
    get_callable(opts, compiler, loc, name)
}

/// Given a call to an inline function in incoming_spec from an inline function,
/// generate a list of expressions and optional tail expression that convert the
/// given argument expressions into their reified forms that inline the
/// expressions given in the ultimate original call.  This allows inline functions
/// to seem to call each other as long as there's no cycle.
fn make_args_for_call_from_inline(
    visited_inlines: &HashSet<Vec<u8>>,
    runner: Rc<dyn TRunProgram>,
    opts: Rc<dyn CompilerOpts>,
    compiler: &PrimaryCodegen,
    inline: &InlineFunction,
    incoming_spec: &CallSpec,
    call_spec: &CallSpec,
) -> Result<ArgsAndTail, CompileErr> {
    if call_spec.args.is_empty() {
        // This is a nil.
        return Ok(ArgsAndTail {
            args: vec![],
            tail: call_spec.tail.clone(),
        });
    }

    let mut new_args = Vec::new();

    for (i, arg) in call_spec.args.iter().enumerate() {
        if i == 0 {
            new_args.push(arg.clone());
            continue;
        }

        // Since we're going into an argument, pass on a new copy of the visited
        // set.
        let mut new_visited = visited_inlines.clone();
        let replaced = replace_inline_body(
            &mut new_visited,
            runner.clone(),
            opts.clone(),
            compiler,
            arg.loc(),
            inline,
            incoming_spec.args,
            incoming_spec.tail.clone(),
            call_spec.loc.clone(),
            arg.clone(),
        )?;
        new_args.push(replaced);
    }

    // Now that there are tail arguments, the tail gets a new visited set as well.
    let mut new_visited = visited_inlines.clone();
    let replaced_tail = if let Some(t) = call_spec.tail.as_ref() {
        Some(replace_inline_body(
            &mut new_visited,
            runner,
            opts,
            compiler,
            t.loc(),
            inline,
            incoming_spec.args,
            incoming_spec.tail.clone(),
            call_spec.loc.clone(),
            t.clone(),
        )?)
    } else {
        None
    };

    Ok(ArgsAndTail {
        args: new_args,
        tail: replaced_tail,
    })
}

// The main workhorse of inlining, given a bodyform and the elements specifying
// how the inline function was called, generate an expansion of the expression
// that relies on the incoming argument expressions.
#[allow(clippy::too_many_arguments)]
fn replace_inline_body(
    visited_inlines: &mut HashSet<Vec<u8>>,
    runner: Rc<dyn TRunProgram>,
    opts: Rc<dyn CompilerOpts>,
    compiler: &PrimaryCodegen,
    loc: Srcloc,
    inline: &InlineFunction,
    args: &[Rc<BodyForm>],
    tail: Option<Rc<BodyForm>>,
    callsite: Srcloc,
    expr: Rc<BodyForm>,
) -> Result<Rc<BodyForm>, CompileErr> {
    match expr.borrow() {
        BodyForm::Let(_, _) => Err(CompileErr(
            loc,
            "let binding should have been hoisted before optimization".to_string(),
        )),
        BodyForm::Call(l, call_args, call_tail) => {
            // Ensure that we don't count branched invocations when checking
            // each call downstream of the main expr is recursive.
            //
            // Previously, this program detected as recursive:
            // (mod (A) ;; 11
            //   (include *standard-cl-22*)
            //   (defun-inline <= (A B) (not (> A B)))
            //   (assign
            //     foo (<= 2 A)
            //     bar (<= 1 A)
            //
            //     baz (<= foo bar)
            //
            //     yorgle (<= baz bar)
            //
            //     (<= yorgle foo)
            //     ))
            //
            // <= appears in the arguments, but isn't called recursively on itself.
            // We ensure here that each argument has a separate visited stack.
            // Recursion only happens when the same stack encounters an inline
            // twice.
            //
            let args_and_tail = make_args_for_call_from_inline(
                visited_inlines,
                runner.clone(),
                opts.clone(),
                compiler,
                inline,
                &CallSpec {
                    loc: callsite.clone(),
                    name: &inline.name,
                    args,
                    tail,
                    original: expr.clone(),
                },
                &CallSpec {
                    loc: callsite.clone(),
                    name: &inline.name,
                    args: call_args,
                    tail: call_tail.clone(),
                    original: expr.clone(),
                },
            )?;

            // If the called function is an inline, we'll expand it here.
            // This is so we can preserve the context of argument expressions
            // so no new mapping is needed.  This solves a number of problems
            // with the previous implementation.
            //
            // If it's a macro we'll expand it here so we can recurse and
            // determine whether an inline is the next level.
            //
            // It's an inline, so we need to fulfill its arguments.
            match get_inline_callable(opts.clone(), compiler, l.clone(), call_args[0].clone())? {
                Callable::CallInline(l, new_inline) => {
                    if visited_inlines.contains(&new_inline.name) {
                        return Err(CompileErr(
                            l,
                            format!(
                                "recursive call to inline function {}",
                                decode_string(&inline.name)
                            ),
                        ));
                    }

                    visited_inlines.insert(new_inline.name.clone());

                    let pass_on_args: Vec<Rc<BodyForm>> =
                        args_and_tail.args.iter().skip(1).cloned().collect();
                    replace_inline_body(
                        visited_inlines,
                        runner,
                        opts.clone(),
                        compiler,
                        l, // clippy update since 1.59
                        &new_inline,
                        &pass_on_args,
                        args_and_tail.tail,
                        callsite,
                        new_inline.body.clone(),
                    )
                }
                _ => {
                    // Tail passes through to a normal call form.
                    let call = BodyForm::Call(l.clone(), args_and_tail.args, args_and_tail.tail);
                    Ok(Rc::new(call))
                }
            }
        }
        BodyForm::Value(SExp::Atom(l, a)) => {
            if a == b"@*env*" {
                // Reify the environment as it looks from here.
                let left_env = Rc::new(BodyForm::Call(
                    l.clone(),
                    vec![
                        Rc::new(BodyForm::Value(SExp::Atom(l.clone(), b"@".to_vec()))),
                        Rc::new(BodyForm::Value(SExp::Integer(
                            l.clone(),
                            2_u32.to_bigint().unwrap(),
                        ))),
                    ],
                    // Builtin
                    None,
                ));
                let mut env = Rc::new(BodyForm::Quoted(SExp::Nil(l.clone())));
                for arg in args.iter().rev() {
                    env = Rc::new(make_cons(l.clone(), arg.clone(), env));
                }
                env = Rc::new(make_cons(l.clone(), left_env, env));
                return Ok(env);
            } else if a == b"@" {
                return Ok(Rc::new(BodyForm::Value(SExp::Atom(
                    l.clone(),
                    b"@".to_vec(),
                ))));
            }

            let alookup = arg_lookup(callsite, inline.args.clone(), args, tail, a.clone())?
                .unwrap_or_else(|| expr.clone());
            Ok(alookup)
        }
        BodyForm::Lambda(ldata) => {
            let rewritten_captures = replace_inline_body(
                visited_inlines,
                runner,
                opts,
                compiler,
                loc,
                inline,
                args,
                tail,
                callsite,
                ldata.captures.clone(),
            )?;
            Ok(Rc::new(BodyForm::Lambda(Box::new(LambdaData {
                captures: rewritten_captures,
                ..*ldata.clone()
            }))))
        }
        _ => Ok(expr.clone()),
    }
}

/// Given an inline function and a list of arguments, return compiled code that
/// stands in for the inline expansion.  Along the way, generate code for the
/// expressions in the argument list.
///
/// This will probably be changed at some point to return Rc<BodyForm> so it
/// can be treated as a desugaring step that's subject to frontend optimization.
///
/// There are two cases when we have rest arguments:
///
/// 1) Too few arguments are provided along with a rest argument.
///    In this case, we can't tell what's been provided for the variable arguments
///    so we synthesize paths into the rest argument and hope for the best.
///
/// 2) Too many arguments are provided (optionally with a rest argument).
///    In this case, we collect the following arguments and pass them to a
///    tail argument if one exists.  If not, then they're discarded.
#[allow(clippy::too_many_arguments)]
pub fn replace_in_inline(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    compiler: &PrimaryCodegen,
    loc: Srcloc,
    inline: &InlineFunction,
    callsite: Srcloc,
    args: &[Rc<BodyForm>],
    tail: Option<Rc<BodyForm>>,
) -> Result<CompiledCode, CompileErr> {
    let mut visited = HashSet::new();
    let runner = context.runner();
    visited.insert(inline.name.clone());
    replace_inline_body(
        &mut visited,
        runner.clone(),
        opts.clone(),
        compiler,
        loc,
        inline,
        args,
        tail,
        callsite,
        inline.body.clone(),
    )
    .and_then(|x| {
        let mut symbols = HashMap::new();
        let runner = context.runner();
        let optimizer = context.optimizer.duplicate();
        let mut context_wrapper =
            CompileContextWrapper::new(context.allocator(), runner, &mut symbols, optimizer);
        generate_expr_code(&mut context_wrapper.context, opts, compiler, x)
    })
}
