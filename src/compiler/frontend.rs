use std::borrow::Borrow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::compiler::comptypes::{
    list_to_cons, match_as_named, ArgsAndTail, Binding, BindingPattern, BodyForm, CompileErr,
    CompileForm, CompilerOpts, ConstantKind, DefconstData, DefmacData, DefunData, Export,
    ExportProgramDesc, FrontendOutput, HelperForm, ImportLongName, IncludeDesc, LetData,
    LetFormInlineHint, LetFormKind, LongNameTranslation, ModAccum, ModuleImportSpec, NamespaceData,
    NamespaceRefData,
};
use crate::compiler::lambda::handle_lambda;
use crate::compiler::preprocessor::{
    detect_chialisp_module, parse_toplevel_mod, preprocess, Preprocessor, ToplevelModParseResult,
};
use crate::compiler::rename::rename_children_compileform;
use crate::compiler::sexp::{decode_string, enlist, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::util::u8_from_number;

pub fn collect_used_names_sexp(body: Rc<SExp>) -> Vec<Vec<u8>> {
    match body.borrow() {
        SExp::Atom(_, name) => vec![name.to_vec()],
        SExp::Cons(_, head, tail) => {
            let mut head_collected = collect_used_names_sexp(head.clone());
            let mut tail_collected = collect_used_names_sexp(tail.clone());
            head_collected.append(&mut tail_collected);
            head_collected
        }
        _ => vec![],
    }
}

fn collect_used_names_binding(body: &Binding) -> Vec<Vec<u8>> {
    collect_used_names_bodyform(body.body.borrow())
}

pub fn collect_used_names_bodyform(body: &BodyForm) -> Vec<Vec<u8>> {
    match body {
        BodyForm::Let(_, letdata) => {
            let mut result = Vec::new();
            for b in letdata.bindings.iter() {
                let mut new_binding_names = collect_used_names_binding(b);
                result.append(&mut new_binding_names);
            }

            let mut body_names = collect_used_names_bodyform(letdata.body.borrow());
            result.append(&mut body_names);
            result
        }
        BodyForm::Quoted(_) => vec![],
        BodyForm::Value(atom) => match atom {
            SExp::Atom(_l, v) => vec![v.to_vec()],
            SExp::Cons(_l, f, r) => {
                let mut first_names = collect_used_names_sexp(f.clone());
                let mut rest_names = collect_used_names_sexp(r.clone());
                first_names.append(&mut rest_names);
                first_names
            }
            _ => Vec::new(),
        },
        BodyForm::Call(_l, vs, tail) => {
            let mut result = Vec::new();
            for a in vs {
                let mut argnames = collect_used_names_bodyform(a);
                result.append(&mut argnames);
            }
            if let Some(t) = tail {
                let mut tail_names = collect_used_names_bodyform(t);
                result.append(&mut tail_names);
            }
            result
        }
        BodyForm::Mod(_, _) => vec![],
        BodyForm::Lambda(ldata) => {
            let mut capture_names = collect_used_names_bodyform(ldata.captures.borrow());
            capture_names.append(&mut collect_used_names_bodyform(ldata.body.borrow()));
            capture_names
        }
    }
}

pub fn collect_used_names_helperform(body: &HelperForm) -> Vec<Vec<u8>> {
    match body {
        HelperForm::Defconstant(defc) => collect_used_names_bodyform(defc.body.borrow()),
        HelperForm::Defmacro(mac) => {
            let mut res = collect_used_names_compileform(mac.program.borrow());
            // Ensure any other names mentioned in qq blocks are included.
            let mut all_token_res = collect_used_names_sexp(mac.program.to_sexp());
            res.append(&mut all_token_res);
            res
        }
        HelperForm::Defun(_, defun) => collect_used_names_bodyform(&defun.body),
        HelperForm::Defnamespace(_ns) => Vec::new(),
        HelperForm::Defnsref(_ns) => Vec::new(),
    }
}

fn collect_used_names_compileform(body: &CompileForm) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    for h in body.helpers.iter() {
        let mut helper_list = collect_used_names_helperform(h);
        result.append(&mut helper_list);
    }

    let mut ex_list = collect_used_names_bodyform(body.exp.borrow());
    result.append(&mut ex_list);
    result
}

fn calculate_live_helpers(
    last_names: &HashSet<Vec<u8>>,
    names: &HashSet<Vec<u8>>,
    helper_map: &HashMap<Vec<u8>, HelperForm>,
) -> HashSet<Vec<u8>> {
    if last_names.len() == names.len() {
        names.clone()
    } else {
        let new_names: HashSet<Vec<u8>> =
            names.difference(last_names).map(|x| x.to_vec()).collect();
        let mut needed_helpers: HashSet<Vec<u8>> = names.clone();

        for name in new_names {
            if let Some(new_helper) = helper_map.get(&name) {
                let even_newer_names: HashSet<Vec<u8>> = collect_used_names_helperform(new_helper)
                    .iter()
                    .map(|x| x.to_vec())
                    .collect();
                needed_helpers = needed_helpers
                    .union(&even_newer_names)
                    .map(|x| x.to_vec())
                    .collect();
            }
        }

        calculate_live_helpers(names, &needed_helpers, helper_map)
    }
}

fn qq_to_expression(opts: Rc<dyn CompilerOpts>, body: Rc<SExp>) -> Result<BodyForm, CompileErr> {
    let body_copy: &SExp = body.borrow();

    match body.borrow() {
        SExp::Cons(l, f, r) => {
            let op = match f.borrow() {
                SExp::Atom(_, o) => o.clone(),
                SExp::QuotedString(_, _, s) => s.clone(),
                SExp::Integer(_, i) => u8_from_number(i.clone()),
                _ => Vec::new(),
            };

            if op.len() == 1 && (op[0] == b'q' || op[0] == 1) {
                return Ok(BodyForm::Quoted(body_copy.clone()));
            } else if let Some(list) = r.proper_list() {
                if op == b"quote" {
                    if list.len() != 1 {
                        return Err(CompileErr(l.clone(), format!("bad form {body}")));
                    }

                    return Ok(BodyForm::Quoted(list[0].clone()));
                } else if op == b"unquote" {
                    if list.len() != 1 {
                        return Err(CompileErr(l.clone(), format!("bad form {body}")));
                    }

                    return compile_bodyform(opts.clone(), Rc::new(list[0].clone()));
                }
            }

            qq_to_expression_list(opts, body.clone())
        }
        _ => Ok(BodyForm::Quoted(body_copy.clone())),
    }
}

fn qq_to_expression_list(
    opts: Rc<dyn CompilerOpts>,
    body: Rc<SExp>,
) -> Result<BodyForm, CompileErr> {
    match body.borrow() {
        SExp::Cons(l, f, r) => {
            m! {
                f_qq <- qq_to_expression(opts.clone(), f.clone());
                r_qq <- qq_to_expression_list(opts, r.clone());
                Ok(BodyForm::Call(l.clone(), vec!(
                    Rc::new(BodyForm::Value(
                        SExp::Atom(l.clone(), "c".as_bytes().to_vec())
                    )),
                    Rc::new(f_qq),
                    Rc::new(r_qq)
                ), None))
            }
        }
        SExp::Nil(l) => Ok(BodyForm::Quoted(SExp::Nil(l.clone()))),
        _ => Err(CompileErr(
            body.loc(),
            format!("Bad list tail in qq {body}"),
        )),
    }
}

fn args_to_expression_list(
    opts: Rc<dyn CompilerOpts>,
    body: Rc<SExp>,
) -> Result<ArgsAndTail, CompileErr> {
    if body.nilp() {
        Ok(ArgsAndTail::default())
    } else {
        match body.borrow() {
            SExp::Cons(_l, first, rest) => {
                if let SExp::Atom(_fl, fname) = first.borrow() {
                    if fname == b"&rest" {
                        // Rest is a list containing one item that becomes the
                        // tail.
                        //
                        // Downstream, we'll use the tail instead of a nil as the
                        // final element of a call form.  In the inline case, this
                        // means that argument references will be generated into
                        // the runtime tail expression when appropriate.
                        let args_no_tail = args_to_expression_list(opts, rest.clone())?;

                        if args_no_tail.tail.is_some() {
                            return Err(CompileErr(
                                rest.loc(),
                                "only one use of &rest is allowed".to_string(),
                            ));
                        }

                        if args_no_tail.args.len() != 1 {
                            return Err(CompileErr(
                                body.loc(),
                                "&rest specified with bad tail".to_string(),
                            ));
                        }

                        // Return a tail.
                        return Ok(ArgsAndTail {
                            args: vec![],
                            tail: Some(args_no_tail.args[0].clone()),
                        });
                    }
                }
                let mut result_list = Vec::new();
                let f_compiled = compile_bodyform(opts.clone(), first.clone())?;
                result_list.push(Rc::new(f_compiled));
                let mut args_and_tail = args_to_expression_list(opts, rest.clone())?;
                result_list.append(&mut args_and_tail.args);
                Ok(ArgsAndTail {
                    args: result_list,
                    tail: args_and_tail.tail,
                })
            }
            _ => Err(CompileErr(body.loc(), format!("Bad arg list tail {body}"))),
        }
    }
}

fn make_let_bindings(
    opts: Rc<dyn CompilerOpts>,
    body: Rc<SExp>,
) -> Result<Vec<Rc<Binding>>, CompileErr> {
    let err = Err(CompileErr(body.loc(), format!("Bad binding tail {body:?}")));
    let do_atomize = if !opts.dialect().strict {
        |a: &SExp| -> SExp { a.atomize() }
    } else {
        |a: &SExp| -> SExp { a.clone() }
    };
    match body.borrow() {
        SExp::Nil(_) => Ok(vec![]),
        SExp::Cons(_, head, tl) => head
            .proper_list()
            .filter(|x| x.len() == 2)
            .map(|x| match (do_atomize(&x[0]), &x[1]) {
                (SExp::Atom(l, name), expr) => {
                    let compiled_body = compile_bodyform(opts.clone(), Rc::new(expr.clone()))?;
                    let mut result = Vec::new();
                    let mut rest_bindings = make_let_bindings(opts, tl.clone())?;
                    result.push(Rc::new(Binding {
                        loc: l.clone(),
                        nl: l,
                        pattern: BindingPattern::Name(name.to_vec()),
                        body: Rc::new(compiled_body),
                    }));
                    result.append(&mut rest_bindings);
                    Ok(result)
                }
                _ => err.clone(),
            })
            .unwrap_or_else(|| err.clone()),
        _ => err,
    }
}

// Make a set of names in this sexp.
pub fn make_provides_set(provides_set: &mut HashSet<Vec<u8>>, body_sexp: Rc<SExp>) {
    match body_sexp.atomize() {
        SExp::Cons(_, a, b) => {
            make_provides_set(provides_set, a);
            make_provides_set(provides_set, b);
        }
        SExp::Atom(_, name) => {
            provides_set.insert(name);
        }
        _ => {}
    }
}

fn handle_assign_form(
    opts: Rc<dyn CompilerOpts>,
    l: Srcloc,
    v: &[SExp],
    inline_hint: Option<LetFormInlineHint>,
) -> Result<BodyForm, CompileErr> {
    if v.len().is_multiple_of(2) {
        return Err(CompileErr(
            l,
            "assign form should be in pairs of pattern value followed by an expression".to_string(),
        ));
    }

    let mut bindings = Vec::new();
    let mut check_duplicates = HashSet::new();

    for idx in (0..(v.len() - 1) / 2).map(|idx| idx * 2) {
        let destructure_pattern = Rc::new(v[idx].clone());
        let binding_body = compile_bodyform(opts.clone(), Rc::new(v[idx + 1].clone()))?;
        // Ensure bindings aren't duplicated as we won't be able to
        // guarantee their order during toposort.
        let mut this_provides = HashSet::new();
        make_provides_set(&mut this_provides, destructure_pattern.clone());

        for item in this_provides.iter() {
            if check_duplicates.contains(item) {
                return Err(CompileErr(
                    destructure_pattern.loc(),
                    format!("Duplicate binding {}", decode_string(item)),
                ));
            }
            check_duplicates.insert(item.clone());
        }

        bindings.push(Rc::new(Binding {
            loc: v[idx].loc().ext(&v[idx + 1].loc()),
            nl: destructure_pattern.loc(),
            pattern: BindingPattern::Complex(destructure_pattern),
            body: Rc::new(binding_body),
        }));
    }

    let compiled_body = compile_bodyform(opts.clone(), Rc::new(v[v.len() - 1].clone()))?;
    // We don't need to do much if there were no bindings.
    if bindings.is_empty() {
        return Ok(compiled_body);
    }

    // Return a precise representation of this assign while storing up the work
    // we did breaking it down.
    Ok(BodyForm::Let(
        LetFormKind::Assign,
        Box::new(LetData {
            loc: l.clone(),
            kw: Some(l),
            bindings,
            inline_hint,
            body: Rc::new(compiled_body),
        }),
    ))
}

pub fn compile_bodyform(
    opts: Rc<dyn CompilerOpts>,
    body: Rc<SExp>,
) -> Result<BodyForm, CompileErr> {
    match body.borrow() {
        SExp::Cons(l, op, tail) => {
            let application = || {
                args_to_expression_list(opts.clone(), tail.clone()).and_then(|atail| {
                    compile_bodyform(opts.clone(), op.clone()).map(|func| {
                        let mut result_call = vec![Rc::new(func)];
                        let mut args_clone = atail.args.to_vec();
                        // Ensure that the full extent of the call expression
                        // in the source file becomes the Srcloc given to the
                        // call itself.
                        let ending = if atail.args.is_empty() {
                            l.ending()
                        } else {
                            atail.args[atail.args.len() - 1].loc().ending()
                        };
                        result_call.append(&mut args_clone);
                        BodyForm::Call(l.ext(&ending), result_call, atail.tail)
                    })
                })
            };

            let finish_err = |site| {
                Err(CompileErr(
                    l.clone(),
                    format!("{site}: bad argument list for form {body}"),
                ))
            };

            match op.borrow() {
                SExp::Atom(l, atom_name) => {
                    if *atom_name == b"q" || (atom_name.len() == 1 && atom_name[0] == 1) {
                        let tail_copy: &SExp = tail.borrow();
                        return Ok(BodyForm::Quoted(tail_copy.clone()));
                    }

                    let assign_lambda = *atom_name == "assign-lambda".as_bytes().to_vec();
                    let assign_inline = *atom_name == "assign-inline".as_bytes().to_vec();

                    match tail.proper_list() {
                        Some(v) => {
                            if *atom_name == b"let" || *atom_name == b"let*" {
                                if v.len() != 2 {
                                    return finish_err("let");
                                }

                                let kind = if *atom_name == b"let" {
                                    LetFormKind::Parallel
                                } else {
                                    LetFormKind::Sequential
                                };

                                let bindings = v[0].clone();
                                let body = v[1].clone();

                                let let_bindings =
                                    make_let_bindings(opts.clone(), Rc::new(bindings))?;
                                let compiled_body = compile_bodyform(opts, Rc::new(body))?;
                                Ok(BodyForm::Let(
                                    kind,
                                    Box::new(LetData {
                                        loc: l.clone(),
                                        kw: Some(l.clone()),
                                        bindings: let_bindings,
                                        inline_hint: None,
                                        body: Rc::new(compiled_body),
                                    }),
                                ))
                            } else if assign_lambda
                                || assign_inline
                                || *atom_name == "assign".as_bytes().to_vec()
                            {
                                handle_assign_form(
                                    opts.clone(),
                                    l.clone(),
                                    &v,
                                    if assign_lambda {
                                        Some(LetFormInlineHint::NonInline(l.clone()))
                                    } else if assign_inline {
                                        Some(LetFormInlineHint::Inline(l.clone()))
                                    } else {
                                        Some(LetFormInlineHint::NoChoice)
                                    },
                                )
                            } else if *atom_name == "quote".as_bytes().to_vec() {
                                if v.len() != 1 {
                                    return finish_err("quote");
                                }

                                let quote_body = v[0].clone();

                                Ok(BodyForm::Quoted(quote_body))
                            } else if *atom_name == b"qq" {
                                if v.len() != 1 {
                                    return finish_err("qq");
                                }

                                let quote_body = v[0].clone();

                                qq_to_expression(opts, Rc::new(quote_body))
                            } else if *atom_name == b"mod" {
                                let subparse = frontend(opts, std::slice::from_ref(&body))?;
                                Ok(BodyForm::Mod(op.loc(), subparse.compileform().clone()))
                            } else if *atom_name == b"lambda" {
                                handle_lambda(opts, body.loc(), Some(l.clone()), &v)
                            } else {
                                application()
                            }
                        }
                        None => finish_err("tail_proper"),
                    }
                }
                SExp::Integer(il, i) => compile_bodyform(
                    opts,
                    Rc::new(SExp::Cons(
                        il.clone(),
                        Rc::new(SExp::Atom(il.clone(), u8_from_number(i.clone()))),
                        tail.clone(),
                    )),
                ),
                SExp::QuotedString(_, _, _) => {
                    let body_copy: &SExp = body.borrow();
                    Ok(BodyForm::Value(body_copy.clone()))
                }
                SExp::Nil(_l) => {
                    let body_copy: &SExp = body.borrow();
                    Ok(BodyForm::Quoted(body_copy.clone()))
                }
                SExp::Cons(_, _, _) => finish_err("bad cons"),
            }
        }
        _ => {
            let body_copy: &SExp = body.borrow();
            Ok(BodyForm::Value(body_copy.clone()))
        }
    }
}

// More modern constant definition that interprets code ala constexpr.
fn compile_defconst(
    opts: Rc<dyn CompilerOpts>,
    l: Srcloc,
    nl: Srcloc,
    kl: Option<Srcloc>,
    name: Vec<u8>,
    body: Rc<SExp>,
) -> Result<HelperForm, CompileErr> {
    let bf = compile_bodyform(opts.clone(), body)?;
    Ok(HelperForm::Defconstant(DefconstData {
        kw: kl,
        nl,
        loc: l,
        kind: ConstantKind::Complex,
        name: name.to_vec(),
        body: Rc::new(bf),
        tabled: opts.frontend_opt() || opts.dialect().stepping.unwrap_or(0) > 22,
    }))
}

fn compile_defconstant(
    opts: Rc<dyn CompilerOpts>,
    l: Srcloc,
    nl: Srcloc,
    kl: Option<Srcloc>,
    name: Vec<u8>,
    body: Rc<SExp>,
) -> Result<HelperForm, CompileErr> {
    let body_borrowed: &SExp = body.borrow();
    if let SExp::Cons(_, _, _) = body_borrowed {
        Ok(HelperForm::Defconstant(DefconstData {
            loc: l,
            nl,
            kw: kl,
            kind: ConstantKind::Simple,
            name: name.to_vec(),
            body: Rc::new(BodyForm::Value(body_borrowed.clone())),
            tabled: false,
        }))
    } else {
        compile_bodyform(opts, body).map(|bf| {
            HelperForm::Defconstant(DefconstData {
                loc: l,
                nl,
                kw: kl,
                kind: ConstantKind::Simple,
                name: name.to_vec(),
                body: Rc::new(bf),
                tabled: false,
            })
        })
    }
}

fn location_span(l_: Srcloc, lst_: Rc<SExp>) -> Srcloc {
    let mut l = l_;
    let mut lst = lst_;
    while let SExp::Cons(_, a, b) = lst.borrow() {
        l = location_span(l.clone(), a.clone()).ext(&b.loc());
        lst = b.clone();
    }
    l
}

pub struct CompileDefun {
    pub l: Srcloc,
    pub nl: Srcloc,
    pub kwl: Option<Srcloc>,
    pub inline: bool,
    pub name: Vec<u8>,
    pub args: Rc<SExp>,
    pub body: Rc<SExp>,
}

fn compile_defun(opts: Rc<dyn CompilerOpts>, data: CompileDefun) -> Result<HelperForm, CompileErr> {
    let mut take_form = data.body.clone();

    if let SExp::Cons(_, f, _r) = data.body.borrow() {
        take_form = f.clone();
    }
    compile_bodyform(opts, take_form).map(|bf| {
        HelperForm::Defun(
            data.inline,
            Box::new(DefunData {
                loc: data.l,
                nl: data.nl,
                kw: data.kwl,
                name: data.name,
                args: data.args.clone(),
                orig_args: data.args,
                body: Rc::new(bf),
                synthetic: None,
            }),
        )
    })
}

fn compile_defmacro(
    opts: Rc<dyn CompilerOpts>,
    l: Srcloc,
    nl: Srcloc,
    kwl: Option<Srcloc>,
    name: Vec<u8>,
    args: Rc<SExp>,
    body: Rc<SExp>,
) -> Result<HelperForm, CompileErr> {
    let program = SExp::Cons(
        l.clone(),
        Rc::new(SExp::Atom(l.clone(), b"mod".to_vec())),
        Rc::new(SExp::Cons(l.clone(), args.clone(), body)),
    );
    let new_opts = opts.set_stdenv(false);
    frontend(new_opts, &[Rc::new(program)]).map(|p| {
        HelperForm::Defmacro(DefmacData {
            loc: l,
            nl,
            kw: kwl,
            name,
            args: args.clone(),
            program: Rc::new(p.compileform().clone()),
            advanced: false,
        })
    })
}

struct OpName4Match {
    opl: Srcloc,
    op_name: Vec<u8>,
    nl: Srcloc,
    name: Vec<u8>,
    args: Rc<SExp>,
    body: Rc<SExp>,
    orig: Vec<SExp>,
}

fn match_op_name_4(pl: &[SExp]) -> Option<OpName4Match> {
    if pl.is_empty() {
        return None;
    }

    // Atomize ensures that the result is presented as an atom if possible.  This makes the parser
    // a bit more forgiving when we're examining code that has round-tripped through traditional
    // clvm, which happens when modern tools such as use check are used on classic programs.
    // The modern frontend requires atom representation in positions 0 and 1 of helpers anyway.
    match &pl[0].atomize() {
        SExp::Atom(l, op_name) => {
            if pl.len() < 3 {
                return Some(OpName4Match {
                    opl: l.clone(),
                    op_name: op_name.clone(),
                    nl: l.clone(),
                    name: Vec::new(),
                    args: Rc::new(SExp::Nil(l.clone())),
                    body: Rc::new(SExp::Nil(l.clone())),
                    orig: pl.to_owned(),
                });
            }

            match &pl[1].atomize() {
                SExp::Atom(ll, name) => {
                    let tail_idx = 3;
                    let mut tail_list = Vec::new();
                    for elt in pl.iter().skip(tail_idx) {
                        tail_list.push(Rc::new(elt.clone()));
                    }

                    Some(OpName4Match {
                        nl: ll.clone(),
                        op_name: op_name.clone(),
                        opl: l.clone(),
                        name: name.clone(),
                        args: Rc::new(pl[2].clone()),
                        body: Rc::new(enlist(l.clone(), &tail_list)),
                        orig: pl.to_owned(),
                    })
                }
                _ => Some(OpName4Match {
                    nl: pl[0].loc(),
                    opl: l.clone(),
                    op_name: op_name.clone(),
                    name: Vec::new(),
                    args: Rc::new(SExp::Nil(l.clone())),
                    body: Rc::new(SExp::Nil(l.clone())),
                    orig: pl.to_owned(),
                }),
            }
        }
        _ => None,
    }
}

pub fn match_export_form(
    opts: Rc<dyn CompilerOpts>,
    form: Rc<SExp>,
) -> Result<Option<Export>, CompileErr> {
    if let Some(lst) = form.proper_list() {
        // Empty form isn't export
        if lst.is_empty() {
            return Ok(None);
        }

        // Export if it has an export keyword.
        if let SExp::Atom(_, export_name) = lst[0].borrow() {
            if export_name != b"export" {
                return Ok(None);
            }
        } else {
            // No export kw, not export.
            return Ok(None);
        }

        if let Some(efd) = match_as_named(form.loc(), &lst, 1) {
            return Ok(Some(Export::Function(Box::new(efd))));
        }

        // A main export
        if lst.len() != 3 {
            return Err(CompileErr(form.loc(), format!("Malformed export {form}")));
        }

        let expr = compile_bodyform(opts.clone(), Rc::new(lst[2].clone()))?;
        return Ok(Some(Export::MainProgram(ExportProgramDesc {
            loc: form.loc(),
            kw_loc: Some(lst[0].loc()),
            args: Rc::new(lst[1].clone()),
            expr: Rc::new(expr),
        })));
    }

    Ok(None)
}

#[derive(Debug, Clone)]
pub struct HelperFormResult {
    pub new_helpers: Vec<HelperForm>,
}

impl HelperFormResult {
    pub fn new(helpers: &[HelperForm]) -> Self {
        HelperFormResult {
            new_helpers: helpers.to_vec(),
        }
    }
}

/// Produce a helperform from a namespace.  This acts as a storage container for helpers that
/// can later be resolved via module resolution.
///
/// Nobody will write this, but it is parsed and transits this representation:
///
/// (namespace Foo.Bar
///    (defconst C 99)
///    (defun F (X) ...)
/// )
///
/// These are used at compile time as helper buckets which can be drawn from to reseolve
/// external dependencies a lot like how object files are used during linking of native code.
///
/// Module style constructs a program that contains namespaces and then resolves them to
/// construct a non-namespaced program where all helpers have fully qualified names.
pub fn compile_namespace(
    opts: Rc<dyn CompilerOpts>,
    loc: Srcloc,
    internal: &[SExp],
) -> Result<HelperForm, CompileErr> {
    // A namespace needs a keyword and a name.
    if internal.len() < 2 {
        return Err(CompileErr(loc, "Namespace must have a name".to_string()));
    }

    // Its name should parse as a qualified name, even if it has only one component.
    let (_, parsed) = if let SExp::Atom(_, name) = &internal[1] {
        ImportLongName::parse(name)
    } else {
        return Err(CompileErr(
            internal[1].loc(),
            "Namespace name must be an atom".to_string(),
        ));
    };

    let mut helpers = Vec::new();
    // Parse all helpers.
    for sexp in internal.iter().skip(2) {
        if let Some(hresult) = compile_helperform(opts.clone(), Rc::new(sexp.clone()))? {
            for h in hresult.new_helpers.iter() {
                helpers.push(h.clone());
            }
        } else {
            return Err(CompileErr(
                sexp.loc(),
                "Namespaces must contain only definitions".to_string(),
            ));
        }
    }

    Ok(HelperForm::Defnamespace(Box::new(NamespaceData {
        loc: loc.clone(),
        kw: internal[0].loc(),
        nl: internal[1].loc(),
        rendered_name: parsed.as_u8_vec(LongNameTranslation::Namespace),
        longname: parsed,
        helpers,
    })))
}

/// Add a namespace reference from an input form.  These specify namespaces to use when adding
/// helper forms.  A namespaced program is rewritten so that every helper has a fully resolved
/// name.
///
/// An nsref is in one of these forms:
///   (import Foo.Bar)
///   (import qualified Foo.Bar)
///   (import qualified Foo.Bar as A.B)
///   (import Foo.Bar exposing baz (a as b))
///   (import Foo.Bar hiding z)
///
/// This is mostly like what haskell does and it's pretty expressive.
pub fn compile_nsref(loc: Srcloc, internal: &[SExp]) -> Result<HelperForm, CompileErr> {
    if internal.len() < 2 {
        return Err(CompileErr(
            loc.clone(),
            "import must import a module".to_string(),
        ));
    }

    // Compile a module import spec from the tail of the import.
    let import_spec = ModuleImportSpec::parse(loc.clone(), internal, 1)?;
    // Qualfiied forms are simpler so we short circuit here.
    if let ModuleImportSpec::Qualified(q) = &import_spec {
        return Ok(HelperForm::Defnsref(Box::new(NamespaceRefData {
            loc,
            kw: internal[0].loc(),
            nl: q.nl.clone(),
            rendered_name: q.name.as_u8_vec(LongNameTranslation::Namespace),
            longname: q.name.clone(),
            specification: import_spec.clone(),
        })));
    }

    // Not qualified path, so we haven't validated the name yet.
    let (_, parsed) = if let SExp::Atom(_nl, name) = &internal[1] {
        ImportLongName::parse(name)
    } else {
        return Err(CompileErr(
            internal[1].loc(),
            "Import name must be an atom".to_string(),
        ));
    };

    Ok(HelperForm::Defnsref(Box::new(NamespaceRefData {
        loc,
        kw: internal[0].loc(),
        nl: internal[1].loc(),
        rendered_name: parsed.as_u8_vec(LongNameTranslation::Namespace),
        longname: parsed,
        specification: import_spec,
    })))
}

pub fn compile_helperform(
    opts: Rc<dyn CompilerOpts>,
    body: Rc<SExp>,
) -> Result<Option<HelperFormResult>, CompileErr> {
    compile_helperform_mm(opts, body, false)
}

pub fn compile_helperform_mm(
    opts: Rc<dyn CompilerOpts>,
    body: Rc<SExp>,
    modules: bool,
) -> Result<Option<HelperFormResult>, CompileErr> {
    let l = location_span(body.loc(), body.clone());
    let plist = body.proper_list();

    if let Some(matched) = plist.and_then(|pl| match_op_name_4(&pl)) {
        let inline = matched.op_name == "defun-inline".as_bytes().to_vec();
        let is_defmac = matched.op_name == "defmac".as_bytes().to_vec();
        if matched.op_name == "defconstant".as_bytes().to_vec() {
            let definition = compile_defconstant(
                opts,
                l,
                matched.nl,
                Some(matched.opl),
                matched.name.to_vec(),
                matched.args,
            )?;
            Ok(Some(HelperFormResult {
                new_helpers: vec![definition],
            }))
        } else if matched.op_name == b"defconst" {
            let definition = compile_defconst(
                opts,
                l,
                matched.nl,
                Some(matched.opl),
                matched.name.to_vec(),
                matched.args,
            )?;
            Ok(Some(HelperFormResult {
                new_helpers: vec![definition],
            }))
        } else if matched.op_name == b"defmacro" || is_defmac {
            if is_defmac && modules {
                return Ok(Some(HelperFormResult {
                    new_helpers: vec![],
                }));
            }

            let definition = compile_defmacro(
                opts,
                l,
                matched.nl,
                Some(matched.opl),
                matched.name.to_vec(),
                matched.args,
                matched.body,
            )?;
            Ok(Some(HelperFormResult {
                new_helpers: vec![definition],
            }))
        } else if matched.op_name == "defun".as_bytes().to_vec() || inline {
            let definition = compile_defun(
                opts,
                CompileDefun {
                    l,
                    nl: matched.nl,
                    kwl: Some(matched.opl),
                    inline,
                    name: matched.name.to_vec(),
                    args: matched.args.clone(),
                    body: matched.body,
                },
            )?;
            Ok(Some(HelperFormResult {
                new_helpers: vec![definition],
            }))
        } else if matched.op_name == "namespace".as_bytes().to_vec() {
            let ns = compile_namespace(opts, body.loc(), &matched.orig)?;
            Ok(Some(HelperFormResult {
                new_helpers: vec![ns],
            }))
        } else if matched.op_name == "import".as_bytes().to_vec() {
            let nsref = compile_nsref(body.loc(), &matched.orig)?;
            Ok(Some(HelperFormResult {
                new_helpers: vec![nsref],
            }))
        } else {
            Err(CompileErr(
                matched.body.loc(),
                format!("unknown keyword in helper {body}"),
            ))
        }
    } else {
        Ok(None)
    }
}

trait ModCompileForms {
    fn compile_mod_body(
        &self,
        opts: Rc<dyn CompilerOpts>,
        include_forms: Vec<IncludeDesc>,
        args: Rc<SExp>,
        body: Rc<SExp>,
    ) -> Result<ModAccum, CompileErr>;

    fn compile_mod_helper(
        &self,
        opts: Rc<dyn CompilerOpts>,
        args: Rc<SExp>,
        body: Rc<SExp>,
    ) -> Result<ModAccum, CompileErr>;
}

impl ModCompileForms for ModAccum {
    fn compile_mod_body(
        &self,
        opts: Rc<dyn CompilerOpts>,
        include_forms: Vec<IncludeDesc>,
        args: Rc<SExp>,
        body: Rc<SExp>,
    ) -> Result<ModAccum, CompileErr> {
        Ok(self.set_final(&CompileForm {
            loc: self.loc.clone(),
            args,
            include_forms,
            helpers: self.helpers.clone(),
            exp: Rc::new(compile_bodyform(opts, body)?),
        }))
    }

    fn compile_mod_helper(
        &self,
        opts: Rc<dyn CompilerOpts>,
        _args: Rc<SExp>,
        body: Rc<SExp>,
    ) -> Result<ModAccum, CompileErr> {
        let mut mc = self.clone();
        if let Some(helpers) = compile_helperform(opts.clone(), body.clone())? {
            for form in helpers.new_helpers.iter() {
                mc = mc.add_helper(form.clone());
            }
            Ok(mc)
        } else {
            Err(CompileErr(
                body.loc(),
                "only the last form can be an exprssion in mod".to_string(),
            ))
        }
    }
}

fn frontend_step_finish(
    opts: Rc<dyn CompilerOpts>,
    includes: &mut Vec<IncludeDesc>,
    loc: Srcloc,
    pre_forms: &[Rc<SExp>],
) -> Result<ModAccum, CompileErr> {
    frontend_start(
        opts.clone(),
        includes,
        &[Rc::new(SExp::Cons(
            loc.clone(),
            Rc::new(SExp::Atom(loc.clone(), "mod".as_bytes().to_vec())),
            Rc::new(SExp::Cons(
                loc.clone(),
                Rc::new(SExp::Nil(loc.clone())),
                Rc::new(list_to_cons(loc, pre_forms)),
            )),
        ))],
    )
}

fn frontend_start(
    opts: Rc<dyn CompilerOpts>,
    includes: &mut Vec<IncludeDesc>,
    pre_forms: &[Rc<SExp>],
) -> Result<ModAccum, CompileErr> {
    let top_loc = pre_forms.iter().next().map(|f| f.loc());
    match parse_toplevel_mod(opts.clone(), top_loc.clone(), pre_forms)? {
        ToplevelModParseResult::Mod(tm) => {
            let ls = preprocess(opts.clone(), includes, &tm.forms)?;

            if ls.forms.is_empty() {
                // Empty mod is illegal.
                return Err(CompileErr(
                    top_loc.unwrap_or_else(|| Srcloc::start(&opts.filename())),
                    "Empty module isn't allowed".to_string(),
                ));
            }

            let mut ma = ModAccum::new(tm.loc.clone());
            for form in ls.forms.iter().take(ls.forms.len() - 1) {
                ma = ma.compile_mod_helper(opts.clone(), tm.stripped_args.clone(), form.clone())?;
            }

            ma.compile_mod_body(
                opts.clone(),
                includes.clone(),
                tm.stripped_args,
                ls.forms[ls.forms.len() - 1].clone(),
            )
        }
        ToplevelModParseResult::Simple(l, t) => frontend_step_finish(opts.clone(), includes, l, &t),
    }
}

fn is_namespace_decl(h: &HelperForm) -> bool {
    matches!(h, HelperForm::Defnamespace(_) | HelperForm::Defnsref(_))
}

/// Given the available helper list and the main expression, compute the list of
/// reachable helpers.
pub fn compute_live_helpers(
    opts: Rc<dyn CompilerOpts>,
    helper_list: &[HelperForm],
    main_exp: Rc<BodyForm>,
) -> Vec<HelperForm> {
    let expr_names: HashSet<Vec<u8>> = collect_used_names_bodyform(main_exp.borrow())
        .iter()
        .map(|x| x.to_vec())
        .collect();

    let mut helper_map = HashMap::new();

    for h in helper_list.iter() {
        helper_map.insert(h.name().clone(), h.clone());
    }

    let helper_names = calculate_live_helpers(&HashSet::new(), &expr_names, &helper_map);

    helper_list
        .iter()
        .filter(|h| {
            !opts.frontend_check_live() || is_namespace_decl(h) || helper_names.contains(h.name())
        })
        .cloned()
        .collect()
}

/// Entrypoint for compilation.  This yields a CompileForm which represents a full
/// program.
///
/// Given a CompilerOpts specifying the global options for the compilation, return
/// a representation of the parsed program.  Desugaring is not done in this step
/// so this is a close representation of the user's input, containing location
/// references etc.
///
/// pre_forms is a list of forms, because most SExp readers, including parse_sexp
/// parse a list of complete forms from a source text.  It is possible for frontend
/// to use a list of forms, but it is most often used with a single list in
/// chialisp.  Usually pre_forms will contain a slice containing one list or
/// mod form.
pub fn frontend(
    opts: Rc<dyn CompilerOpts>,
    pre_forms: &[Rc<SExp>],
) -> Result<FrontendOutput, CompileErr> {
    let mut includes = Vec::new();

    if let Some(_dialect) = detect_chialisp_module(Srcloc::start(&opts.filename()), pre_forms)? {
        let mut other_forms = vec![];
        let mut exports = vec![];

        let mut preprocessor = Preprocessor::new(opts.clone());
        let output_forms = preprocessor.run_modules(&mut includes, pre_forms)?;

        if output_forms.forms.is_empty() {
            return Err(CompileErr(
                Srcloc::start(&opts.filename()),
                "Module style chialisp programs require at least one export".to_string(),
            ));
        }

        for form in output_forms.forms.iter() {
            if let Some(export) = match_export_form(opts.clone(), form.clone())? {
                exports.push(export);
            } else if let Some(helper) = compile_helperform_mm(opts.clone(), form.clone(), true)? {
                for h in helper.new_helpers.iter() {
                    other_forms.push(h.clone());
                }
            }
        }

        let loc = output_forms.forms[0].loc();
        let program = CompileForm {
            loc: loc.clone(),
            include_forms: includes.to_vec(),
            args: Rc::new(SExp::Nil(loc.clone())),
            helpers: other_forms.clone(),
            exp: Rc::new(BodyForm::Quoted(SExp::Nil(loc.clone()))),
        };

        return Ok(FrontendOutput::Module(program, exports));
    }

    let started = frontend_start(opts.clone(), &mut includes, pre_forms)?;

    for i in includes.iter() {
        started.add_include(i.clone());
    }

    let compiled: Result<CompileForm, CompileErr> = match started.exp_form {
        None => Err(CompileErr(
            started.loc,
            "mod must end on an expression".to_string(),
        )),
        Some(v) => {
            let compiled_val: &CompileForm = &v;
            Ok(compiled_val.clone())
        }
    };

    let our_mod = rename_children_compileform(&compiled?)?;

    let live_helpers = compute_live_helpers(opts.clone(), &our_mod.helpers, our_mod.exp.clone());

    Ok(FrontendOutput::CompileForm(CompileForm {
        include_forms: includes.to_vec(),
        helpers: live_helpers,
        ..our_mod
    }))
}

fn is_quote_op(sexp: Rc<SExp>) -> bool {
    match sexp.borrow() {
        SExp::Atom(_, name) => name.len() == 1 && name[0] as char == 'q',
        SExp::Integer(_, v) => v == &bi_one(),
        _ => false,
    }
}

fn from_clvm_args(args: Rc<SExp>) -> Rc<SExp> {
    match args.borrow() {
        SExp::Cons(l, arg, rest) => {
            let new_arg = from_clvm(arg.clone());
            let new_rest = from_clvm_args(rest.clone());
            Rc::new(SExp::Cons(l.clone(), new_arg, new_rest))
        }
        _ => {
            // Treat tail of proper application list as expression.
            from_clvm(args.clone())
        }
    }
}

// Form proper frontend code from CLVM.
// The languages are related but not identical:
// - Left env references refer to functions from the env.
// - Right env references refer to user arguments.
// We can introduce defconstant helpers that allow us to keep track of what's
// being called via 'a' and use that information.
// Bare numbers in operator position are only prims.
// Bare numbers in argument position are references, rewrite as (@ ..)
pub fn from_clvm(sexp: Rc<SExp>) -> Rc<SExp> {
    match sexp.borrow() {
        SExp::Atom(l, _name) => {
            // An atom encountered as an expression is treated as a path.
            from_clvm(Rc::new(SExp::Integer(l.clone(), sexp.to_bigint().unwrap())))
        }
        SExp::QuotedString(l, _, _v) => {
            // A string is treated as a number.
            // An atom encountered as an expression is treated as a path.
            from_clvm(Rc::new(SExp::Integer(l.clone(), sexp.to_bigint().unwrap())))
        }
        SExp::Integer(l, _n) => {
            // A number is treated as a reference in expression position.
            // Results in (@ n).
            Rc::new(SExp::Cons(
                l.clone(),
                Rc::new(SExp::atom_from_string(l.clone(), "@")),
                Rc::new(SExp::Cons(
                    l.clone(),
                    sexp.clone(),
                    Rc::new(SExp::Nil(l.clone())),
                )),
            ))
        }
        SExp::Nil(_l) => {
            // Nil represents nil in both systems.
            sexp.clone()
        }
        SExp::Cons(l, op, args) => {
            // This expression represents applying some primitive.
            if is_quote_op(op.clone()) {
                Rc::new(SExp::Cons(
                    l.clone(),
                    Rc::new(SExp::atom_from_string(l.clone(), "q")),
                    args.clone(),
                ))
            } else {
                let new_args = from_clvm_args(args.clone());
                Rc::new(SExp::Cons(l.clone(), op.clone(), new_args))
            }
        }
    }
}
