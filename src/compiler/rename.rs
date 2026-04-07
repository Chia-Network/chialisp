use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;

use crate::compiler::codegen::toposort_assign_bindings;
use crate::compiler::comptypes::{
    map_m, map_m_reverse, Binding, BindingPattern, BodyForm, CompileErr, CompileForm, DefconstData,
    DefmacData, DefunData, HelperForm, LambdaData, LetData, LetFormKind,
};
use crate::compiler::gensym::gensym;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::util::TopoSortItem;

/// Rename in a qq form.  This searches for (unquote ...) forms inside and performs
/// rename inside them, leaving the rest of the qq form as is.
fn rename_in_qq(namemap: &HashMap<Vec<u8>, Vec<u8>>, body: Rc<SExp>) -> Rc<SExp> {
    body.proper_list()
        .and_then(|x| {
            if let [SExp::Atom(_, q), body] = &x[..] {
                if q == b"unquote" {
                    return Some(rename_in_cons(namemap, Rc::new(body.clone()), true));
                }
            }

            None
        })
        .unwrap_or_else(|| match body.borrow() {
            SExp::Cons(l, x, y) => {
                let l_renamed = rename_in_qq(namemap, x.clone());
                let r_renamed = rename_in_qq(namemap, y.clone());
                Rc::new(SExp::Cons(l.clone(), l_renamed, r_renamed))
            }
            _ => body,
        })
}

/* Given a cons cell, rename occurrences of oldname to newname */
pub fn rename_in_cons(
    namemap: &HashMap<Vec<u8>, Vec<u8>>,
    body: Rc<SExp>,
    qq_handling: bool,
) -> Rc<SExp> {
    match body.borrow() {
        SExp::Atom(l, name) => match namemap.get(name) {
            Some(v) => Rc::new(SExp::Atom(l.clone(), v.to_vec())),
            None => body,
        },
        SExp::Cons(l, f, r) => {
            if let SExp::Atom(la, q) = f.borrow() {
                if q == b"q" {
                    return Rc::new(SExp::Cons(
                        l.clone(),
                        Rc::new(SExp::Atom(la.clone(), "q".as_bytes().to_vec())),
                        r.clone(),
                    ));
                } else if *q == "quote".as_bytes().to_vec() {
                    return r
                        .proper_list()
                        .map(|x| match &x[..] {
                            [v] => Rc::new(SExp::Cons(
                                l.clone(),
                                Rc::new(SExp::atom_from_string(la.clone(), "quote")),
                                Rc::new(SExp::Cons(
                                    v.loc(),
                                    Rc::new(v.clone()),
                                    Rc::new(SExp::Nil(v.loc())),
                                )),
                            )),
                            _ => body.clone(),
                        })
                        .unwrap_or_else(|| body.clone());
                } else if *q == "qq".as_bytes().to_vec() && qq_handling {
                    return r
                        .proper_list()
                        .map(|x| match &x[..] {
                            [qqexpr] => rename_in_qq(namemap, Rc::new(qqexpr.clone())),
                            _ => body.clone(),
                        })
                        .unwrap_or_else(|| body.clone());
                }
            }

            Rc::new(SExp::Cons(
                l.clone(),
                rename_in_cons(namemap, f.clone(), qq_handling),
                rename_in_cons(namemap, r.clone(), qq_handling),
            ))
        }
        _ => body.clone(),
    }
}

/* Returns a list of pairs containing the old and new atom names */
fn invent_new_names_sexp(body: Rc<SExp>) -> Vec<(Vec<u8>, Vec<u8>)> {
    match body.borrow() {
        SExp::Atom(_, name) => {
            if name != b"@" {
                vec![(name.to_vec(), gensym(name.to_vec()))]
            } else {
                vec![]
            }
        }
        SExp::Cons(_, head, tail) => {
            let mut head_list = invent_new_names_sexp(head.clone());
            let mut tail_list = invent_new_names_sexp(tail.clone());
            head_list.append(&mut tail_list);
            head_list
        }
        _ => {
            vec![]
        }
    }
}

#[derive(Debug, Clone)]
struct InnerRenameList {
    bindings: HashMap<Vec<u8>, Vec<u8>>,
    from_wing: Binding,
}

fn make_binding_unique(b: &Binding) -> InnerRenameList {
    match &b.pattern {
        BindingPattern::Name(name) => {
            let mut single_name_map = HashMap::new();
            let new_name = gensym(name.clone());
            single_name_map.insert(name.to_vec(), new_name.clone());
            InnerRenameList {
                bindings: single_name_map,
                from_wing: Binding {
                    pattern: BindingPattern::Name(new_name),
                    ..b.clone()
                },
            }
        }
        BindingPattern::Complex(pat) => {
            let new_names_vec = invent_new_names_sexp(pat.clone());
            let mut new_names = HashMap::new();

            for (n, v) in new_names_vec.iter() {
                new_names.insert(n.clone(), v.clone());
            }

            let renamed_pattern = rename_in_cons(&new_names, pat.clone(), false);
            InnerRenameList {
                bindings: new_names,
                from_wing: Binding {
                    pattern: BindingPattern::Complex(renamed_pattern),
                    ..b.clone()
                },
            }
        }
    }
}

pub fn rename_assign_bindings(
    l: &Srcloc,
    bindings: &[Rc<Binding>],
    body: Rc<BodyForm>,
) -> Result<(BodyForm, Vec<Rc<Binding>>), CompileErr> {
    // Order the bindings.
    let sorted_bindings = toposort_assign_bindings(l, bindings)?;
    let mut renames: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    // Process in reverse order so we rename from inner to outer.
    let bindings_to_rename: Vec<TopoSortItem<_>> = sorted_bindings.to_vec();
    let renamed_bindings = map_m_reverse(
        |item: &TopoSortItem<_>| -> Result<Rc<Binding>, CompileErr> {
            let b: &Binding = bindings[item.index].borrow();
            if let BindingPattern::Complex(p) = &b.pattern {
                let new_names = invent_new_names_sexp(p.clone());
                for (name, renamed) in new_names.iter() {
                    renames.insert(name.clone(), renamed.clone());
                }

                let renamed_in_body = Rc::new(rename_args_bodyform(b.body.borrow())?);
                Ok(Rc::new(Binding {
                    pattern: BindingPattern::Complex(rename_in_cons(&renames, p.clone(), false)),
                    body: Rc::new(rename_in_bodyform(&renames, renamed_in_body)?),
                    ..b.clone()
                }))
            } else {
                Ok(bindings[item.index].clone())
            }
        },
        &bindings_to_rename,
    )?;
    let new_body = Rc::new(rename_args_bodyform(body.borrow())?);
    Ok((rename_in_bodyform(&renames, new_body)?, renamed_bindings))
}

fn rename_in_bodyform(
    namemap: &HashMap<Vec<u8>, Vec<u8>>,
    b: Rc<BodyForm>,
) -> Result<BodyForm, CompileErr> {
    match b.borrow() {
        BodyForm::Let(kind, letdata) => {
            let new_bindings = map_m(
                &|b: &Rc<Binding>| -> Result<Rc<Binding>, CompileErr> {
                    let b_borrowed: &Binding = b.borrow();
                    Ok(Rc::new(Binding {
                        body: Rc::new(rename_in_bodyform(namemap, b.body.clone())?),
                        ..b_borrowed.clone()
                    }))
                },
                &letdata.bindings,
            )?;
            let new_body = rename_in_bodyform(namemap, letdata.body.clone())?;
            Ok(BodyForm::Let(
                kind.clone(),
                Box::new(LetData {
                    bindings: new_bindings,
                    body: Rc::new(new_body),
                    ..*letdata.clone()
                }),
            ))
        }

        BodyForm::Quoted(atom) => match atom {
            SExp::Atom(l, n) => match namemap.get(n) {
                Some(named) => Ok(BodyForm::Quoted(SExp::Atom(l.clone(), named.to_vec()))),
                None => Ok(BodyForm::Quoted(atom.clone())),
            },
            _ => Ok(BodyForm::Quoted(atom.clone())),
        },

        BodyForm::Value(atom) => match atom {
            SExp::Atom(l, n) => match namemap.get(n) {
                Some(named) => Ok(BodyForm::Value(SExp::Atom(l.clone(), named.to_vec()))),
                None => Ok(BodyForm::Value(atom.clone())),
            },
            _ => Ok(BodyForm::Value(atom.clone())),
        },

        BodyForm::Call(l, vs, tail) => {
            let mut new_vs = map_m(
                &|x: &Rc<BodyForm>| -> Result<Rc<BodyForm>, CompileErr> {
                    Ok(Rc::new(rename_in_bodyform(namemap, x.clone())?))
                },
                vs,
            )?;
            // Ensure that we haven't renamed the 0th element of a call
            // They exist in a separate (global) namespace of callables
            // and aren't in the variable scope stack.
            if !vs.is_empty() {
                new_vs[0] = vs[0].clone();
            }
            let new_tail = if let Some(t) = tail.as_ref() {
                Some(Rc::new(rename_in_bodyform(namemap, t.clone())?))
            } else {
                None
            };
            Ok(BodyForm::Call(l.clone(), new_vs, new_tail))
        }

        BodyForm::Mod(l, prog) => Ok(BodyForm::Mod(l.clone(), prog.clone())),
        // Rename lambda arguments down the lexical scope.
        BodyForm::Lambda(ldata) => {
            let renamed_capture_inputs =
                Rc::new(rename_in_bodyform(namemap, ldata.captures.clone())?);
            let renamed_capture_outputs =
                rename_in_cons(namemap, ldata.capture_args.clone(), false);
            let renamed_body = Rc::new(rename_args_bodyform(ldata.body.borrow())?);
            let outer_renamed_body = rename_in_bodyform(namemap, renamed_body)?;
            Ok(BodyForm::Lambda(Box::new(LambdaData {
                captures: renamed_capture_inputs,
                capture_args: renamed_capture_outputs,
                body: Rc::new(outer_renamed_body),
                ..*ldata.clone()
            })))
        }
    }
}

/// Given a set of sequential bindings, create a stack of let forms that have
/// the same meaning.  This is used to propogate renaming.
pub fn desugar_sequential_let_bindings(
    bindings: &[Rc<Binding>],
    body: &BodyForm,
    n: usize, // Zero is for post-termination
) -> BodyForm {
    if n == 0 {
        body.clone()
    } else {
        let want_binding = bindings[n - 1].clone();
        desugar_sequential_let_bindings(
            bindings,
            &BodyForm::Let(
                LetFormKind::Parallel,
                Box::new(LetData {
                    loc: want_binding.loc(),
                    kw: None,
                    bindings: vec![want_binding],
                    inline_hint: None,
                    body: Rc::new(body.clone()),
                }),
            ),
            n - 1,
        )
    }
}

fn rename_args_bodyform(b: &BodyForm) -> Result<BodyForm, CompileErr> {
    match b {
        BodyForm::Let(LetFormKind::Sequential, letdata) => {
            // Renaming a sequential let is exactly as if the bindings were
            // nested in separate parallel lets.
            let res = rename_args_bodyform(&desugar_sequential_let_bindings(
                &letdata.bindings,
                letdata.body.borrow(),
                letdata.bindings.len(),
            ))?;
            Ok(res)
        }

        BodyForm::Let(LetFormKind::Parallel, letdata) => {
            let renames: Vec<InnerRenameList> = letdata
                .bindings
                .iter()
                .map(|x| make_binding_unique(x.borrow()))
                .collect();
            let new_renamed_bindings: Vec<Rc<Binding>> = renames
                .iter()
                .map(|ir| Rc::new(ir.from_wing.clone()))
                .collect();
            let mut local_namemap = HashMap::new();
            for ir in renames.iter() {
                for (oldname, binding_name) in ir.bindings.iter() {
                    local_namemap.insert(oldname.to_vec(), binding_name.clone());
                }
            }
            let new_bindings = map_m(
                &|x: &Rc<Binding>| -> Result<Rc<Binding>, CompileErr> {
                    Ok(Rc::new(Binding {
                        loc: x.loc.clone(),
                        nl: x.nl.clone(),
                        pattern: x.pattern.clone(),
                        body: Rc::new(rename_args_bodyform(&x.body)?),
                    }))
                },
                &new_renamed_bindings,
            )?;
            let args_renamed = rename_args_bodyform(letdata.body.borrow())?;
            let locally_renamed_body = rename_in_bodyform(&local_namemap, Rc::new(args_renamed))?;
            let new_form = BodyForm::Let(
                LetFormKind::Parallel,
                Box::new(LetData {
                    bindings: new_bindings,
                    body: Rc::new(locally_renamed_body),
                    ..*letdata.clone()
                }),
            );
            Ok(new_form)
        }

        BodyForm::Let(LetFormKind::Assign, letdata) => {
            let (new_compiled_body, new_bindings) =
                rename_assign_bindings(&letdata.loc, &letdata.bindings, letdata.body.clone())?;
            Ok(BodyForm::Let(
                LetFormKind::Assign,
                Box::new(LetData {
                    body: Rc::new(new_compiled_body),
                    bindings: new_bindings,
                    ..*letdata.clone()
                }),
            ))
        }

        BodyForm::Quoted(e) => Ok(BodyForm::Quoted(e.clone())),
        BodyForm::Value(v) => Ok(BodyForm::Value(v.clone())),

        BodyForm::Call(l, vs, tail) => {
            let new_vs = map_m(
                &|a: &Rc<BodyForm>| -> Result<Rc<BodyForm>, CompileErr> {
                    Ok(Rc::new(rename_args_bodyform(a)?))
                },
                vs,
            )?;
            let new_tail = if let Some(t) = tail.as_ref() {
                Some(Rc::new(rename_args_bodyform(t.borrow())?))
            } else {
                None
            };
            Ok(BodyForm::Call(l.clone(), new_vs, new_tail))
        }
        BodyForm::Mod(l, program) => Ok(BodyForm::Mod(l.clone(), program.clone())),
        BodyForm::Lambda(ldata) => {
            let mut own_args = HashMap::new();
            for (n, v) in invent_new_names_sexp(ldata.args.clone()).iter() {
                own_args.insert(n.clone(), v.clone());
            }
            let new_args = rename_in_cons(&own_args, ldata.args.clone(), false);
            let new_body = rename_args_bodyform(ldata.body.borrow())?;
            let renamed_with_own_args = rename_in_bodyform(&own_args, Rc::new(new_body))?;
            Ok(BodyForm::Lambda(Box::new(LambdaData {
                args: new_args,
                body: Rc::new(renamed_with_own_args),
                ..*ldata.clone()
            })))
        }
    }
}

fn rename_in_helperform(
    namemap: &HashMap<Vec<u8>, Vec<u8>>,
    h: &HelperForm,
) -> Result<HelperForm, CompileErr> {
    match h {
        HelperForm::Defconstant(defc) => Ok(HelperForm::Defconstant(DefconstData {
            body: Rc::new(rename_in_bodyform(namemap, defc.body.clone())?),
            ..defc.clone()
        })),
        HelperForm::Defmacro(mac) => Ok(HelperForm::Defmacro(DefmacData {
            program: Rc::new(rename_in_compileform(namemap, mac.program.clone())?),
            ..mac.clone()
        })),
        HelperForm::Defun(inline, defun) => Ok(HelperForm::Defun(
            *inline,
            Box::new(DefunData {
                body: Rc::new(rename_in_bodyform(namemap, defun.body.clone())?),
                ..*defun.clone()
            }),
        )),
    }
}

pub fn rename_args_helperform(h: &HelperForm) -> Result<HelperForm, CompileErr> {
    match h {
        HelperForm::Defconstant(defc) => Ok(HelperForm::Defconstant(DefconstData {
            body: Rc::new(rename_args_bodyform(defc.body.borrow())?),
            ..defc.clone()
        })),
        HelperForm::Defmacro(mac) => {
            let mut new_names: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
            for x in invent_new_names_sexp(mac.args.clone()).iter() {
                new_names.insert(x.0.clone(), x.1.clone());
            }
            let mut local_namemap = HashMap::new();
            for x in new_names.iter() {
                local_namemap.insert(x.0.to_vec(), x.1.to_vec());
            }
            let local_renamed_arg = rename_in_cons(&local_namemap, mac.args.clone(), true);
            let local_renamed_body = rename_args_compileform(mac.program.borrow())?;
            Ok(HelperForm::Defmacro(DefmacData {
                args: local_renamed_arg,
                program: Rc::new(rename_in_compileform(
                    &local_namemap,
                    Rc::new(local_renamed_body),
                )?),
                ..mac.clone()
            }))
        }
        HelperForm::Defun(inline, defun) => {
            let new_names = invent_new_names_sexp(defun.args.clone());
            let mut local_namemap = HashMap::new();
            for x in new_names.iter() {
                local_namemap.insert(x.0.clone(), x.1.clone());
            }
            let local_renamed_arg = rename_in_cons(&local_namemap, defun.args.clone(), true);
            let local_renamed_body = rename_args_bodyform(defun.body.borrow())?;
            Ok(HelperForm::Defun(
                *inline,
                Box::new(DefunData {
                    args: local_renamed_arg,
                    body: Rc::new(rename_in_bodyform(
                        &local_namemap,
                        Rc::new(local_renamed_body),
                    )?),
                    ..*defun.clone()
                }),
            ))
        }
    }
}

fn rename_in_compileform(
    namemap: &HashMap<Vec<u8>, Vec<u8>>,
    c: Rc<CompileForm>,
) -> Result<CompileForm, CompileErr> {
    let c_ref: &CompileForm = c.borrow();
    Ok(CompileForm {
        helpers: map_m(|x| rename_in_helperform(namemap, x), &c.helpers)?,
        exp: Rc::new(rename_in_bodyform(namemap, c.exp.clone())?),
        ..c_ref.clone()
    })
}

/// For all the HelperForms in a CompileForm, do renaming in them so that all
/// unique variable bindings in the program have unique names.
pub fn rename_children_compileform(c: &CompileForm) -> Result<CompileForm, CompileErr> {
    let local_renamed_helpers = map_m(&rename_args_helperform, &c.helpers)?;
    let local_renamed_body = rename_args_bodyform(c.exp.borrow())?;
    Ok(CompileForm {
        helpers: local_renamed_helpers,
        exp: Rc::new(local_renamed_body),
        ..c.clone()
    })
}

/// Given a compileform, perform renaming in descendants so that every variable
/// name that lives in a different scope has a unique name.  This allows
/// compilation to treat identical forms as equivalent and ensures that forms
/// that look the same but refer to different variables are different.  It also
/// ensures that future tricky variable name uses decide on one binding from their
/// lexical scope.
pub fn rename_args_compileform(c: &CompileForm) -> Result<CompileForm, CompileErr> {
    let new_names = invent_new_names_sexp(c.args.clone());
    let mut local_namemap = HashMap::new();
    for x in new_names.iter() {
        local_namemap.insert(x.0.clone(), x.1.clone());
    }
    let local_renamed_arg = rename_in_cons(&local_namemap, c.args.clone(), true);
    let local_renamed_helpers: Vec<HelperForm> = map_m(&rename_args_helperform, &c.helpers)?;
    let local_renamed_body = rename_args_bodyform(c.exp.borrow())?;
    Ok(CompileForm {
        loc: c.loc(),
        args: local_renamed_arg,
        include_forms: c.include_forms.clone(),
        helpers: map_m(
            |x| rename_in_helperform(&local_namemap, x),
            &local_renamed_helpers,
        )?,
        exp: Rc::new(rename_in_bodyform(
            &local_namemap,
            Rc::new(local_renamed_body),
        )?),
    })
}
