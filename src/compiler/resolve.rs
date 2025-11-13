use std::borrow::Borrow;
use std::collections::{BTreeMap, HashSet};
use std::mem::swap;
use std::rc::Rc;

use crate::compiler::codegen::toposort_assign_bindings;
use crate::compiler::compiler::is_at_capture;
use crate::compiler::comptypes::{
    map_m, Binding, BindingPattern, BodyForm, CompileErr, CompileForm, CompilerOpts, DefconstData,
    DefmacData, DefunData, HelperForm, ImportLongName, LambdaData, LetData, LetFormKind,
    LongNameTranslation, ModuleImportListedName, ModuleImportSpec, NamespaceData,
};
use crate::compiler::frontend::HelperFormResult;
use crate::compiler::rename::rename_args_helperform;
use crate::compiler::sexp::{decode_string, SExp};

fn capture_scope(in_scope: &mut HashSet<Vec<u8>>, args: Rc<SExp>) {
    match args.borrow() {
        SExp::Cons(_, a, b) => {
            if let Some((parent, children)) = is_at_capture(a.clone(), b.clone()) {
                in_scope.insert(parent.clone());
                capture_scope(in_scope, children);
            } else {
                capture_scope(in_scope, a.clone());
                capture_scope(in_scope, b.clone());
            }
        }
        SExp::Atom(_, a) => {
            in_scope.insert(a.clone());
        }
        _ => {}
    }
}

pub struct FindNamespaceLookingAtHelpers<'a> {
    hlist: &'a [HelperForm],
    namespace: Option<&'a ImportLongName>,
    offset: usize,
}

pub struct TourNamespaces<'a> {
    helpers: &'a [HelperForm],
    look_stack: Vec<FindNamespaceLookingAtHelpers<'a>>,
}

pub struct FoundHelper<'a> {
    pub helpers: &'a [HelperForm],
    pub namespace: Option<&'a ImportLongName>,
    pub helper: &'a HelperForm,
}

impl<'a> Iterator for TourNamespaces<'a> {
    type Item = FoundHelper<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.look_stack.is_empty() {
                return None;
            }

            let ls_at = self.look_stack.len() - 1;
            let ls_len = self.look_stack[ls_at].hlist.len();

            if self.look_stack[ls_at].offset >= ls_len {
                self.look_stack.pop();
                continue;
            }

            let current_offset = self.look_stack[ls_at].offset;
            let current = &self.look_stack[ls_at].hlist[current_offset];
            self.look_stack[ls_at].offset += 1;

            if let HelperForm::Defnamespace(ns) = current {
                self.look_stack.push(FindNamespaceLookingAtHelpers {
                    hlist: &ns.helpers,
                    namespace: Some(&ns.longname),
                    offset: 0,
                });
                continue;
            }

            return Some(FoundHelper {
                helpers: self.helpers,
                namespace: self.look_stack[ls_at].namespace,
                helper: current,
            });
        }
    }
}

fn namespace_helper(name: &ImportLongName, value: &HelperForm) -> HelperFormResult {
    match value {
        HelperForm::Defun(inline, dd) => HelperFormResult::new(&[HelperForm::Defun(
            *inline,
            Box::new(DefunData {
                name: name.as_u8_vec(LongNameTranslation::Namespace),
                ..*dd.clone()
            }),
        )]),
        HelperForm::Defconstant(dc) => {
            HelperFormResult::new(&[HelperForm::Defconstant(DefconstData {
                name: name.as_u8_vec(LongNameTranslation::Namespace),
                ..dc.clone()
            })])
        }
        HelperForm::Defmacro(dm) => HelperFormResult::new(&[HelperForm::Defmacro(DefmacData {
            name: name.as_u8_vec(LongNameTranslation::Namespace),
            ..dm.clone()
        })]),
        _ => HelperFormResult::new(&[value.clone()]),
    }
}

pub fn tour_helpers(helpers: &[HelperForm]) -> TourNamespaces<'_> {
    TourNamespaces {
        helpers,
        look_stack: vec![FindNamespaceLookingAtHelpers {
            hlist: helpers,
            namespace: None,
            offset: 0,
        }],
    }
}

pub fn rename_args_named_helper(
    pair: (ImportLongName, HelperForm),
) -> Result<(ImportLongName, HelperForm), CompileErr> {
    Ok((pair.0.clone(), rename_args_helperform(&pair.1)?))
}

fn exposed_name_matches(exposed: &ModuleImportListedName, orig_name: &[u8]) -> bool {
    if let Some(alias) = exposed.alias.as_ref() {
        orig_name == alias
    } else {
        orig_name == exposed.name
    }
}

pub fn is_macro_name(name: &ImportLongName) -> bool {
    if name.components.is_empty() {
        return false;
    }

    name.components[name.components.len() - 1].starts_with(b"__chia__defmac__")
}

pub fn find_helper_target(
    opts: Rc<dyn CompilerOpts>,
    helpers: &[HelperForm],
    parent_ns: Option<&ImportLongName>,
    orig_name: &[u8],
    name: &ImportLongName,
) -> Result<Option<(ImportLongName, HelperForm)>, CompileErr> {
    // XXX speed this up, remove iteration.
    // Decompose into parent and child.
    let (parent, child) = name.parent_and_name();

    // Get a list namespace refs from the namespace identified by parent_ns.
    let tour_helpers: Vec<FoundHelper> = tour_helpers(helpers).collect();
    let home_ns: Vec<&FoundHelper> = tour_helpers
        .iter()
        .filter(|found| found.namespace == parent_ns)
        .collect();

    // check the matching namespace to the one specified to see if we can find the
    // target.
    for h in home_ns.iter() {
        if h.helper.name() == &child
            && !matches!(
                h.helper,
                HelperForm::Defnsref(_) | HelperForm::Defnamespace(_)
            )
        {
            // A nsref or namespace doesn't name a helper, so don't match it
            // by name.
            let combined = if let Some(p) = parent_ns {
                p.with_child(&child)
            } else {
                let (_, p) = ImportLongName::parse(&child);
                p
            };
            return Ok(Some((combined, h.helper.clone())));
        }
    }

    // Look at each import specification and construct a target namespace, then
    // try to find a helper in that namespace that matches the target name.

    for ns_spec in home_ns.iter().filter_map(|found| {
        if let HelperForm::Defnsref(nsref) = found.helper {
            Some(nsref.clone())
        } else {
            None
        }
    }) {
        match &ns_spec.specification {
            ModuleImportSpec::Qualified(q) => {
                if let Some(t) = &q.target {
                    // Qualified as [t.name] only matches when we look use the 'as' qualifier.
                    if Some(&t.name) == parent.as_ref() {
                        let target_name = ns_spec.longname.with_child(&child);
                        if let Some(helper) = find_helper_target(
                            opts.clone(),
                            helpers,
                            Some(&ns_spec.longname),
                            orig_name,
                            &target_name,
                        )? {
                            return Ok(Some(rename_args_named_helper(helper)?));
                        }
                    }
                } else {
                    // Qualified namespace matches the canonical name
                    if parent.as_ref() == Some(&ns_spec.longname) {
                        let target_name = ns_spec.longname.with_child(&child);
                        if let Some(helper) = find_helper_target(
                            opts.clone(),
                            helpers,
                            Some(&ns_spec.longname),
                            orig_name,
                            &target_name,
                        )? {
                            return Ok(Some(rename_args_named_helper(helper)?));
                        }
                    }
                }
            }
            ModuleImportSpec::Exposing(_, x) => {
                if parent.is_some() {
                    continue;
                }

                for exposed in x.iter() {
                    if exposed_name_matches(exposed, orig_name) {
                        // If we're matching a macro name, then we must propogate
                        // the search for a macro name.

                        let target_name = if is_macro_name(name) {
                            let (_, child) = name.parent_and_name();
                            ns_spec.longname.with_child(&child)
                        } else {
                            ns_spec.longname.with_child(&exposed.name)
                        };

                        if let Some(helper) = find_helper_target(
                            opts.clone(),
                            helpers,
                            Some(&ns_spec.longname),
                            orig_name,
                            &target_name,
                        )? {
                            return Ok(Some(rename_args_named_helper(helper)?));
                        }
                    }
                }
            }
            ModuleImportSpec::Hiding(_, h) => {
                if parent.is_some() {
                    continue;
                }

                // Hiding means we don't match this name.
                if h.iter().any(|h| h.name == orig_name) {
                    continue;
                }

                let target_name = ns_spec.longname.with_child(&child);
                if let Some(helper) = find_helper_target(
                    opts.clone(),
                    helpers,
                    Some(&ns_spec.longname),
                    orig_name,
                    &target_name,
                )? {
                    return Ok(Some(rename_args_named_helper(helper)?));
                }
            }
        }
    }

    Ok(None)
}

fn display_namespace(parent_ns: Option<&ImportLongName>) -> String {
    if let Some(p) = parent_ns {
        format!(
            "namespace {}",
            decode_string(&p.as_u8_vec(LongNameTranslation::Namespace))
        )
    } else {
        "the root module".to_string()
    }
}

fn is_compiler_builtin(name: &[u8]) -> bool {
    name == b"com" || name == b"@"
}

fn add_binding_names(bindings: &mut HashSet<Vec<u8>>, pattern: &BindingPattern) {
    match pattern {
        BindingPattern::Name(n) => {
            bindings.insert(n.clone());
        }
        BindingPattern::Complex(c) => match c.borrow() {
            SExp::Cons(_, a, b) => {
                add_binding_names(bindings, &BindingPattern::Complex(a.clone()));
                add_binding_names(bindings, &BindingPattern::Complex(b.clone()));
            }
            SExp::Atom(_, a) => {
                bindings.insert(a.clone());
            }
            _ => {}
        },
    }
}

fn resolve_namespaces_in_expr(
    resolved_helpers: &mut BTreeMap<ImportLongName, HelperFormResult>,
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
    parent_ns: Option<&ImportLongName>,
    in_scope: &HashSet<Vec<u8>>,
    expr: Rc<BodyForm>,
) -> Result<Rc<BodyForm>, CompileErr> {
    match expr.borrow() {
        BodyForm::Call(loc, args, tail) => {
            let new_tail = if let Some(t) = tail.as_ref() {
                Some(resolve_namespaces_in_expr(
                    resolved_helpers,
                    opts.clone(),
                    program,
                    parent_ns,
                    in_scope,
                    t.clone(),
                )?)
            } else {
                None
            };

            Ok(Rc::new(BodyForm::Call(
                loc.clone(),
                map_m(
                    |e: &Rc<BodyForm>| {
                        resolve_namespaces_in_expr(
                            resolved_helpers,
                            opts.clone(),
                            program,
                            parent_ns,
                            in_scope,
                            e.clone(),
                        )
                    },
                    args,
                )?,
                new_tail,
            )))
        }
        BodyForm::Value(SExp::Atom(nl, name)) => {
            // if the short name is in scope, we can just return it.
            if in_scope.contains(name) {
                return Ok(expr.clone());
            }

            let (_, parsed_name) = ImportLongName::parse(name);
            let (parent, child) = parsed_name.parent_and_name();

            let (target_full_name, target_helper) = if let Some((target_full_name, target_helper)) =
                find_helper_target(
                    opts.clone(),
                    &program.helpers,
                    parent_ns,
                    name,
                    &parsed_name,
                )? {
                (target_full_name, target_helper)
            } else if is_compiler_builtin(name) {
                return Ok(expr.clone());
            } else {
                // If not namespaced, then it could be a primitive
                if parent.is_none() {
                    let prim_map = opts.prim_map();
                    if prim_map.get(&child).is_some() {
                        return Ok(expr.clone());
                    }

                    let child_sexp = SExp::Atom(nl.clone(), name.clone());
                    for v in prim_map.values() {
                        let v_borrowed: &SExp = v.borrow();
                        if v_borrowed == &child_sexp {
                            return Ok(expr.clone());
                        }
                    }
                }

                eprintln!(
                    "could not find helper {} in {}",
                    decode_string(name),
                    display_namespace(parent_ns)
                );
                return Err(CompileErr(
                    expr.loc(),
                    format!(
                        "could not find helper {} in {}",
                        decode_string(name),
                        display_namespace(parent_ns)
                    ),
                ));
            };

            resolved_helpers.insert(
                target_full_name.clone(),
                HelperFormResult::new(&[rename_args_helperform(&target_helper)?]),
            );
            Ok(Rc::new(BodyForm::Value(SExp::Atom(
                nl.clone(),
                target_full_name.as_u8_vec(LongNameTranslation::Namespace),
            ))))
        }
        BodyForm::Value(_) => Ok(expr.clone()),
        BodyForm::Quoted(_) => Ok(expr.clone()),
        BodyForm::Let(LetFormKind::Sequential, ld) => {
            let mut new_scope = in_scope.clone();
            let mut new_bindings = Vec::new();
            for b in ld.bindings.iter() {
                let b_borrowed: &Binding = b.borrow();
                let new_binding = Binding {
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        &new_scope,
                        b.body.clone(),
                    )?,
                    ..b_borrowed.clone()
                };
                new_bindings.push(Rc::new(new_binding));
                add_binding_names(&mut new_scope, &b.pattern);
            }
            Ok(Rc::new(BodyForm::Let(
                LetFormKind::Sequential,
                Box::new(LetData {
                    bindings: new_bindings,
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        &new_scope,
                        ld.body.clone(),
                    )?,
                    ..*ld.clone()
                }),
            )))
        }
        BodyForm::Let(LetFormKind::Parallel, ld) => {
            let mut new_scope = in_scope.clone();
            let mut new_bindings = Vec::new();
            for b in ld.bindings.iter() {
                let b_borrowed: &Binding = b.borrow();
                let new_binding = Binding {
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        in_scope,
                        b.body.clone(),
                    )?,
                    ..b_borrowed.clone()
                };
                new_bindings.push(Rc::new(new_binding));
                add_binding_names(&mut new_scope, &b.pattern);
            }
            Ok(Rc::new(BodyForm::Let(
                LetFormKind::Sequential,
                Box::new(LetData {
                    bindings: new_bindings,
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        &new_scope,
                        ld.body.clone(),
                    )?,
                    ..*ld.clone()
                }),
            )))
        }
        BodyForm::Let(LetFormKind::Assign, ld) => {
            let mut new_scope = in_scope.clone();
            let mut new_bindings = Vec::new();
            let sorted_bindings = toposort_assign_bindings(&expr.loc(), &ld.bindings)?;
            for b in sorted_bindings.iter() {
                let b_borrowed: &Binding = ld.bindings[b.index].borrow();
                let new_binding = Binding {
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        &new_scope,
                        b_borrowed.body.clone(),
                    )?,
                    ..b_borrowed.clone()
                };
                new_bindings.push(Rc::new(new_binding));
                add_binding_names(&mut new_scope, &b_borrowed.pattern);
            }
            Ok(Rc::new(BodyForm::Let(
                LetFormKind::Assign,
                Box::new(LetData {
                    bindings: new_bindings,
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        &new_scope,
                        ld.body.clone(),
                    )?,
                    ..*ld.clone()
                }),
            )))
        }
        BodyForm::Mod(_, _) => Ok(expr.clone()),
        BodyForm::Lambda(ld) => {
            let new_captures = resolve_namespaces_in_expr(
                resolved_helpers,
                opts.clone(),
                program,
                parent_ns,
                in_scope,
                ld.captures.clone(),
            )?;
            let mut scope_inside_lambda = in_scope.clone();
            capture_scope(&mut scope_inside_lambda, ld.capture_args.clone());
            capture_scope(&mut scope_inside_lambda, ld.args.clone());
            let new_body = resolve_namespaces_in_expr(
                resolved_helpers,
                opts.clone(),
                program,
                parent_ns,
                &scope_inside_lambda,
                ld.body.clone(),
            )?;
            Ok(Rc::new(BodyForm::Lambda(Box::new(LambdaData {
                captures: new_captures,
                body: new_body,
                ..*ld.clone()
            }))))
        }
    }
}

fn resolve_namespaces_in_helper(
    resolved_helpers: &mut BTreeMap<ImportLongName, HelperFormResult>,
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
    parent_ns: Option<&ImportLongName>,
    helper: &HelperForm,
) -> Result<HelperFormResult, CompileErr> {
    match helper {
        HelperForm::Defnamespace(ns) => {
            let combined_ns = if let Some(p) = parent_ns {
                p.combine(&ns.longname)
            } else {
                ns.longname.clone()
            };

            let mut result_helpers = Vec::new();

            for h in ns.helpers.iter() {
                let newly_created = resolve_namespaces_in_helper(
                    resolved_helpers,
                    opts.clone(),
                    program,
                    Some(&combined_ns),
                    h,
                )?;
                result_helpers.extend(newly_created.new_helpers);
            }

            Ok(HelperFormResult::new(&[HelperForm::Defnamespace(
                Box::new(NamespaceData {
                    helpers: result_helpers,
                    ..*ns.clone()
                }),
            )]))
        }
        HelperForm::Defnsref(_) => Ok(HelperFormResult::new(&[helper.clone()])),
        HelperForm::Defun(inline, dd) => {
            let mut in_scope = HashSet::new();
            capture_scope(&mut in_scope, dd.args.clone());
            let new_defun = HelperForm::Defun(
                *inline,
                Box::new(DefunData {
                    body: resolve_namespaces_in_expr(
                        resolved_helpers,
                        opts.clone(),
                        program,
                        parent_ns,
                        &in_scope,
                        dd.body.clone(),
                    )?,
                    ..*dd.clone()
                }),
            );
            Ok(HelperFormResult::new(&[new_defun]))
        }
        HelperForm::Defconstant(dc) => {
            let in_scope = HashSet::new();
            let new_defconst = HelperForm::Defconstant(DefconstData {
                body: resolve_namespaces_in_expr(
                    resolved_helpers,
                    opts.clone(),
                    program,
                    parent_ns,
                    &in_scope,
                    dc.body.clone(),
                )?,
                ..dc.clone()
            });
            Ok(HelperFormResult::new(&[new_defconst]))
        }
        HelperForm::Defmacro(_) => Err(CompileErr(
            helper.loc(),
            "Classic macros are deprecated in module style chialisp".to_string(),
        )),
    }
}

pub fn resolve_namespaces(
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
) -> Result<CompileForm, CompileErr> {
    let mut resolved_helpers: BTreeMap<ImportLongName, HelperFormResult> = BTreeMap::new();
    let mut new_resolved_helpers: BTreeMap<ImportLongName, HelperFormResult> = BTreeMap::new();

    // The main expression is in the scope of the program arguments.
    let mut program_scope = HashSet::new();
    capture_scope(&mut program_scope, program.args.clone());

    let new_expr = resolve_namespaces_in_expr(
        &mut new_resolved_helpers,
        opts.clone(),
        program,
        None,
        &program_scope,
        program.exp.clone(),
    )?;

    // Since we're resolving names now ahead of compilation, take this opportunity
    // to do it definitely by visiting every reachable helper from the main
    // expression.
    while !new_resolved_helpers.is_empty() {
        let mut round_resolved_helpers: BTreeMap<ImportLongName, HelperFormResult> =
            BTreeMap::new();
        for (name, helpers) in new_resolved_helpers.iter() {
            if resolved_helpers.contains_key(name) {
                continue;
            }

            let (parent, _) = name.parent_and_name();

            for helper in helpers.new_helpers.iter() {
                let mut result_helpers = Vec::new();
                let mut renamed_helpers = namespace_helper(name, helper);

                // This is ugly but working: if we have phantom type helpers, we
                // add their individual names as well with no outputs.  This allows
                // the type to be fully expanded each time and each helper to
                // individally ensure that it's only emitted once.
                for h in renamed_helpers.new_helpers.iter() {
                    let full_name = if let Some(p) = parent.as_ref() {
                        p.with_child(h.name())
                    } else {
                        let (_, parsed) = ImportLongName::parse(h.name());
                        parsed
                    };

                    if resolved_helpers.contains_key(&full_name) {
                        continue;
                    }

                    let results = resolve_namespaces_in_helper(
                        &mut round_resolved_helpers,
                        opts.clone(),
                        program,
                        parent.as_ref(),
                        h,
                    )?;

                    result_helpers.extend(results.new_helpers.clone());
                    resolved_helpers.insert(full_name, HelperFormResult::new(&[]));
                }

                renamed_helpers.new_helpers = result_helpers;
                resolved_helpers.insert(name.clone(), renamed_helpers.clone());
            }
        }
        swap(&mut new_resolved_helpers, &mut round_resolved_helpers);
    }

    // The set of helpers is the set of helpers in resolved_helpers al
    let mut all_helpers = Vec::new();
    for v in resolved_helpers.into_values() {
        all_helpers.extend(v.new_helpers);
    }
    Ok(CompileForm {
        helpers: all_helpers,
        exp: new_expr.clone(),
        ..program.clone()
    })
}
