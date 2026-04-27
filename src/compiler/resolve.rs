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
use crate::compiler::rename::rename_args_helperform;
use crate::compiler::sexp::{decode_string, SExp};

/// Ensure that we know the full set of local names referenced by a lambda so we know that they
/// aren't external references.
///
/// This just captures atoms in the argument set.  Captured or internal both count as 'local'
/// here for module resolution purposes.
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

/// A structure that represents our tour through a single namespace during TourNamespaces.
struct FindNamespaceLookingAtHelpers<'a> {
    hlist: &'a [HelperForm],
    namespace: Option<&'a ImportLongName>,
    offset: usize,
}

/// An iterator which finds all the reachable helpers in a namespaced program, allowing them
/// to be selected via outside criteria.
pub struct TourNamespaces<'a> {
    helpers: &'a [HelperForm],
    look_stack: Vec<FindNamespaceLookingAtHelpers<'a>>,
}

/// Gives a candidate helper to match a given name in a namespace, fully specifying the source
/// and original name.
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
                // Note: the scope stack here is to return to the current namespace
                // to continue resolving helpers.  Namespaces act independently as
                // providers of named helpers in the namespace they individually
                // identify.
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

/// Given a helper, rewrite its name to reflect the original source.
///
/// This changes the name of the helper to be fully qualified so that fully qualified references
/// rewritten in other helpers will match when the full program is assembled.
fn namespace_helper(name: &ImportLongName, value: &HelperForm) -> HelperForm {
    match value {
        HelperForm::Defun(inline, dd) => HelperForm::Defun(
            *inline,
            Box::new(DefunData {
                name: name.as_u8_vec(LongNameTranslation::Namespace),
                ..*dd.clone()
            }),
        ),
        HelperForm::Defconstant(dc) => HelperForm::Defconstant(DefconstData {
            name: name.as_u8_vec(LongNameTranslation::Namespace),
            ..dc.clone()
        }),
        HelperForm::Defmacro(dm) => HelperForm::Defmacro(DefmacData {
            name: name.as_u8_vec(LongNameTranslation::Namespace),
            ..dm.clone()
        }),
        _ => value.clone(),
    }
}

/// Produce a traversal of all reachable helpers in the program.
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

/// Given a match name and a name to test against, tell whether the name matches the target rename
/// offered by the import ... exposing directive that contained it.
///
/// Example:
///   (import Foo exposing (bar as baz)), examining (bar as baz) matches baz and not bar.
///   (import Foo exposing bar), examining bar matches bar.
fn exposed_name_matches(exposed: &ModuleImportListedName, orig_name: &[u8]) -> bool {
    if let Some(alias) = exposed.alias.as_ref() {
        orig_name == alias
    } else {
        orig_name == exposed.name
    }
}

/// Macros are renamed during preprocessing, so determine whether the name given is the rename
/// of a macro.
///
/// Macros are moved away from typical names during preprocessing to ensure they don't conflict
/// with anything in the output program when it's produced.  Each macro has access to all functions
/// in the current module and all namespaces referred to by an import that appears lexically earlier
/// in the module.
fn is_macro_name(name: &ImportLongName) -> bool {
    if name.components.is_empty() {
        return false;
    }

    name.components[name.components.len() - 1].starts_with(b"__chia__defmac__")
}

/// Main function which resolves a given short name to a helper retrieved from somewhere in the
/// namespace tree, given the import directives that are active in the current namespace.
///
/// The result is the helper's fully qualified name and the matching helper.
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
        if (parent.is_none() || parent.as_ref() == parent_ns)
            && h.helper.name() == &child
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
                // We already know that qualified imports are external references, the only question
                // is what namespace they refer to.  We unwind it here for positive resolution later
                // on.
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
                            return Ok(Some(helper));
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
                            return Ok(Some(helper));
                        }
                    }
                }
            }
            ModuleImportSpec::Exposing(_, x) => {
                // We don't process short named imports directly at the namespace level, instead
                // resolving imports when a newly imported helper needs resolution.
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
                            return Ok(Some(helper));
                        }
                    }
                }
            }
            ModuleImportSpec::Hiding(_, h) => {
                // We don't process short named imports directly at the namespace level, instead
                // resolving imports when a newly imported helper needs resolution.
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
                    return Ok(Some(helper));
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

/// Applyable function-like operators provided by the compiler.
fn is_compiler_builtin(name: &[u8]) -> bool {
    name == b"com" || name == b"@"
}

/// Find and record binding names in let and assign forms in order to ensure we don't try to match
/// them to external references.
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

/// Given an expression, fully resolve needed helpers from the available namespaces.  The output
/// is a rewritten expression that calls each referenced helper by its fully qualified name.
/// resolved_helpers is left containing the set imports which are used by the expression, indexed
/// by their fully qualified names.
fn resolve_namespaces_in_expr(
    resolved_helpers: &mut BTreeMap<ImportLongName, HelperForm>,
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
                rename_args_helperform(&target_helper)?,
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
                LetFormKind::Parallel,
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
            let mut new_bindings = ld.bindings.clone();
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
                new_bindings[b.index] = Rc::new(new_binding);
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

/// Given a helper, fully resolve needed imported helpers from other namespaces.
/// One step up from resolve_namespaces_in_expr, perform the same task for a whole helper, yielding
/// a version of the helper that refers to everything it depends on by fully qualified name.
///
/// resolved_helpers is left containing anything this helper depends on, indexed by fully qualified
/// name.
fn resolve_namespaces_in_helper(
    resolved_helpers: &mut BTreeMap<ImportLongName, HelperForm>,
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
    parent_ns: Option<&ImportLongName>,
    helper: &HelperForm,
) -> Result<HelperForm, CompileErr> {
    match helper {
        HelperForm::Defnamespace(ns) => {
            let mut result_helpers = Vec::new();

            for h in ns.helpers.iter() {
                let newly_created = resolve_namespaces_in_helper(
                    resolved_helpers,
                    opts.clone(),
                    program,
                    Some(&ns.longname),
                    h,
                )?;
                result_helpers.push(newly_created);
            }

            Ok(HelperForm::Defnamespace(Box::new(NamespaceData {
                helpers: result_helpers,
                ..*ns.clone()
            })))
        }
        HelperForm::Defnsref(_) => Ok(helper.clone()),
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
            Ok(new_defun)
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
            Ok(new_defconst)
        }
        HelperForm::Defmacro(_) => Err(CompileErr(
            helper.loc(),
            "Classic macros are deprecated in module style chialisp".to_string(),
        )),
    }
}

/// Given a program containing namespaces and namespace references, rewrite it so that any impoted
/// helpers from outside namespaces have fully qualified names and include them in the program.
///
/// In practice this takes an input that looks like:
///
/// (mod (A)
///   (namespace N (import M exposing C) (defun F (X) (+ X C)))
///
///   (namespace M (defconst C 1))
///
///   (import qualified N as Z.Y.X)
///
///   (Z.Y.X.F A)
/// )
///
/// And transforms it into
///
/// (mod (A)
///   (defconst M.C 1)
///
///   (defun N.F (X) (+ X M.C))
///
///   (N.F A)
/// )
pub fn resolve_namespaces(
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
) -> Result<CompileForm, CompileErr> {
    let mut resolved_helpers: BTreeMap<ImportLongName, HelperForm> = BTreeMap::new();
    let mut new_resolved_helpers: BTreeMap<ImportLongName, HelperForm> = BTreeMap::new();

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
        let mut round_resolved_helpers: BTreeMap<ImportLongName, HelperForm> = BTreeMap::new();
        for (name, helper) in new_resolved_helpers.iter() {
            if resolved_helpers.contains_key(name) {
                continue;
            }

            let (parent, _) = name.parent_and_name();

            let renamed_helper = namespace_helper(name, helper);

            let result = resolve_namespaces_in_helper(
                &mut round_resolved_helpers,
                opts.clone(),
                program,
                parent.as_ref(),
                &renamed_helper,
            )?;

            resolved_helpers.insert(name.clone(), result.clone());
        }
        swap(&mut new_resolved_helpers, &mut round_resolved_helpers);
    }

    // The set of helpers is the set of helpers in resolved_helpers al
    let mut all_helpers = Vec::new();
    for v in resolved_helpers.into_values() {
        all_helpers.push(v);
    }
    Ok(CompileForm {
        helpers: all_helpers,
        exp: new_expr.clone(),
        ..program.clone()
    })
}
