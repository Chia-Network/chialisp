use std::borrow::Borrow;
use std::cmp::min;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Debug, Error, Formatter};
use std::rc::Rc;

use crate::compiler::clvm::sha256tree;
use crate::compiler::comptypes::{
    Binding, BindingPattern, BodyForm, CompileErr, LambdaData, LetData, LetFormInlineHint,
    LetFormKind,
};
use crate::compiler::evaluate::{is_apply_atom, is_i_atom};
use crate::compiler::frontend::{collect_used_names_bodyform, collect_used_names_sexp};
use crate::compiler::gensym::gensym;
use crate::compiler::lambda::make_cons;
use crate::compiler::optimize::bodyform::{
    path_overlap_one_way, replace_in_bodyform, retrieve_bodyform, visit_detect_in_bodyform,
    BodyformPathArc, PathDetectVisitorResult,
};
use crate::compiler::sexp::{decode_string, SExp};
use crate::compiler::srcloc::Srcloc;

// Common subexpression elimintation.
// catalog subexpressions of the given bodyform and
#[derive(Debug, Clone)]
pub struct CSEInstance {
    pub path: Vec<BodyformPathArc>,
}

#[derive(Debug, Clone)]
pub struct CSEDetectionWithoutConditions {
    pub hash: Vec<u8>,
    pub subexp: BodyForm,
    pub instances: Vec<CSEInstance>,
}

#[derive(Clone)]
pub struct CSEDetection {
    pub hash: Vec<u8>,
    pub root: Vec<BodyformPathArc>,
    pub saturated: bool,
    pub subexp: BodyForm,
    pub instances: Vec<CSEInstance>,
}

impl Debug for CSEDetection {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "CSEDetection {{ hash: {:?}, root: {:?}, saturated: {}, subexp: {}, instances: {:?} }}",
            self.hash,
            self.root,
            self.saturated,
            self.subexp.to_sexp(),
            self.instances
        )
    }
}

#[derive(Debug, Clone)]
pub struct CSECondition {
    pub path: Vec<BodyformPathArc>,
    pub canonical: bool,
}

#[derive(Debug, Clone)]
pub struct BindingStackEntry {
    pub binding: Rc<Binding>,
    pub merge: bool,
}

// in a chain of conditions:
//
// (if a *b *c) // Can precompute.
//
// (if a *b (if c (if d *e *f) h)) // Can't precompute; might not be safe in h.
//
// The question we have to ask for each condition is:
// does every branch use the cse?
//
// If it is used in every branch of a condition, then it dominates that condition
// and it can be computed definitely above the condition.
//
// If it is missing from some downstream elements of a condition, then we must
// pass it on as a lambda that can be applied.
//
fn is_constant(bf: &BodyForm) -> bool {
    matches!(
        bf,
        BodyForm::Value(SExp::Nil(_))
            | BodyForm::Value(SExp::Integer(_, _))
            | BodyForm::Value(SExp::QuotedString(_, _, _))
            | BodyForm::Quoted(_)
    )
}

// A detection is fully dominated if every instance of it is used in the same
// other detection.
fn is_fully_dominated(
    cse: &CSEDetectionWithoutConditions,
    detections: &[CSEDetectionWithoutConditions],
) -> bool {
    let mut host_set = HashSet::new();

    for i in cse.instances.iter() {
        for d in detections.iter() {
            if d.hash == cse.hash {
                continue;
            }
            for d_i in d.instances.iter() {
                if path_overlap_one_way(&d_i.path, &i.path) {
                    host_set.insert(d.hash.clone());
                }
            }
        }
    }

    // No overlaps means all uses are unique, otherwise it is fully dominated if
    // if all uses are in the same host.  If there are multiple hosts then it is
    // not fully dominated since we can still deduplicate it among other hosts
    // which are themselves going to be deduplicated.
    host_set.len() == 1
}

pub fn cse_detect(fe: &BodyForm) -> Result<Vec<CSEDetectionWithoutConditions>, CompileErr> {
    let found_exprs = visit_detect_in_bodyform(
        &|path, _original, form| {
            // The whole expression isn't a repeat.
            if path.is_empty() {
                return Ok(None);
            }

            // Skip the function name of a call.
            if path[path.len() - 1] == BodyformPathArc::CallArgument(0) {
                return Ok(None);
            }

            // Skip individual variable references.
            if matches!(form, BodyForm::Value(SExp::Atom(_, _))) {
                return Ok(None);
            }

            // We can't take a com directly, but we can take parents or children.
            if get_com_body(form).is_some() {
                return Ok(None);
            }

            // Skip cheap constants.
            if is_constant(form) {
                return Ok(None);
            }

            let hash_of = sha256tree(form.to_sexp());
            let res: Result<Option<Vec<u8>>, CompileErr> = Ok(Some(hash_of));
            res
        },
        fe,
    )?;

    // Group them by hash since we've renamed variables.
    let mut by_hash: BTreeMap<Vec<u8>, Vec<PathDetectVisitorResult<Vec<u8>>>> = BTreeMap::new();
    for expr in found_exprs.iter() {
        if let Some(lst) = by_hash.get_mut(&expr.context) {
            lst.push(expr.clone());
        } else {
            by_hash.insert(expr.context.clone(), vec![expr.clone()]);
        }
    }

    let detections: Vec<CSEDetectionWithoutConditions> = by_hash
        .into_iter()
        .filter_map(|(k, v)| {
            if v.len() < 2 {
                return None;
            }

            let subexp = v[0].subexp.clone();

            let instances = v
                .into_iter()
                .map(|item| CSEInstance { path: item.path })
                .collect();

            Some(CSEDetectionWithoutConditions {
                hash: k,
                subexp,
                instances,
            })
        })
        .collect();

    let useful_detections = detections
        .iter()
        .filter(|d| !is_fully_dominated(d, &detections))
        .cloned()
        .collect();

    Ok(useful_detections)
}

// Number of other CSE detections this one depends on.
// We can't apply it until the ones it depends on are applied.
fn number_of_overlaps(detections: &[CSEDetection], cse: &CSEDetection) -> usize {
    cse.instances
        .iter()
        .map(|i: &CSEInstance| {
            detections
                .iter()
                .filter(|d| d.hash != cse.hash)
                .map(|d| {
                    d.instances
                        .iter()
                        .filter(|j: &&CSEInstance| path_overlap_one_way(&i.path, &j.path))
                        .count()
                })
                .sum::<usize>()
        })
        .sum()
}

fn sorted_cse_detections_by_applicability(
    cse_detections: &[CSEDetection],
) -> Vec<(usize, CSEDetection)> {
    let mut detections_with_dependencies: Vec<(usize, CSEDetection)> = cse_detections
        .iter()
        .map(|a| (number_of_overlaps(cse_detections, a), a.clone()))
        .collect();
    detections_with_dependencies.sort_by(|(a, _), (b, _)| a.cmp(b));
    detections_with_dependencies
}

fn is_one_env_ref(bf: &BodyForm) -> bool {
    bf.to_sexp() == Rc::new(SExp::Atom(bf.loc(), vec![1]))
        || bf.to_sexp() == Rc::new(SExp::Atom(bf.loc(), vec![b'@']))
        || bf.to_sexp() == Rc::new(SExp::Atom(bf.loc(), b"@*env*".to_vec()))
}

pub fn is_canonical_apply_parent(
    p: &[BodyformPathArc],
    root: &BodyForm,
) -> Result<bool, CompileErr> {
    if p.is_empty() {
        return Ok(false);
    }

    let last_idx = p.len() - 1;
    if p[last_idx] != BodyformPathArc::CallArgument(1) {
        return Ok(false); // Not the right position in the parent.
    }

    let path_to_parent: Vec<BodyformPathArc> = p.iter().take(last_idx).cloned().collect();
    let parent_exp =
        if let Some(parent) = retrieve_bodyform(&path_to_parent, root, &|bf| bf.clone()) {
            parent
        } else {
            return Err(CompileErr(
                root.loc(),
                format!(
                    "Impossible: could not retrieve parent of existing expression (root {})",
                    root.to_sexp()
                ),
            ));
        };

    // Checking for a primitive, so no tail.
    if let BodyForm::Call(_, parts, None) = &parent_exp {
        if parts.len() != 3 {
            return Ok(false);
        }

        if !is_apply_atom(parts[0].to_sexp()) {
            return Ok(false);
        }

        Ok(is_one_env_ref(&parts[2]))
    } else {
        Ok(false)
    }
}

fn get_com_body(bf: &BodyForm) -> Option<&BodyForm> {
    // Checking for com so no tail.
    if let BodyForm::Call(_, parts, None) = bf {
        if parts.len() != 2 {
            return None;
        }

        if parts[0].to_sexp() != Rc::new(SExp::Atom(bf.loc(), b"com".to_vec())) {
            return None;
        }

        return Some(&parts[1]);
    }

    None
}

// Detect uses of the 'i' operator in chialisp code.
// When written (a (i x (com A) (com B)) 1)
// it is canonical.
pub fn detect_conditions(bf: &BodyForm) -> Result<Vec<CSECondition>, CompileErr> {
    let found_conditions = visit_detect_in_bodyform(
        &|path, root, form| -> Result<Option<bool>, CompileErr> {
            // Must have (a ... 1) surrounding it to be canonical.
            if !is_canonical_apply_parent(path, root)? {
                return Ok(None);
            }

            // Checking for a primitive so no tail.
            if let BodyForm::Call(_, parts, None) = form {
                if parts.len() != 4 {
                    return Ok(None);
                }

                if !is_i_atom(parts[0].to_sexp()) {
                    return Ok(None);
                }

                // We're expecting (com A) and (com B) for the last two
                // arguments.
                // XXX also recognize a tree of (i ...) forms whose leaves
                // are all (com X).
                let a_body = get_com_body(parts[2].borrow());
                let b_body = get_com_body(parts[3].borrow());
                if let (Some(_), Some(_)) = (a_body, b_body) {
                    return Ok(Some(true));
                }

                // It is a proper conditional expression, but not in the
                // canonical form.
                return Ok(Some(false));
            }

            Ok(None)
        },
        bf,
    )?;

    let results = found_conditions
        .into_iter()
        .map(|f| CSECondition {
            path: f.path,
            canonical: f.context,
        })
        .collect();

    Ok(results)
}

// True if for some condition path c_path there are matching instance paths
// for either c_path + [CallArgument(1)] or both
// c_path + [CallArgument(2)] and c_path + [CallArgument(3)]
fn cse_is_covering(c_path: &[BodyformPathArc], instances: &[CSEInstance]) -> bool {
    let mut target_paths = [c_path.to_vec(), c_path.to_vec(), c_path.to_vec()];
    target_paths[0].push(BodyformPathArc::CallArgument(1));
    target_paths[1].push(BodyformPathArc::CallArgument(2));
    target_paths[2].push(BodyformPathArc::CallArgument(3));

    let have_targets: Vec<bool> = target_paths
        .iter()
        .map(|t| instances.iter().any(|i| path_overlap_one_way(t, &i.path)))
        .collect();

    have_targets[0] || (have_targets[1] && have_targets[2])
}

pub fn cse_classify_by_conditions(
    conditions: &[CSECondition],
    detections: &[CSEDetectionWithoutConditions],
) -> Vec<CSEDetection> {
    detections
        .iter()
        .filter_map(|d| {
            // Detect the common root of all instances.
            if d.instances.is_empty() {
                return None;
            }

            let mut path_limit = 0;
            let possible_root = d.instances[0].path.clone();
            for i in d.instances.iter().skip(1) {
                path_limit = min(path_limit, i.path.len());
                for (idx, item) in possible_root.iter().take(path_limit).enumerate() {
                    if &i.path[idx] != item {
                        path_limit = idx;
                        break;
                    }
                }
            }

            // path_limit points to the common root of all instances of this
            // cse detection.
            //
            // now find conditions that are downstream of the cse root.
            let applicable_conditions: Vec<CSECondition> = conditions
                .iter()
                .filter(|c| path_overlap_one_way(&c.path, &possible_root))
                .cloned()
                .collect();

            // We don't need to delay the CSE if 1) all conditions below it
            // are canonical and 2) it appears downstream of all conditions
            // it encloses.
            let fully_canonical = applicable_conditions
                .iter()
                .all(|c| c.canonical && cse_is_covering(&c.path, &d.instances));

            Some(CSEDetection {
                hash: d.hash.clone(),
                subexp: d.subexp.clone(),
                instances: d.instances.clone(),
                saturated: fully_canonical,
                root: possible_root,
            })
        })
        .collect()
}

fn detect_common_cse_root(instances: &[CSEInstance]) -> Vec<BodyformPathArc> {
    // No instances, we can choose the root.
    let min_size = if let Some(m) = instances.iter().map(|i| i.path.len()).min() {
        m
    } else {
        return Vec::new();
    };

    let mut target_path = instances[0].path.clone();
    let mut last_match = min_size;
    for idx in 0..min_size {
        for i in instances.iter() {
            if i.path[idx] != instances[0].path[idx] {
                // If we don't match here, then the common root is up to here.
                last_match = last_match.min(idx);
                target_path = instances[0].path.iter().take(last_match).cloned().collect();
                break;
            }
        }
    }

    // Back it up to the body of a let binding.
    for (idx, f) in target_path.iter().enumerate().rev() {
        if f == &BodyformPathArc::BodyOf {
            return target_path.iter().take(idx + 1).cloned().collect();
        }
    }

    // No internal root if there was no let traversal.
    Vec::new()
}

// Finds lambdas that contain CSE detection instances from the provided list.
fn find_affected_lambdas(
    instances: &[CSEInstance],
    common_root: &[BodyformPathArc],
    bf: &BodyForm,
) -> Result<Vec<PathDetectVisitorResult<()>>, CompileErr> {
    visit_detect_in_bodyform(
        &|path, _root, form| -> Result<Option<()>, CompileErr> {
            // The common root is inside this lambda.
            if path_overlap_one_way(path, common_root) {
                return Ok(None);
            }
            if let BodyForm::Lambda(_) = form {
                if instances
                    .iter()
                    .any(|i| path_overlap_one_way(path, &i.path))
                {
                    return Ok(Some(()));
                }
            }

            Ok(None)
        },
        bf,
    )
}

// Adds a new variable on the left of the lambda captures.
// "x" + (lambda ((& a b) z) ...) -> (lambda ((& x a b) z) ...)
fn add_variable_to_lambda_capture(vn: &[u8], bf: &BodyForm) -> BodyForm {
    let new_var_sexp = SExp::Atom(bf.loc(), vn.to_vec());
    if let BodyForm::Lambda(ldata) = bf {
        let ldata_borrowed: &LambdaData = ldata.borrow();
        let new_captures = Rc::new(make_cons(
            bf.loc(),
            Rc::new(BodyForm::Value(new_var_sexp.clone())),
            ldata.captures.clone(),
        ));
        BodyForm::Lambda(Box::new(LambdaData {
            capture_args: Rc::new(SExp::Cons(
                bf.loc(),
                Rc::new(new_var_sexp),
                ldata.capture_args.clone(),
            )),
            captures: new_captures,
            ..ldata_borrowed.clone()
        }))
    } else {
        bf.clone()
    }
}

#[derive(Clone, Debug)]
struct CSEBindingSite {
    target_path: Vec<BodyformPathArc>,
    binding: Binding,
}

#[derive(Default, Debug)]
struct CSEBindingInfo {
    info: BTreeMap<Vec<BodyformPathArc>, Vec<CSEBindingSite>>,
}

impl CSEBindingInfo {
    fn push(&mut self, site: CSEBindingSite) {
        if let Some(reference) = self.info.get_mut(&site.target_path) {
            reference.push(site.clone());
        } else {
            self.info.insert(site.target_path.clone(), vec![site]);
        }
    }
}

fn detect_merge_into_host_assign(
    target: &[BodyformPathArc],
    body: &BodyForm,
    binding: Rc<Binding>,
) -> bool {
    let root_expr =
        if let Some(root_expr) = retrieve_bodyform(target, body, &|b: &BodyForm| b.clone()) {
            root_expr
        } else {
            return false;
        };

    // Lifting out of a parallel let can't cause bound variables to move out
    // of their scope.
    if let BodyForm::Let(kind, letdata) = &root_expr {
        // Sequential let forms are degraded to parallel let stacks earlier.
        // Parallel let forms don't have interdependent bindings, so no need to
        // treat them here.
        debug_assert!(!matches!(kind, LetFormKind::Sequential));
        if matches!(kind, LetFormKind::Parallel) {
            return false;
        }

        let used_names: HashSet<Vec<u8>> = collect_used_names_bodyform(binding.body.borrow())
            .iter()
            .cloned()
            .collect();

        let mut provided_names: Vec<Vec<u8>> = Vec::new();
        for b in letdata.bindings.iter() {
            match b.pattern.borrow() {
                BindingPattern::Name(name) => {
                    provided_names.push(name.clone());
                }
                BindingPattern::Complex(pat) => {
                    provided_names.append(&mut collect_used_names_sexp(pat.clone()));
                }
            }
        }

        // If one of the bindings defines a name used in the proposed binding,
        // it needs merging.
        return provided_names.iter().any(|p| used_names.contains(p));
    }

    false
}

fn merge_cse_binding(body: &BodyForm, binding: Rc<Binding>) -> BodyForm {
    if let BodyForm::Let(kind, letdata) = body {
        if matches!(kind, LetFormKind::Assign) {
            let mut new_bindings = letdata.bindings.clone();
            new_bindings.push(binding.clone());
            return BodyForm::Let(
                kind.clone(),
                Box::new(LetData {
                    bindings: new_bindings,
                    ..*letdata.clone()
                }),
            );
        }
    }

    body.clone()
}

type CSEReplacementTargetAndBindings<'a> = Vec<&'a (Vec<BodyformPathArc>, Vec<BindingStackEntry>)>;

/// Given a bodyform, CSE analyze and produce a semantically equivalent bodyform
/// that has common expressions removed into assignments to variables prefixed
/// with cse.
///
/// Note: allow_merge is an option only for regression testing.
pub fn cse_optimize_bodyform(
    loc: &Srcloc,
    name: &[u8],
    allow_merge: bool,
    b: &BodyForm,
) -> Result<BodyForm, CompileErr> {
    let conditions = detect_conditions(b)?;
    let cse_raw_detections = cse_detect(b)?;

    let cse_detections = cse_classify_by_conditions(&conditions, &cse_raw_detections);

    // While we have them, apply any detections that overlap no others.
    let mut detections_with_dependencies: Vec<(usize, CSEDetection)> =
        sorted_cse_detections_by_applicability(&cse_detections);

    let mut function_body = b.clone();
    let mut new_binding_stack: Vec<(Vec<BodyformPathArc>, Vec<BindingStackEntry>)> = Vec::new();

    while !detections_with_dependencies.is_empty() {
        let detections_to_apply: Vec<CSEDetection> = detections_with_dependencies
            .iter()
            .take_while(|(c, _d)| *c == 0)
            .map(|(_c, d)| d.clone())
            .collect();
        let keep_detections: Vec<CSEDetection> = detections_with_dependencies
            .iter()
            .skip_while(|(c, _d)| *c == 0)
            .map(|(_c, d)| d.clone())
            .collect();

        // It's an error if applications are deadlocked.
        // I don't think it's possible but this will prevent infinite
        // looping.
        if detections_to_apply.is_empty() && !keep_detections.is_empty() {
            return Err(CompileErr(
                loc.clone(),
                format!("CSE optimization failed in helper {}", decode_string(name)),
            ));
        }

        let mut binding_set: CSEBindingInfo = Default::default();

        for d in detections_to_apply.iter() {
            // If for some reason there are none to apply, we can
            // skip it.  That should not be possible.
            if d.instances.is_empty() {
                break;
            }

            // Skip unsaturated conditional CSE clauses at the moment.
            // This is improvable in the future.
            if !d.saturated {
                continue;
            }

            // All detections should have been transformed equally.
            // Therefore, we can pick one out and use its form.
            //
            // These might have changed from when they were detected
            // because other common subexpressions were substuted.
            let prototype_instance = if let Some(r) =
                retrieve_bodyform(&d.instances[0].path, &function_body, &|b: &BodyForm| {
                    b.clone()
                }) {
                r
            } else {
                return Err(CompileErr(
                    loc.clone(),
                    format!(
                        "CSE Error in {}: could not find form to replace for path {:?}",
                        decode_string(name),
                        d.instances[0].path
                    ),
                ));
            };

            // We'll assign a fresh variable for each of the detections
            // that are applicable now.
            let new_variable_name = gensym(b"cse".to_vec());
            let new_variable_bf_alone = BodyForm::Value(SExp::Atom(
                prototype_instance.loc(),
                new_variable_name.clone(),
            ));

            let new_variable_bf = new_variable_bf_alone;

            let replacement_spec: Vec<PathDetectVisitorResult<()>> = d
                .instances
                .iter()
                .map(|i| PathDetectVisitorResult {
                    path: i.path.clone(),
                    subexp: new_variable_bf.clone(),
                    context: (),
                })
                .collect();

            // Detect the root of the CSE as the innermost expression that covers
            // all uses.
            let replace_path = detect_common_cse_root(&d.instances);

            // Route the captured repeated subexpression into intervening lambdas.
            // This means that the lambdas will gain a capture on the left side of
            // their captures.
            //
            // Ensure that lambdas above replace_path aren't targeted.
            let affected_lambdas = find_affected_lambdas(&d.instances, &replace_path, b)?;
            if let Some(res) = replace_in_bodyform(
                &affected_lambdas,
                &function_body,
                &|_v: &PathDetectVisitorResult<()>, b| {
                    add_variable_to_lambda_capture(&new_variable_name, b)
                },
            ) {
                function_body = res;
            } else {
                return Err(CompileErr(
                    loc.clone(),
                    "error forwarding cse capture into lambda, which should work".to_string(),
                ));
            }

            if let Some(res) = replace_in_bodyform(
                &replacement_spec,
                &function_body,
                &|v: &PathDetectVisitorResult<()>, _b| v.subexp.clone(),
            ) {
                function_body = res;
            } else {
                return Err(CompileErr(
                    loc.clone(),
                    format!(
                        "cse replacement failed in helper {}, which shouldn't be possible",
                        decode_string(name)
                    ),
                ));
            }

            // Put aside the definition in this binding set.
            let name_atom = SExp::Atom(prototype_instance.loc(), new_variable_name.clone());
            binding_set.push(CSEBindingSite {
                target_path: replace_path,
                binding: Binding {
                    loc: prototype_instance.loc(),
                    nl: prototype_instance.loc(),
                    pattern: BindingPattern::Complex(Rc::new(name_atom)),
                    body: Rc::new(prototype_instance),
                },
            });
        }

        detections_with_dependencies = sorted_cse_detections_by_applicability(&keep_detections);

        new_binding_stack.append(
            &mut binding_set
                .info
                .iter()
                .rev()
                .map(|(target_path, sites)| {
                    let bindings: Vec<BindingStackEntry> = sites
                        .iter()
                        .map(|site| {
                            // Detect whether this binding should be merged into its own
                            // host assign form.  That depends on whether
                            // (1) target_path names that assign form. let* forms
                            //   have been broken down by this point into a stack
                            //   of let forms.
                            // (2) it uses bindings from that assign form.
                            let rc_binding = Rc::new(site.binding.clone());
                            let should_merge = allow_merge
                                && detect_merge_into_host_assign(
                                    target_path,
                                    &function_body,
                                    rc_binding.clone(),
                                );
                            BindingStackEntry {
                                binding: rc_binding,
                                merge: should_merge,
                            }
                        })
                        .collect();
                    (target_path.clone(), bindings)
                })
                .collect(),
        );
    }

    // We might not have completely sorted sites anymore due to joining up each
    // site set under a common target path (which themselves need sorting).
    if new_binding_stack.is_empty() {
        return Ok(function_body);
    }

    // We need to topologically sort the CSE insertions by dominance otherwise
    // The inserted let bindings farther up the tree will disrupt lower down
    // replacements.
    //
    // Sort the target paths so we put in deeper paths before outer ones.
    let mut sorted_bindings: Vec<(Vec<BodyformPathArc>, Vec<BindingStackEntry>)> = Vec::new();

    // We'll do this by finding bindings that are not dominated and processing
    // them last.
    while !new_binding_stack.is_empty() {
        let (still_dominated, not_dominated): (
            CSEReplacementTargetAndBindings<'_>,
            CSEReplacementTargetAndBindings<'_>,
        ) = new_binding_stack.iter().partition(|(t, _)| {
            new_binding_stack.iter().any(|(t_other, _)| {
                // t is dominated if t_other contains it.
                t_other != t && path_overlap_one_way(t_other, t)
            })
        });
        let mut not_dominated_vec: Vec<(Vec<BodyformPathArc>, Vec<BindingStackEntry>)> =
            not_dominated.into_iter().cloned().collect();
        sorted_bindings.append(&mut not_dominated_vec);
        let still_dominated_vec: Vec<(Vec<BodyformPathArc>, Vec<BindingStackEntry>)> =
            still_dominated.into_iter().cloned().collect();
        new_binding_stack = still_dominated_vec;
    }

    // All CSE replacements are done.  We unwind the new bindings
    // into a stack of parallel let forms.
    for (target_path, binding_list) in sorted_bindings.into_iter().rev() {
        let replacement_spec = &[PathDetectVisitorResult {
            path: target_path.clone(),
            subexp: function_body.clone(),
            context: (),
        }];
        let (to_merge, not_to_merge): (Vec<&BindingStackEntry>, Vec<&BindingStackEntry>) =
            binding_list.iter().partition(|b| b.merge);

        if let Some(res) = replace_in_bodyform(
            replacement_spec,
            &function_body,
            &|_v: &PathDetectVisitorResult<()>, b| {
                let mut output_body = b.clone();

                // If any bindings need to be merged, merge them.
                // This will not change any code that previously compiled because
                // the result would have previously been a compile error:
                // Unbound use of bound_name_$_238 as a variable name.
                // This is because rename has already happened on the let forms
                // and caused downstream bindings to have names uniquely present
                // in the binding patterns.
                for b in to_merge.iter() {
                    output_body = merge_cse_binding(&output_body, b.binding.clone());
                }

                if not_to_merge.is_empty() {
                    return output_body;
                }

                BodyForm::Let(
                    LetFormKind::Parallel,
                    Box::new(LetData {
                        loc: function_body.loc(),
                        kw: None,
                        inline_hint: Some(LetFormInlineHint::NonInline(loc.clone())),
                        bindings: not_to_merge.iter().map(|b| b.binding.clone()).collect(),
                        body: Rc::new(output_body.clone()),
                    }),
                )
            },
        ) {
            debug_assert!(res.to_sexp() != function_body.to_sexp());
            function_body = res;
        } else {
            return Err(CompileErr(
                function_body.loc(),
                format!(
                    "Could not find the target to replace for path {target_path:?} in {}",
                    b.to_sexp()
                ),
            ));
        }
    }

    Ok(function_body)
}
