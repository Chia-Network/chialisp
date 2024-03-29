use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use crate::compiler::codegen::codegen;
use crate::compiler::optimize::depgraph::{DepgraphKind, FunctionDependencyGraph};
use crate::compiler::optimize::{sexp_scale, SyntheticType};
use crate::compiler::{BasicCompileContext, CompileErr, CompileForm, CompilerOpts, HelperForm};

// Find the roots for the given function.
fn find_roots(
    visited: &mut HashSet<Vec<u8>>,
    root_set: &mut BTreeSet<Vec<u8>>,
    depgraph: &FunctionDependencyGraph,
    function: &[u8],
) {
    if visited.contains(function) {
        return;
    }

    visited.insert(function.to_vec());

    // If it's non-inline, it's a root.
    if let Some(f) = depgraph.helpers.get(function) {
        if matches!(f.status, DepgraphKind::UserNonInline) {
            root_set.insert(function.to_vec());
            return;
        }
    }

    if let Some(parents) = depgraph.parents(function) {
        for p in parents.iter() {
            find_roots(visited, root_set, depgraph, p);
        }
    }
}

// Should take a desugared program.
pub fn deinline_opt(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    mut compileform: CompileForm,
) -> Result<CompileForm, CompileErr> {
    // Short circuit return: no helpers.
    if compileform.helpers.is_empty() {
        return Ok(compileform);
    }

    let depgraph = FunctionDependencyGraph::new(&compileform);

    let mut best_compileform = compileform.clone();
    let generated_program = codegen(context, opts.clone(), &best_compileform)?;
    let mut metric = sexp_scale(&generated_program);
    let flip_helper = |h: &mut HelperForm| {
        if let HelperForm::Defun(inline, defun) = h {
            if matches!(&defun.synthetic, Some(SyntheticType::NoInlinePreference)) {
                *h = HelperForm::Defun(!*inline, defun.clone());
                return true;
            }
        }

        false
    };

    let helper_to_index: HashMap<Vec<u8>, usize> = compileform
        .helpers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.name().to_vec(), i))
        .collect();

    // defun F -> Synthetic letbinding_$_1
    //            Synthetic letbinding_$_2 -> Synthetic letbinding_$_3
    //
    // defun H_inline ->
    //            Synthetic letbinding_$_4 -> Synthetic letbinding_$_5
    //                                        Synthetic letbinding_$_6
    //
    // defun G -> Synthetic letbinding_$_7 -> H_inline
    //
    // - Synthetic Roots -
    //
    // letbinding_$_1, letbinding_$_2, letbinding_$_7
    // letbinding_$_4 is not a root, because it's in H_inline, called from G.
    //
    // So for each synthetic function, we traverse functions that depend on
    // it as long as it's a synthetic function or a non-synthetic inline.
    // The functions we reach are the roots.
    //
    // If any two roots share dependencies, they must be merged.
    //
    // So we take the set of each root and every synthetic function reachable
    // from it and for each of those sets, we do the normal optimizataion loop.

    // Find leaf synthetic functions by first finding leaf functions, then
    // until we find a synthetic function, go up to each depended_on_by function
    // until we reach a root.
    //
    // Remember the root this function belongs to.
    let leaves: Vec<Vec<u8>> = depgraph
        .leaves()
        .iter()
        .filter(|l| {
            depgraph
                .helpers
                .get(&l.to_vec())
                .map(|l| !matches!(l.status, DepgraphKind::UserNonInline))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    let mut roots: HashMap<Vec<u8>, BTreeSet<Vec<u8>>> = HashMap::new();

    // For each leaf, find roots.
    for l in leaves.iter() {
        let mut visited = HashSet::new();
        let mut leaf_roots = BTreeSet::new();
        find_roots(&mut visited, &mut leaf_roots, &depgraph, l);
        if leaf_roots.is_empty() {
            leaf_roots.insert(l.to_vec());
        }
        roots.insert(l.to_vec(), leaf_roots);
    }

    // Make a set of root sets to coalesce them.
    let mut roots_set: HashSet<BTreeSet<Vec<u8>>> = HashSet::new();
    for (_, common_roots) in roots.iter() {
        roots_set.insert(common_roots.clone());
    }

    // roots is a map from leaf inline to root container.  We can use the roots_set
    // with this collection to make a set of leaves reachable from each root set.
    // Each root set is a set of functions that will change representation when
    // inlining is changed so we have to handle each root set as a unit.
    let mut root_set_to_leaf: HashMap<BTreeSet<Vec<u8>>, BTreeSet<Vec<u8>>> = roots_set
        .iter()
        .map(|root_set| (root_set.clone(), BTreeSet::new()))
        .collect();

    for l in leaves.iter() {
        let root = if let Some(root) = roots.get(l) {
            root.clone()
        } else {
            return Err(CompileErr(
                compileform.loc.clone(),
                "Error in deinline, depgraph gave a leaf that didn't yield a root".to_string(),
            ));
        };

        let from_root_set: Vec<BTreeSet<Vec<u8>>> = roots_set
            .iter()
            .filter(|r| {
                let intersection_of_roots: HashSet<Vec<u8>> =
                    r.intersection(&root).cloned().collect();
                !intersection_of_roots.is_empty()
            })
            .cloned()
            .collect();

        for root_set in from_root_set.iter() {
            if let Some(leaf_set) = root_set_to_leaf.get_mut(root_set) {
                leaf_set.insert(l.to_vec());
            }
        }
    }

    // Now collect the tree of synthetic functions rooted at any of the roots in
    // each root set.
    let root_set_to_inline_tree: HashMap<BTreeSet<Vec<u8>>, HashSet<Vec<u8>>> = root_set_to_leaf
        .iter()
        .map(|(root_set, leaves)| {
            let mut full_tree_set = HashSet::new();
            for root in root_set.iter() {
                let mut full_tree = HashSet::new();
                depgraph.get_full_depends_on(&mut full_tree, root);
                full_tree_set = full_tree.union(&full_tree_set).cloned().collect();
            }
            if full_tree_set.is_empty() {
                full_tree_set = leaves.iter().cloned().collect();
            }
            (root_set.clone(), full_tree_set)
        })
        .collect();

    for (_, function_set) in root_set_to_inline_tree.iter() {
        loop {
            let start_metric = metric;

            for f in function_set.iter() {
                // Get index of helper identified by this leaf name.
                let i = if let Some(i) = helper_to_index.get(f) {
                    *i
                } else {
                    return Err(CompileErr(
                        compileform.loc.clone(),
                        "We have a helper name that has no index?".to_string(),
                    ));
                };

                // Try flipped.
                let old_helper = compileform.helpers[i].clone();
                if !flip_helper(&mut compileform.helpers[i]) {
                    continue;
                }

                let maybe_smaller_program = codegen(context, opts.clone(), &compileform)?;
                let new_metric = sexp_scale(&maybe_smaller_program);

                // Don't keep this change if it made things worse.
                if new_metric >= metric {
                    compileform.helpers[i] = old_helper;
                } else {
                    metric = new_metric;
                    best_compileform = compileform.clone();
                }
            }

            if start_metric == metric {
                break;
            }
        }
    }

    Ok(best_compileform)
}
