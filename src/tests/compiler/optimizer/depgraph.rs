use std::collections::HashSet;

use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::depgraph::FunctionDependencyGraph;
use crate::compiler::sexp::parse_sexp;
use crate::compiler::srcloc::Srcloc;

fn get_depgraph_for_program(prog: &str) -> FunctionDependencyGraph {
    let filename = "*test*";
    let forms = parse_sexp(Srcloc::start(filename), prog.bytes()).expect("should parse");
    let opts = DefaultCompilerOpts::new(filename).set_dialect(AcceptedDialect {
        stepping: Some(21),
        strict: true,
        int_fix: false,
    });
    let compileform = frontend(opts.clone(), &forms).expect("should frontend");

    FunctionDependencyGraph::new(&compileform)
}

#[test]
fn test_dependency_graph_smoke() {
    let depgraph = get_depgraph_for_program(indoc! {"
        (mod (X)
            (include *strict-cl-21*)

            (defun DependedOnByFGAndH (X) (+ 1 X))

            (defun F (X) (DependedOnByFGAndH X))

            (defun H (X) (+ (G X) (* X 2)))

            (defun G (X) (DependedOnByFGAndH X))

            (+ (F X) (G X) (H X))
            )
    "});

    let mut depended_on_by = HashSet::default();
    depgraph.get_full_depended_on_by(&mut depended_on_by, b"DependedOnByFGAndH");

    let want_depended_on_by_set: HashSet<Vec<u8>> = [b"F".to_vec(), b"G".to_vec(), b"H".to_vec()]
        .iter()
        .cloned()
        .collect();
    assert_eq!(want_depended_on_by_set, depended_on_by);

    let mut g_depended_on_by = HashSet::default();
    depgraph.get_full_depended_on_by(&mut g_depended_on_by, b"G");
    let want_g_depended_on_by_set = [b"H".to_vec()].iter().cloned().collect();
    assert_eq!(g_depended_on_by, want_g_depended_on_by_set);

    let mut f_depends_on = HashSet::default();
    depgraph.get_full_depends_on(&mut f_depends_on, b"F");
    let want_f_depends_on_set = [b"DependedOnByFGAndH".to_vec()].iter().cloned().collect();
    assert_eq!(f_depends_on, want_f_depends_on_set);

    let mut h_depends_on = HashSet::default();
    depgraph.get_full_depends_on(&mut h_depends_on, b"H");

    let want_h_depends_on_set = [b"DependedOnByFGAndH".to_vec(), b"G".to_vec()]
        .iter()
        .cloned()
        .collect();
    assert_eq!(h_depends_on, want_h_depends_on_set);
}
