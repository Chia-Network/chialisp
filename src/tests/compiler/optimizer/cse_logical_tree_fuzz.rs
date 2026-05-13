use rand::prelude::*;
use std::collections::BTreeMap;
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::tests::compiler::fuzz::{
    compose_sexp, perform_compile_of_file, simple_run, simple_seeded_rng,
};

const GENERATED_LOGICAL_TREE_PROGRAMS: u32 = 120;
const MAX_SPEC_DEPTH: u8 = 6;
const MAX_BINDING_STACK_DEPTH: usize = 6;

#[derive(Clone, Debug, Eq, PartialEq)]
enum LogicalTreeShape {
    Scalar {
        path: u64,
        value: bool,
    },
    Condition {
        path: u64,
        value: bool,
        index: usize,
    },
}

impl LogicalTreeShape {
    fn path(&self) -> u64 {
        match self {
            LogicalTreeShape::Scalar { path, .. } => *path,
            LogicalTreeShape::Condition { path, .. } => *path,
        }
    }

    fn value(&self) -> bool {
        match self {
            LogicalTreeShape::Scalar { value, .. } => *value,
            LogicalTreeShape::Condition { value, .. } => *value,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum LogicalTree {
    Nil,
    Atom(bool),
    Pair(Box<LogicalTree>, Box<LogicalTree>),
}

impl LogicalTree {
    fn to_program_text(&self) -> String {
        match self {
            LogicalTree::Nil => "()".to_string(),
            LogicalTree::Atom(value) => bool_atom(*value),
            LogicalTree::Pair(first, rest) => {
                format!("({} . {})", first.to_program_text(), rest.to_program_text())
            }
        }
    }
}

#[derive(Clone, Debug)]
struct LogicalTreeCase {
    specs: Vec<LogicalTreeShape>,
    satisfied: Vec<bool>,
    selected_index: usize,
    argument: LogicalTree,
}

#[derive(Clone, Debug)]
struct RenderedProgram {
    program: String,
    expected: String,
    run_args: String,
}

#[derive(Clone, Debug)]
struct ProgramRenderer {
    specs: Vec<LogicalTreeShape>,
    srcloc: Srcloc,
    next_variable: usize,
}

impl ProgramRenderer {
    fn new(specs: Vec<LogicalTreeShape>) -> Self {
        ProgramRenderer {
            specs,
            srcloc: Srcloc::start("*cse-logical-tree-fuzz*"),
            next_variable: 0,
        }
    }

    fn fresh_variable(&mut self) -> String {
        let name = format!("v{}", self.next_variable);
        self.next_variable += 1;
        name
    }

    fn wrap_expression<R: Rng>(&mut self, rng: &mut R, expression: String) -> String {
        let mut result = expression;
        let stack_depth = rng.random_range(0..=MAX_BINDING_STACK_DEPTH);
        for _ in 0..stack_depth {
            result = match rng.random_range(0..4) {
                0 => {
                    let var = self.fresh_variable();
                    format!("(let (({var} {result})) {var})")
                }
                1 => {
                    let first = self.fresh_variable();
                    let second = self.fresh_variable();
                    format!("(let* (({first} {result}) ({second} {first})) {second})")
                }
                2 => {
                    let var = self.fresh_variable();
                    format!("(assign {var} {result} {var})")
                }
                _ => format!("(if 1 {result} {result})"),
            };
        }

        result
    }

    fn safe_path_equals<R: Rng>(&mut self, rng: &mut R, path: u64, value: bool) -> String {
        let comparison = format!("(= {} {})", path_expr(path, "X"), bool_atom(value));
        let guarded = path_prefixes(path)
            .into_iter()
            .rev()
            .fold(comparison, |inner, prefix| {
                format!("(if (l {}) {inner} 0)", path_expr(prefix, "X"))
            });
        self.wrap_expression(rng, guarded)
    }

    fn entry_is_one<R: Rng>(
        &mut self,
        rng: &mut R,
        index: usize,
        memo: &mut BTreeMap<usize, String>,
    ) -> String {
        if let Some(existing) = memo.get(&index) {
            return existing.clone();
        }

        let shape = self.specs[index].clone();
        let result = if !shape.value() {
            "0".to_string()
        } else {
            match shape {
                LogicalTreeShape::Scalar { path, .. } => self.safe_path_equals(rng, path, true),
                LogicalTreeShape::Condition { .. } => self.condition_true(rng, index, memo),
            }
        };

        memo.insert(index, result.clone());
        result
    }

    fn condition_true<R: Rng>(
        &mut self,
        rng: &mut R,
        index: usize,
        memo: &mut BTreeMap<usize, String>,
    ) -> String {
        let LogicalTreeShape::Condition {
            path,
            value,
            index: prerequisite,
        } = self.specs[index].clone()
        else {
            return self.safe_path_equals(rng, self.specs[index].path(), self.specs[index].value());
        };

        let prerequisite_true = self.entry_is_one(rng, prerequisite, memo);
        let this_condition = self.safe_path_equals(rng, path, value);
        let combined = if rng.random_bool(0.5) {
            format!("(and {prerequisite_true} {this_condition})")
        } else {
            format!("(if {prerequisite_true} {this_condition} 0)")
        };

        self.wrap_expression(rng, combined)
    }

    fn maybe_value_list<R: Rng>(
        &mut self,
        rng: &mut R,
        index: usize,
        memo: &mut BTreeMap<usize, String>,
    ) -> String {
        let shape = self.specs[index].clone();
        let result = self.wrap_expression(rng, path_expr(shape.path(), "X"));
        let present = match shape {
            LogicalTreeShape::Scalar { path, .. } => {
                self.safe_path_equals(rng, path, self.specs[index].value())
            }
            LogicalTreeShape::Condition { .. } => self.condition_true(rng, index, memo),
        };

        if rng.random_bool(0.5) {
            format!("(if {present} (list {result}) ())")
        } else {
            format!("(if (not {present}) () (list {result}))")
        }
    }

    fn condition_values_expression<R: Rng>(&mut self, rng: &mut R) -> String {
        let mut result = "()".to_string();
        for index in (0..self.specs.len()).rev() {
            if matches!(self.specs[index], LogicalTreeShape::Condition { .. }) {
                let mut memo = BTreeMap::new();
                let condition = self.maybe_value_list(rng, index, &mut memo);
                result = format!("(append {condition} {result})");
            }
        }

        self.wrap_expression(rng, result)
    }

    fn render<R: Rng>(&mut self, rng: &mut R, case: &LogicalTreeCase) -> RenderedProgram {
        let mut selected_memo = BTreeMap::new();
        let selected = self.maybe_value_list(rng, case.selected_index, &mut selected_memo);
        let condition_values = self.condition_values_expression(rng);
        let body = self.wrap_expression(rng, format!("(c {selected} {condition_values})"));
        let program = format!(
            "(mod (X)
  (include *standard-cl-23*)
  (defun append (A B) (if A (c (f A) (append (r A) B)) B))
  {body}
)"
        );

        RenderedProgram {
            program,
            expected: expected_result_text(case),
            run_args: format!("({})", case.argument.to_program_text()),
        }
    }
}

fn bool_atom(value: bool) -> String {
    if value {
        "1".to_string()
    } else {
        "0".to_string()
    }
}

fn path_depth(path: u64) -> u32 {
    u64::BITS - path.leading_zeros() - 1
}

fn is_ancestor(ancestor: u64, mut path: u64) -> bool {
    while path > ancestor {
        path >>= 1;
    }

    path == ancestor
}

fn conflicts_with_existing_path(path: u64, paths: &[u64]) -> bool {
    paths
        .iter()
        .any(|existing| is_ancestor(*existing, path) || is_ancestor(path, *existing))
}

fn random_path<R: Rng>(rng: &mut R) -> u64 {
    let depth = rng.random_range(1..=MAX_SPEC_DEPTH);
    let mut path = 1_u64;
    for _ in 0..depth {
        path = (path << 1) | u64::from(rng.random_bool(0.5));
    }

    path
}

fn path_prefixes(path: u64) -> Vec<u64> {
    let mut prefixes = Vec::new();
    let mut current = path;
    while current > 1 {
        current >>= 1;
        prefixes.push(current);
    }
    prefixes.reverse();
    prefixes
}

fn path_expr(path: u64, base: &str) -> String {
    let mut result = base.to_string();
    for bit in (0..path_depth(path)).rev() {
        if ((path >> bit) & 1) == 0 {
            result = format!("(f {result})");
        } else {
            result = format!("(r {result})");
        }
    }

    result
}

fn generate_specs<R: Rng>(rng: &mut R) -> Vec<LogicalTreeShape> {
    let target_len = rng.random_range(5..=12);
    let mut paths = Vec::new();
    while paths.len() < target_len {
        let path = random_path(rng);
        if !conflicts_with_existing_path(path, &paths) {
            paths.push(path);
        }
    }

    // Build the list deepest first, so a path that would be a parent of another
    // specified scalar is rejected before the argument tree is constructed.
    paths.sort_by_key(|path| std::cmp::Reverse(path_depth(*path)));

    let mut specs = Vec::new();
    for (index, path) in paths.into_iter().enumerate() {
        let value = rng.random_bool(0.5);
        let should_be_condition = index > 0 && rng.random_bool(0.65);
        if should_be_condition {
            specs.push(LogicalTreeShape::Condition {
                path,
                value,
                index: rng.random_range(0..index),
            });
        } else {
            specs.push(LogicalTreeShape::Scalar { path, value });
        }
    }

    if !specs
        .iter()
        .any(|shape| matches!(shape, LogicalTreeShape::Condition { .. }))
    {
        let dependency = rng.random_range(0..(specs.len() - 1));
        let path = specs[specs.len() - 1].path();
        let value = specs[specs.len() - 1].value();
        let last = specs.len() - 1;
        specs[last] = LogicalTreeShape::Condition {
            path,
            value,
            index: dependency,
        };
    }

    specs
}

fn satisfied_specs(specs: &[LogicalTreeShape]) -> Vec<bool> {
    let mut result = vec![false; specs.len()];
    for index in 0..specs.len() {
        result[index] = match specs[index] {
            LogicalTreeShape::Scalar { .. } => true,
            LogicalTreeShape::Condition {
                index: prerequisite,
                ..
            } => result[prerequisite] && specs[prerequisite].value(),
        };
    }

    result
}

fn build_tree_at(path: u64, scalars: &BTreeMap<u64, bool>) -> LogicalTree {
    if let Some(value) = scalars.get(&path) {
        return LogicalTree::Atom(*value);
    }

    let left_path = path << 1;
    let right_path = left_path | 1;
    let has_left_descendant = scalars
        .keys()
        .any(|candidate| is_ancestor(left_path, *candidate));
    let has_right_descendant = scalars
        .keys()
        .any(|candidate| is_ancestor(right_path, *candidate));

    if has_left_descendant || has_right_descendant {
        LogicalTree::Pair(
            Box::new(if has_left_descendant {
                build_tree_at(left_path, scalars)
            } else {
                LogicalTree::Nil
            }),
            Box::new(if has_right_descendant {
                build_tree_at(right_path, scalars)
            } else {
                LogicalTree::Nil
            }),
        )
    } else {
        LogicalTree::Nil
    }
}

fn build_case<R: Rng>(rng: &mut R) -> LogicalTreeCase {
    let specs = generate_specs(rng);
    let satisfied = satisfied_specs(&specs);
    let scalars: BTreeMap<u64, bool> = specs
        .iter()
        .zip(satisfied.iter())
        .filter_map(|(shape, satisfied)| {
            if *satisfied {
                Some((shape.path(), shape.value()))
            } else {
                None
            }
        })
        .collect();
    let argument = build_tree_at(1, &scalars);
    let selected_index = rng.random_range(0..specs.len());

    LogicalTreeCase {
        specs,
        satisfied,
        selected_index,
        argument,
    }
}

fn expected_result_text(case: &LogicalTreeCase) -> String {
    let selected = if case.satisfied[case.selected_index] {
        format!("({})", bool_atom(case.specs[case.selected_index].value()))
    } else {
        "()".to_string()
    };
    let condition_values: Vec<String> = case
        .specs
        .iter()
        .zip(case.satisfied.iter())
        .filter_map(|(shape, satisfied)| {
            if *satisfied && matches!(shape, LogicalTreeShape::Condition { .. }) {
                Some(bool_atom(shape.value()))
            } else {
                None
            }
        })
        .collect();

    if condition_values.is_empty() {
        format!("({selected})")
    } else {
        format!("({selected} {})", condition_values.join(" "))
    }
}

fn compile_and_run(program: &str, run_args: &str) -> Rc<SExp> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let compiled = perform_compile_of_file(
        &mut allocator,
        runner,
        "cse-logical-tree-fuzz.clsp",
        program,
    )
    .expect("generated logical tree program should compile");

    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*test*"));
    let args = compose_sexp(Srcloc::start("*cse-logical-tree-args*"), run_args);
    simple_run(opts, compiled.compiled, args).expect("generated logical tree program should run")
}

#[test]
fn test_cse_logical_tree_fuzz() {
    for seed in 0..GENERATED_LOGICAL_TREE_PROGRAMS {
        let mut rng = simple_seeded_rng(0xC5E1_0000 | seed);
        let case = build_case(&mut rng);
        let mut renderer = ProgramRenderer::new(case.specs.clone());
        let rendered = renderer.render(&mut rng, &case);
        let result = compile_and_run(&rendered.program, &rendered.run_args);
        let expected = compose_sexp(renderer.srcloc.clone(), &rendered.expected);

        assert_eq!(
            result, expected,
            "logical tree CSE fuzz mismatch for seed {seed}\nspecs: {:?}\nsatisfied: {:?}\nselected_index: {}\nprogram:\n{}\nargs: {}\nexpected: {}\nresult: {}",
            case.specs, case.satisfied, case.selected_index, rendered.program, rendered.run_args, rendered.expected, result
        );
    }
}
