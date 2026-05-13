use rand::Rng;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::run;
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;

use crate::tests::compiler::clvm::TEST_TIMEOUT;
use crate::tests::compiler::fuzz::{compose_sexp, simple_seeded_rng};

const GENERATED_CSE_LIFT_CASES: u32 = 32;
const MAX_STACK_ENTRIES: usize = 18;
const MAX_EXPR_COST: usize = 90;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Shape {
    Int,
    Atom,
    Pair(Box<Shape>, Box<Shape>),
}

impl Shape {
    fn is_atom_like(&self) -> bool {
        matches!(self, Shape::Int | Shape::Atom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum AbstractExpr {
    Var(&'static str),
    Int(i64),
    F(Box<AbstractExpr>),
    R(Box<AbstractExpr>),
    C(Box<AbstractExpr>, Box<AbstractExpr>),
    Sha256(Box<AbstractExpr>),
    Plus(Box<AbstractExpr>, Box<AbstractExpr>),
    Times(Box<AbstractExpr>, Box<AbstractExpr>),
    If(Box<AbstractExpr>, Box<AbstractExpr>, Box<AbstractExpr>),
    Equals(Box<AbstractExpr>, Box<AbstractExpr>),
    Strlen(Box<AbstractExpr>),
    L(Box<AbstractExpr>),
}

impl AbstractExpr {
    fn render(&self) -> String {
        match self {
            AbstractExpr::Var(name) => name.to_string(),
            AbstractExpr::Int(value) => value.to_string(),
            AbstractExpr::F(value) => format!("(f {})", value.render()),
            AbstractExpr::R(value) => format!("(r {})", value.render()),
            AbstractExpr::C(left, right) => format!("(c {} {})", left.render(), right.render()),
            AbstractExpr::Sha256(value) => format!("(sha256 1 {})", value.render()),
            AbstractExpr::Plus(left, right) => {
                format!("(+ {} {})", left.render(), right.render())
            }
            AbstractExpr::Times(left, right) => {
                format!("(* {} {})", left.render(), right.render())
            }
            AbstractExpr::If(condition, then_body, else_body) => format!(
                "(if {} {} {})",
                condition.render(),
                then_body.render(),
                else_body.render()
            ),
            AbstractExpr::Equals(left, right) => {
                format!("(= {} {})", left.render(), right.render())
            }
            AbstractExpr::Strlen(value) => format!("(strlen {})", value.render()),
            AbstractExpr::L(value) => format!("(l {})", value.render()),
        }
    }

    fn cost(&self) -> usize {
        match self {
            AbstractExpr::Var(_) | AbstractExpr::Int(_) => 1,
            AbstractExpr::F(value)
            | AbstractExpr::R(value)
            | AbstractExpr::Sha256(value)
            | AbstractExpr::Strlen(value)
            | AbstractExpr::L(value) => 1 + value.cost(),
            AbstractExpr::C(left, right)
            | AbstractExpr::Plus(left, right)
            | AbstractExpr::Times(left, right)
            | AbstractExpr::Equals(left, right) => 1 + left.cost() + right.cost(),
            AbstractExpr::If(condition, then_body, else_body) => {
                1 + condition.cost() + then_body.cost() + else_body.cost()
            }
        }
    }
}

#[derive(Clone, Debug)]
struct StackExpr {
    expr: AbstractExpr,
    shape: Shape,
    requirements: BTreeSet<AbstractExpr>,
}

impl StackExpr {
    fn new(expr: AbstractExpr, shape: Shape, requirements: BTreeSet<AbstractExpr>) -> Option<Self> {
        if expr.cost() <= MAX_EXPR_COST {
            Some(StackExpr {
                expr,
                shape,
                requirements,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct GeneratedCseLiftCase {
    program: String,
    simple_expr: String,
}

fn pair_shape(left: Shape, right: Shape) -> Shape {
    Shape::Pair(Box::new(left), Box::new(right))
}

fn initial_stack() -> Vec<StackExpr> {
    let empty = BTreeSet::new();
    let tree_shape = pair_shape(
        pair_shape(Shape::Int, Shape::Int),
        pair_shape(Shape::Int, Shape::Int),
    );

    vec![
        StackExpr {
            expr: AbstractExpr::Var("TREE"),
            shape: tree_shape,
            requirements: empty.clone(),
        },
        StackExpr {
            expr: AbstractExpr::Var("ATOM"),
            shape: Shape::Int,
            requirements: empty.clone(),
        },
        StackExpr {
            expr: AbstractExpr::Int(1),
            shape: Shape::Int,
            requirements: empty.clone(),
        },
        StackExpr {
            expr: AbstractExpr::Int(2),
            shape: Shape::Int,
            requirements: empty,
        },
    ]
}

fn random_index<R, F>(rng: &mut R, stack: &[StackExpr], predicate: F) -> Option<usize>
where
    R: Rng,
    F: Fn(&StackExpr) -> bool,
{
    let candidates: Vec<usize> = stack
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| predicate(entry).then_some(idx))
        .collect();
    if candidates.is_empty() {
        None
    } else {
        Some(candidates[(rng.random::<u64>() as usize) % candidates.len()])
    }
}

fn merge_requirements(entries: &[&StackExpr]) -> BTreeSet<AbstractExpr> {
    entries
        .iter()
        .flat_map(|entry| entry.requirements.iter().cloned())
        .collect()
}

fn random_generated_expr<R: Rng>(rng: &mut R, stack: &[StackExpr]) -> Option<StackExpr> {
    match rng.random::<u8>() % 11 {
        0 => {
            let idx = random_index(rng, stack, |entry| matches!(entry.shape, Shape::Pair(_, _)))?;
            let StackExpr {
                expr,
                shape,
                requirements,
            } = &stack[idx];
            let Shape::Pair(left, _) = shape else {
                return None;
            };
            let mut next_requirements = requirements.clone();
            next_requirements.insert(AbstractExpr::L(Box::new(expr.clone())));
            StackExpr::new(
                AbstractExpr::F(Box::new(expr.clone())),
                left.as_ref().clone(),
                next_requirements,
            )
        }
        1 => {
            let idx = random_index(rng, stack, |entry| matches!(entry.shape, Shape::Pair(_, _)))?;
            let StackExpr {
                expr,
                shape,
                requirements,
            } = &stack[idx];
            let Shape::Pair(_, right) = shape else {
                return None;
            };
            let mut next_requirements = requirements.clone();
            next_requirements.insert(AbstractExpr::L(Box::new(expr.clone())));
            StackExpr::new(
                AbstractExpr::R(Box::new(expr.clone())),
                right.as_ref().clone(),
                next_requirements,
            )
        }
        2 => {
            let left_idx = (rng.random::<u64>() as usize) % stack.len();
            let right_idx = (rng.random::<u64>() as usize) % stack.len();
            let left = &stack[left_idx];
            let right = &stack[right_idx];
            StackExpr::new(
                AbstractExpr::C(Box::new(left.expr.clone()), Box::new(right.expr.clone())),
                pair_shape(left.shape.clone(), right.shape.clone()),
                merge_requirements(&[left, right]),
            )
        }
        3 => {
            let idx = random_index(rng, stack, |entry| entry.shape.is_atom_like())?;
            let entry = &stack[idx];
            StackExpr::new(
                AbstractExpr::Sha256(Box::new(entry.expr.clone())),
                Shape::Atom,
                entry.requirements.clone(),
            )
        }
        4 => {
            let idx = random_index(rng, stack, |entry| entry.shape.is_atom_like())?;
            let entry = &stack[idx];
            StackExpr::new(
                AbstractExpr::Strlen(Box::new(entry.expr.clone())),
                Shape::Int,
                entry.requirements.clone(),
            )
        }
        5 | 6 => {
            let left_idx = random_index(rng, stack, |entry| matches!(entry.shape, Shape::Int))?;
            let right_idx = random_index(rng, stack, |entry| matches!(entry.shape, Shape::Int))?;
            let left = &stack[left_idx];
            let right = &stack[right_idx];
            let expr = if rng.random() {
                AbstractExpr::Plus(Box::new(left.expr.clone()), Box::new(right.expr.clone()))
            } else {
                AbstractExpr::Times(Box::new(left.expr.clone()), Box::new(right.expr.clone()))
            };
            StackExpr::new(expr, Shape::Int, merge_requirements(&[left, right]))
        }
        7 => {
            let left_idx = random_index(rng, stack, |entry| entry.shape.is_atom_like())?;
            let right_idx = random_index(rng, stack, |entry| entry.shape.is_atom_like())?;
            let left = &stack[left_idx];
            let right = &stack[right_idx];
            StackExpr::new(
                AbstractExpr::Equals(Box::new(left.expr.clone()), Box::new(right.expr.clone())),
                Shape::Int,
                merge_requirements(&[left, right]),
            )
        }
        8 => {
            let idx = (rng.random::<u64>() as usize) % stack.len();
            let entry = &stack[idx];
            StackExpr::new(
                AbstractExpr::L(Box::new(entry.expr.clone())),
                Shape::Int,
                entry.requirements.clone(),
            )
        }
        9 => {
            let condition_idx =
                random_index(rng, stack, |entry| matches!(entry.shape, Shape::Int))?;
            let then_idx = (rng.random::<u64>() as usize) % stack.len();
            let then_entry = &stack[then_idx];
            let else_idx = random_index(rng, stack, |entry| entry.shape == then_entry.shape)?;
            let condition = &stack[condition_idx];
            let else_entry = &stack[else_idx];
            StackExpr::new(
                AbstractExpr::If(
                    Box::new(condition.expr.clone()),
                    Box::new(then_entry.expr.clone()),
                    Box::new(else_entry.expr.clone()),
                ),
                then_entry.shape.clone(),
                merge_requirements(&[condition, then_entry, else_entry]),
            )
        }
        _ => {
            let value = 3 + (rng.random::<u8>() % 19) as i64;
            StackExpr::new(AbstractExpr::Int(value), Shape::Int, BTreeSet::new())
        }
    }
}

fn generate_abstract_stack<R: Rng>(rng: &mut R) -> StackExpr {
    let mut stack = initial_stack();
    let rounds = 12 + (rng.random::<u8>() as usize % 14);

    for _ in 0..rounds {
        if let Some(next) = random_generated_expr(rng, &stack) {
            stack.push(next);
        }
        if stack.len() > MAX_STACK_ENTRIES {
            let remove_at = 4 + (rng.random::<u64>() as usize) % (stack.len() - 4);
            stack.remove(remove_at);
        }
    }

    let unsafe_choices: Vec<StackExpr> = stack
        .iter()
        .filter(|entry| !entry.requirements.is_empty())
        .cloned()
        .collect();

    if unsafe_choices.is_empty() {
        let tree = &initial_stack()[0];
        let mut requirements = BTreeSet::new();
        requirements.insert(AbstractExpr::L(Box::new(tree.expr.clone())));
        StackExpr {
            expr: AbstractExpr::F(Box::new(tree.expr.clone())),
            shape: pair_shape(Shape::Int, Shape::Int),
            requirements,
        }
    } else {
        unsafe_choices[(rng.random::<u64>() as usize) % unsafe_choices.len()].clone()
    }
}

fn fallback_for(shape: &Shape) -> &'static str {
    match shape {
        Shape::Int | Shape::Atom => "0",
        Shape::Pair(_, _) => "(c 0 0)",
    }
}

fn safe_binding_expr(depth: usize) -> &'static str {
    match depth % 3 {
        0 => "ATOM",
        1 => "(+ ATOM 1)",
        _ => "(* ATOM 2)",
    }
}

fn render_complex_expr<R: Rng>(rng: &mut R, target: &StackExpr) -> String {
    let target_src = target.expr.render();
    let fallback = fallback_for(&target.shape);
    let mut body = target_src.clone();
    let wrapper_depth = 1 + (rng.random::<u8>() as usize % 6);

    for depth in 0..wrapper_depth {
        let var = format!("v{depth}");
        body = match rng.random::<u8>() % 5 {
            0 => format!("(if (= ATOM ATOM) {body} {target_src})"),
            1 => format!("(if (l (c {target_src} ())) {body} {target_src})"),
            2 => format!("(let (({var} {target_src})) {body})"),
            3 => format!("(let* (({var} {target_src}) ({var}_copy {var})) {body})"),
            _ => format!("(assign {var} {} {body})", safe_binding_expr(depth)),
        };
    }

    // Always include one final condition occurrence of the target, even if the
    // random wrapper stack happened to choose only binding forms.
    let mut guarded = format!("(if (l (c {target_src} ())) {body} {target_src})");
    let mut requirements: Vec<AbstractExpr> = target.requirements.iter().cloned().collect();
    requirements.sort_by_key(|expr| (expr.cost(), expr.render()));

    for requirement in requirements.into_iter().rev() {
        guarded = format!("(if {} {guarded} {fallback})", requirement.render());
    }

    guarded
}

fn generate_cse_lift_case<R: Rng>(rng: &mut R) -> GeneratedCseLiftCase {
    let target = generate_abstract_stack(rng);
    let simple_expr = target.expr.render();
    let complex_expr = render_complex_expr(rng, &target);
    let program = format!(
        "(mod (TREE ATOM)
  (include *standard-cl-23*)
  {complex_expr}
)"
    );

    GeneratedCseLiftCase {
        program,
        simple_expr,
    }
}

fn cl23_opts(filename: &str, optimize: bool) -> Rc<dyn CompilerOpts> {
    Rc::new(DefaultCompilerOpts::new(filename))
        .set_dialect(AcceptedDialect {
            stepping: Some(23),
            strict: true,
            int_fix: false,
            extra_numeric_constants: false,
        })
        .set_optimize(optimize)
        .set_frontend_opt(false)
}

fn compile_generated(program: &str, optimize: bool) -> Rc<SExp> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let mut symbols = HashMap::new();
    compile_file(
        &mut allocator,
        runner,
        cl23_opts("*generated-cse-lift-fuzz*", optimize),
        program,
        &mut symbols,
    )
    .unwrap_or_else(|err| {
        panic!("generated CSE lift program should compile: {err:?}\nprogram:\n{program}")
    })
    .to_sexp()
    .into()
}

fn run_compiled(compiled: Rc<SExp>, args: &str) -> Result<Rc<SExp>, String> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let run_args = compose_sexp(Srcloc::start("*generated-cse-lift-args*"), args);
    run(
        &mut allocator,
        runner,
        cl23_opts("*generated-cse-lift-run*", false).prim_map(),
        compiled,
        run_args,
        None,
        Some(TEST_TIMEOUT),
    )
    .map_err(|err| format!("{err:?}"))
}

fn check_generated_case(seed: u32) {
    let mut rng = simple_seeded_rng(0xC5E1_0000 | seed);
    let generated = generate_cse_lift_case(&mut rng);
    eprintln!(
        "generated CSE lift seed {seed}, simple expr: {}",
        generated.simple_expr
    );
    eprintln!("program:\n{}", generated.program);

    let unoptimized = compile_generated(&generated.program, false);
    let optimized = compile_generated(&generated.program, true);

    for args in ["(((3 . 5) . (7 . 11)) 13)", "(0 13)"] {
        let expected = run_compiled(unoptimized.clone(), args).unwrap_or_else(|err| {
            panic!(
                "unoptimized generated CSE lift program should run for seed {seed} args {args}: {err}\nprogram:\n{}",
                generated.program
            )
        });
        let actual = run_compiled(optimized.clone(), args).unwrap_or_else(|err| {
            panic!(
                "optimized generated CSE lift program should run for seed {seed} args {args}: {err}\nprogram:\n{}",
                generated.program
            )
        });
        assert_eq!(
            actual, expected,
            "optimized result differed for generated CSE lift seed {seed} args {args}\nprogram:\n{}",
            generated.program
        );
    }
}

#[test]
fn test_generated_cse_does_not_lift_past_guarding_if() {
    for seed in 0..GENERATED_CSE_LIFT_CASES {
        check_generated_case(seed);
    }
}
