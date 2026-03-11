use num_bigint::ToBigInt;
use rand::prelude::*;
use regex::Regex;

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;

use crate::compiler::clvm::run;
use crate::compiler::compiler::{compile_from_compileform, DefaultCompilerOpts};
use crate::compiler::comptypes::{BodyForm, CompileForm, CompilerOpts, DefunData, HelperForm};
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::frontend::compile_bodyform;
use crate::compiler::optimize::cse::cse_optimize_bodyform;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::sexp::{enlist, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::CompileContextWrapper;

use crate::tests::classic::run::{do_basic_brun, do_basic_run};
use crate::tests::compiler::clvm::TEST_TIMEOUT;
use crate::tests::util::RngLFSR;

#[test]
fn smoke_test_cse_optimization() {
    let filename = "*test*";
    let source = indoc! {"
    (a (i Q
      (com (G (- Q 1) (* (+ 1 Q) R)))
      (com (* (+ 1 Q) R))
      ) 1)"}
    .to_string();
    let srcloc = Srcloc::start(filename);
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(filename));
    let parsed = parse_sexp(srcloc.clone(), source.bytes()).expect("should parse");
    let bodyform = compile_bodyform(opts.clone(), parsed[0].clone()).expect("should compile");
    let cse_transformed =
        cse_optimize_bodyform(&srcloc, b"test", true, &bodyform).expect("should cse optimize");
    let re_def = r"(let ((cse_[$]_[0-9]+ ([*] ([+] 1 Q) R))) (a (i Q (com (G (- Q 1) cse_[$]_[0-9]+)) (com cse_[$]_[0-9]+)) 1))".replace("(", r"\(").replace(")",r"\)");
    let re = Regex::new(&re_def).expect("should become a regex");
    assert!(re.is_match(&cse_transformed.to_sexp().to_string()));
}

#[test]
fn test_cse_tricky() {
    let filename = "resources/tests/strict/cse-complex-1.clsp";
    let program = do_basic_run(&vec!["run".to_string(), filename.to_string()])
        .trim()
        .to_string();

    let run_result_11 = do_basic_brun(&vec![
        "brun".to_string(),
        program.clone(),
        "(11)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(run_result_11, "506");

    let run_result_41 = do_basic_brun(&vec!["brun".to_string(), program, "(41)".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result_41, "15375");
}

#[test]
fn test_cse_tricky_lambda() {
    let filename = "resources/tests/strict/cse-complex-1-lambda.clsp";
    let program = do_basic_run(&vec!["run".to_string(), filename.to_string()])
        .trim()
        .to_string();

    let run_result_11 = do_basic_brun(&vec![
        "brun".to_string(),
        program.clone(),
        "(11)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(run_result_11, "5566");

    let run_result_41 = do_basic_brun(&vec![
        "brun".to_string(),
        program.clone(),
        "(41)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(run_result_41, "0x099e67");

    let run_result_5 = do_basic_brun(&vec!["brun".to_string(), program, "(5)".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result_5, "240");
}

// Ensure that we're sorting CSE rounds to apply by dominance so we do inner
// replacements before outer ones.  Any that aren't dominated don't have an
// order that matters.
#[test]
fn test_cse_dominance_sorting() {
    let filename = "resources/tests/strict/cse-test-no-dom.clsp";
    let program = do_basic_run(&vec!["run".to_string(), filename.to_string()])
        .trim()
        .to_string();
    let run_result = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program,
        "(((3 3) (2 1 13 19) (5 5) (7 7)))".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(run_result, "(13 19)");
}

// Test out atomsort from bram's chialisp
#[test]
fn test_atomsort_bad_ref_simplified() {
    let filename = "resources/tests/strict/csecond.clsp";

    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        filename.to_string(),
    ])
    .trim()
    .to_string();

    let run_result = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program,
        "((99 101 103))".to_string(),
    ])
    .trim()
    .to_string();

    // Expect test5
    assert_eq!(run_result, "\"test5\"");
}

#[test]
fn test_atomsort_bad_ref() {
    let filename = "resources/tests/strict/test_atomsort.clsp";

    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        filename.to_string(),
    ])
    .trim()
    .to_string();

    let run_result_empty = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program.clone(),
        "(())".to_string(),
    ])
    .trim()
    .to_string();

    // Expect a sorted list, descending order.
    assert_eq!(run_result_empty, "()");

    let run_result_one_item = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program.clone(),
        "((0x100001))".to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(run_result_one_item, "(0x100001)");

    let run_result_two_items = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program.clone(),
        "((0x100001 0x100002))".to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(run_result_two_items, "(0x100002 0x100001)");

    let run_result_three_items = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program,
        "((0x100001 0x100003 0x100002))".to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(run_result_three_items, "(0x100003 0x100002 0x100001)");
}

#[test]
fn test_tricky_handcalc_example() {
    let filename = "resources/tests/strict/cse_tricky_assign.clsp";

    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/game-referee-after-cl21".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        filename.to_string(),
    ])
    .trim()
    .to_string();

    eprintln!("{program}");
    assert!(!program.contains(":"));

    let run_result = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program,
        "((13 . 1) (12 . 1) (10 . 1) (6 . 2) (5 . 2))".to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(run_result, "31");
}

// Produce trees of CSE eligible forms and challenge the CSE optimizer.
//
// Things that might trip up CSE:
//
// A cse occurrence is in a lambda
// A cse occurrence is in an assign or let binding
// A cse occurrence is in an assign or let body
// A cse occurrence is in an if condition
// A cse occurrence is in a conditional branch of an if
// A cse occurrence dominates another
// apply is weird

// Generate expressions of the form
//
// Imagining a random tree structure, we'll generate a program that retrieves
// a few particular values in the tree based on the presence or absence of other
// values.
//
// If we evaluate it in the wrong order, it'll fail, because we'll ensure that
// earlier checks gate later checks at runtime.
//
// Imagine
//
//                             50
//                 25                      75
//          12           37         62            87
//      6      18     31    43   56    68      81    93
//
// Each tree node is (v l . r)
//
// So we can write a function like:
//
// (if (select-tree tree "")
//     (if (select-tree tree "l")
//         (if (all (select-tree tree "ll") (select-tree "llr") (select-tree tree "r") (select-tree tree "rl"))
//             (select-tree tree "llrl")
//             (select-tree tree "ll")
//             )
// ...
//
// Choose a few nodes from the tree along with nodes to return when those nodes are
// present.
//
// ((37 . 18) (75 . 56) (87 . 18) (56 . 68))
//
// We choose points along the paths to each node to insert checks
//
// paths:
//
// -- checks
// (50 25 37 31 . 50)
// (50 25 37 . 18)
// (50 75 . 56)
// (50 75 62 56 . 68)
// (50 75 87 . 18)
// -- returns
// (50)
// (50 25 12 18)
// (50 75 62 56)
// (50 75 62 68)
// -- set
// (12 18 25 37 50 56 62 68 75 87)
// -- choose gates (not leaves)
// ((12) 18 25 37 50 56 62 68 (75) 87)
//
// Then organize them by their relationships to the gates
//
// ((12 18) (75 56 62 68 87) 25 37 50)
//
// Find out which rows are gated by which gates:
//
// Must be sorted by dominance
// (50 25 37 31 . 50) -- not gated
// (50 25 37 . 18) -- gated by 12
// (50 75 62 56 . 68) -- gated by 75
// (50 75 87 . 18) -- gated by 12 and 75
// (50 75 . 56) -- gated by 75
//
// First write out a skeleton that mirrors these choices:
// (if
//   (all
//     ;; path to check
//     (select-tree tree "") (select-tree tree "l") (select-tree tree "lr") (select-tree tree "lrl")
//     ;; path to target (if not subsumed)
//     ;; checks here if needed.
//     )
//
//   (select-tree tree "")
//
//   Next paths (gated by 12)
//   (if
//     (all (select-tree tree "") (select-tree tree "l") (select-tree tree "ll"))
//     ;; Expressions gated by 12
//     (if (all (select-tree tree "") (select-tree tree "l") (select-tree tree "lr") (select-tree tree "lllr"))
//       (select-tree tree "lllr")
//       ;; also gated by 75
//       (if (all (select-tree tree "") (select-tree tree "r"))
//         (if (all (select-tree tree "") (select-tree tree "r") (select-tree tree "rl") (select-tree tree "rll") (select-tree tree "rlr"))
//           (select-tree tree "rlr")
//           ;; Others downstream (50 75 . 56) is after gated by 12 and 75
//           (if (all (select-tree tree "") (select-tree tree "r") (select-tree tree "rl") (select-tree tree "rll"))
//             (select-tree tree "rll")
//             ...

struct GenerateTrickyCSE {
    pub numbers: BTreeSet<u16>,
    pub tree: TreeNode,
    pub choices: Vec<CheckAndRetrieve>,
}

#[derive(Default, Clone, Debug)]
struct TreeNode {
    pub number: u16,
    pub left: Option<Rc<TreeNode>>,
    pub right: Option<Rc<TreeNode>>,
}

fn generate_option<F>(srcloc: Srcloc, opt: Option<&Rc<TreeNode>>, filter: &F) -> SExp
where
    F: Fn(&TreeNode) -> bool,
{
    if let Some(r) = opt {
        if filter(r) {
            return r.generate(srcloc, filter);
        }
    }

    SExp::Nil(srcloc)
}

impl TreeNode {
    fn generate<F>(&self, srcloc: Srcloc, filter: &F) -> SExp
    where
        F: Fn(&TreeNode) -> bool,
    {
        let n = SExp::Integer(srcloc.clone(), self.number.to_bigint().unwrap());
        if self.left.is_none() && self.right.is_none() {
            n
        } else {
            let a = generate_option(srcloc.clone(), self.left.as_ref(), filter);
            let b = generate_option(srcloc.clone(), self.right.as_ref(), filter);
            enlist(srcloc, &[Rc::new(a), Rc::new(b), Rc::new(n)])
        }
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Debug)]
enum Direction {
    L,
    R,
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Debug)]
struct TreeChoice {
    pub path: Vec<Direction>,
}

impl TreeChoice {
    pub fn prerequisites(&self) -> BTreeSet<TreeChoice> {
        self.path
            .iter()
            .enumerate()
            .map(|(i, _)| TreeChoice {
                path: self.path.iter().take(i).cloned().collect(),
            })
            .collect()
    }

    pub fn contains(&self, other: &TreeChoice) -> bool {
        self.prerequisites().contains(other)
    }

    pub fn condition(&self) -> Rc<BodyForm> {
        let srcloc = Srcloc::start("*tree-choice-condition*");
        let mut condition = Rc::new(BodyForm::Value(SExp::Atom(srcloc.clone(), b"arg".to_vec())));
        for p in self.path.iter() {
            let op = match p {
                Direction::L => b"f",
                Direction::R => b"r",
            };
            condition = Rc::new(BodyForm::Call(
                srcloc.clone(),
                vec![
                    Rc::new(BodyForm::Value(SExp::Atom(srcloc.clone(), op.to_vec()))),
                    condition,
                ],
                None,
            ));
        }
        condition
    }

    pub fn conditions(&self, myself: bool, prev: Option<Rc<BodyForm>>) -> Vec<Rc<BodyForm>> {
        let mut condition_vec = prev.map(|b| vec![b.clone()]).unwrap_or_else(|| vec![]);
        let mut prerequisite_conditions =
            self.prerequisites().iter().map(|p| p.condition()).collect();
        condition_vec.append(&mut prerequisite_conditions);
        if myself {
            condition_vec.push(self.condition());
        }
        condition_vec
    }
}

fn all_conditions(c: &[Rc<BodyForm>]) -> Rc<BodyForm> {
    if c.is_empty() {
        return Rc::new(BodyForm::Quoted(SExp::Integer(
            Srcloc::start("*all-conditions-empty*"),
            bi_one(),
        )));
    }
    let mut copy_vec = c.to_vec();
    copy_vec.insert(
        0,
        Rc::new(BodyForm::Value(SExp::Atom(c[0].loc(), b"all".to_vec()))),
    );
    Rc::new(BodyForm::Call(c[0].loc(), copy_vec, None))
}

fn if_expr(
    cond: Rc<BodyForm>,
    then_clause: Rc<BodyForm>,
    else_clause: Rc<BodyForm>,
) -> Rc<BodyForm> {
    Rc::new(BodyForm::Call(
        cond.loc(),
        vec![
            Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), b"a".to_vec()))),
            Rc::new(BodyForm::Call(
                cond.loc(),
                vec![
                    Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), b"i".to_vec()))),
                    cond.clone(),
                    Rc::new(BodyForm::Call(
                        cond.loc(),
                        vec![
                            Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), b"com".to_vec()))),
                            then_clause,
                        ],
                        None,
                    )),
                    Rc::new(BodyForm::Call(
                        cond.loc(),
                        vec![
                            Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), b"com".to_vec()))),
                            else_clause,
                        ],
                        None,
                    )),
                ],
                None,
            )),
            Rc::new(BodyForm::Value(SExp::Atom(cond.loc(), b"@".to_vec()))),
        ],
        None,
    ))
}

#[test]
fn test_tree_choice_conditions() {
    let tree = TreeChoice {
        path: vec![Direction::L, Direction::L, Direction::R],
    };
    let conditions_of = all_conditions(&tree.conditions(false, None));
    assert_eq!(
        conditions_of.to_sexp().to_string(),
        "(all arg (f arg) (f (f arg)))"
    );
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Debug)]
struct CheckAndRetrieve {
    pub gate: Option<TreeChoice>,
    pub retrieve: TreeChoice,
    pub checks: Vec<TreeChoice>,
}

impl CheckAndRetrieve {
    // generate an if of the native checks and the check of the retrievable
    // itself for this object, otherwise doing the else clause.
    pub fn generate_body_if(&self, else_clause: Rc<BodyForm>) -> Rc<BodyForm> {
        let check_conditions: Vec<Rc<BodyForm>> = self
            .checks
            .iter()
            .map(|c| all_conditions(&c.conditions(true, None)))
            .collect();
        let retrieve_condition = all_conditions(
            &self
                .retrieve
                .conditions(false, Some(all_conditions(&check_conditions))),
        );
        let what_to_do = self.retrieve.condition();

        if_expr(retrieve_condition, what_to_do, else_clause)
    }
}

#[test]
fn test_check_and_retrieve_1() {
    let car = CheckAndRetrieve {
        gate: None,
        retrieve: TreeChoice {
            path: vec![Direction::L, Direction::R],
        },
        checks: vec![],
    };
    let else_clause = Rc::new(BodyForm::Value(SExp::Nil(Srcloc::start("*test*"))));
    assert_eq!(
        car.generate_body_if(else_clause).to_sexp().to_string(),
        "(a (i (all (q . 1) arg (f arg)) (com (r (f arg))) (com ())) @)"
    );
}

#[test]
fn test_check_and_retrieve_2() {
    let car = CheckAndRetrieve {
        gate: None,
        retrieve: TreeChoice {
            path: vec![Direction::L, Direction::R],
        },
        checks: vec![TreeChoice {
            path: vec![Direction::R],
        }],
    };
    let else_clause = Rc::new(BodyForm::Value(SExp::Nil(Srcloc::start("*test*"))));
    assert_eq!(
        car.generate_body_if(else_clause).to_sexp().to_string(),
        "(a (i (all (all (all arg (r arg))) arg (f arg)) (com (r (f arg))) (com ())) @)"
    );
}

fn create_number_tree(tree: &mut TreeNode, numbers: &[u16]) {
    assert!(!numbers.is_empty());
    let len = numbers.len();
    if len == 1 {
        tree.number = numbers[0];
        return;
    }

    let mid = len / 2;
    tree.number = numbers[mid];
    if mid > 0 {
        let left_slice = &numbers[0..mid];
        let mut left_node = TreeNode::default();
        create_number_tree(&mut left_node, left_slice);
        tree.left = Some(Rc::new(left_node));
    }
    if len > mid + 1 {
        let right_slice = &numbers[mid + 1..len];
        let mut right_node = TreeNode::default();
        create_number_tree(&mut right_node, right_slice);
        tree.right = Some(Rc::new(right_node));
    }
}

fn choose_in_tree(tree: &TreeNode, number: u16) -> TreeChoice {
    let mut path: Vec<Direction> = Vec::new();
    let mut tn: &TreeNode = tree;
    while tn.number != number {
        if number < tn.number {
            path.push(Direction::L);
            if let Some(t) = tn.left.as_ref().borrow() {
                tn = t;
            } else {
                panic!("should have existed");
            }
        } else {
            path.push(Direction::R);
            if let Some(t) = tn.right.as_ref().borrow() {
                tn = t;
            } else {
                panic!("should have existed");
            }
        }
    }
    TreeChoice { path }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Debug)]
struct IfWithGate {
    gate: TreeChoice,
    in_gate: BTreeSet<CheckAndRetrieve>,
    otherwise_reachable: BTreeSet<CheckAndRetrieve>,
}

impl IfWithGate {
    fn generate(&self) -> Rc<BodyForm> {
        let srcloc = Srcloc::start("*if-with-gate*");
        let gate_conditions = all_conditions(&self.gate.conditions(false, None));
        let mut maybe_also_these = Rc::new(BodyForm::Quoted(SExp::Nil(srcloc.clone())));
        for reachable in self.otherwise_reachable.iter().rev() {
            maybe_also_these = reachable.generate_body_if(maybe_also_these);
        }

        let mut gated = maybe_also_these.clone();
        for new_gated in self.in_gate.iter().rev() {
            gated = new_gated.generate_body_if(gated);
        }

        if_expr(gate_conditions, gated, maybe_also_these)
    }
}

#[test]
fn test_if_with_gate_generate_0() {
    let mut in_gate_set = BTreeSet::new();
    in_gate_set.insert(CheckAndRetrieve {
        gate: Some(TreeChoice {
            path: vec![Direction::R, Direction::L, Direction::R, Direction::L],
        }),
        retrieve: TreeChoice {
            path: vec![
                Direction::R,
                Direction::R,
                Direction::R,
                Direction::R,
                Direction::L,
                Direction::L,
            ],
        },
        checks: vec![],
    });
    let mut otherwise_set = BTreeSet::new();
    otherwise_set.insert(CheckAndRetrieve {
        gate: None,
        retrieve: TreeChoice {
            path: vec![
                Direction::R,
                Direction::R,
                Direction::R,
                Direction::R,
                Direction::L,
                Direction::R,
            ],
        },
        checks: vec![],
    });
    let iwg = IfWithGate {
        gate: TreeChoice {
            path: vec![Direction::R, Direction::R, Direction::R],
        },
        in_gate: in_gate_set,
        otherwise_reachable: otherwise_set,
    };
    assert_eq!(iwg.generate().to_sexp().to_string(), "(a (i (all arg (r arg) (r (r arg))) (com (a (i (all (q . 1) arg (r arg) (r (r arg)) (r (r (r arg))) (r (r (r (r arg)))) (f (r (r (r (r arg)))))) (com (f (f (r (r (r (r arg))))))) (com (a (i (all (q . 1) arg (r arg) (r (r arg)) (r (r (r arg))) (r (r (r (r arg)))) (f (r (r (r (r arg)))))) (com (r (f (r (r (r (r arg))))))) (com (q))) @))) @)) (com (a (i (all (q . 1) arg (r arg) (r (r arg)) (r (r (r arg))) (r (r (r (r arg)))) (f (r (r (r (r arg)))))) (com (r (f (r (r (r (r arg))))))) (com (q))) @))) @)");
}

impl GenerateTrickyCSE {
    fn new<R: Rng + ?Sized>(rng: &mut R) -> Self {
        // Generate number tree.
        let number_of_numbers = 5 + (rng.random::<u64>() as usize) % 11;
        let numbers_set: BTreeSet<u16> = (0..number_of_numbers).map(|_| rng.random()).collect();
        let numbers: Vec<u16> = numbers_set.iter().copied().collect();
        let mut number_tree = TreeNode::default();
        create_number_tree(&mut number_tree, &numbers);

        // Now we have a number tree.  Choose some random values and find their
        // paths in the tree.
        let num_choices = 3 + ((rng.random::<u64>() as usize) % 7);
        let mut choices: Vec<CheckAndRetrieve> = (0..num_choices)
            .map(|_| {
                let retrieve = choose_in_tree(
                    &number_tree,
                    numbers[(rng.random::<u64>() as usize) % number_of_numbers],
                );
                let gate = if rng.random() {
                    Some(choose_in_tree(
                        &number_tree,
                        numbers[(rng.random::<u64>() as usize) % number_of_numbers],
                    ))
                } else {
                    None
                };
                let num_checks = (rng.random::<u64>() as usize) % 3;
                let checks: Vec<TreeChoice> = (0..num_checks)
                    .map(|_| {
                        choose_in_tree(
                            &number_tree,
                            numbers[(rng.random::<u64>() as usize) % number_of_numbers],
                        )
                    })
                    .collect();
                CheckAndRetrieve {
                    gate,
                    retrieve,
                    checks,
                }
            })
            .collect();

        // Actually, we can just sort the choices since ones that dominate will
        // come before ones that are dominated (longer path with same prefix).
        choices.sort();

        eprintln!("choices {choices:?}");
        GenerateTrickyCSE {
            choices,
            tree: number_tree.clone(),
            numbers: numbers_set,
        }
    }

    fn generate_expr(&self) -> Rc<BodyForm> {
        let srcloc = Srcloc::start("*tree-generate*");

        // Generate if statements for each tier of gate
        let mut choice_by_gate: BTreeMap<TreeChoice, IfWithGate> = BTreeMap::new();

        // Make a list of distinct gates and the directly gated choices.
        for choice in self.choices.iter() {
            if let Some(choice_gate) = choice.gate.as_ref() {
                if let Some(gated_by_choice) = choice_by_gate.get_mut(choice_gate) {
                    gated_by_choice.in_gate.insert(choice.clone());
                } else {
                    let mut in_gate = BTreeSet::new();
                    in_gate.insert(choice.clone());
                    choice_by_gate.insert(
                        choice_gate.clone(),
                        IfWithGate {
                            gate: choice_gate.clone(),
                            in_gate,
                            otherwise_reachable: BTreeSet::default(),
                        },
                    );
                }
            }
        }

        // For each gate, filter all the remaining checks into in_gate or
        // otherwise_reachable.
        for (gate, ifwith) in choice_by_gate.iter_mut() {
            let (in_gate, otherwise_reachable) = self
                .choices
                .iter()
                .cloned()
                .partition::<Vec<CheckAndRetrieve>, _>(|choice| {
                    choice
                        .gate
                        .as_ref()
                        .map(|g| gate.contains(g))
                        .unwrap_or(false)
                });
            for g in in_gate.iter() {
                ifwith.in_gate.insert(g.clone());
            }
            for g in otherwise_reachable.iter() {
                ifwith.otherwise_reachable.insert(g.clone());
            }
        }

        // Generate the full expression.
        let mut final_value = Rc::new(BodyForm::Quoted(SExp::Nil(srcloc.clone())));
        for (gate, ifwith) in choice_by_gate.iter().rev() {
            let this_gate = ifwith.generate();
            let gate_conditions = all_conditions(&gate.conditions(true, None));
            final_value = if_expr(gate_conditions, this_gate, final_value);
        }

        final_value
    }

    fn generate_helper(&self) -> HelperForm {
        let srcloc = Srcloc::start("*helper*");
        let args = Rc::new(SExp::Cons(
            srcloc.clone(),
            Rc::new(SExp::Atom(srcloc.clone(), b"arg".to_vec())),
            Rc::new(SExp::Nil(srcloc.clone())),
        ));
        HelperForm::Defun(
            false,
            Box::new(DefunData {
                nl: srcloc.clone(),
                loc: srcloc.clone(),
                name: b"tricky-cse".to_vec(),
                args: args.clone(),
                orig_args: args.clone(),
                body: self.generate_expr(),
                kw: None,
                synthetic: None,
            }),
        )
    }

    fn generate_program(&self) -> CompileForm {
        let helper = self.generate_helper();
        let args = Rc::new(SExp::Atom(helper.loc(), b"arg".to_vec()));
        CompileForm {
            loc: helper.loc(),
            include_forms: vec![],
            args,
            exp: Rc::new(BodyForm::Call(
                helper.loc(),
                vec![
                    Rc::new(BodyForm::Value(SExp::Atom(
                        helper.loc(),
                        helper.name().to_vec(),
                    ))),
                    Rc::new(BodyForm::Value(SExp::Atom(helper.loc(), b"arg".to_vec()))),
                ],
                None,
            )),
            helpers: vec![helper],
        }
    }
}

fn test_generated_cse(n: u32) {
    eprintln!("=== SEED {n} ===");
    let mut rng = RngLFSR::new(n);
    let tcse = GenerateTrickyCSE::new(&mut rng);
    let generated = tcse.generate_program();
    eprintln!("{}", generated.to_sexp());
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*tricky-cse*"));
    let opts21 = opts.set_dialect(AcceptedDialect {
        stepping: Some(21),
        strict: true,
        int_fix: false,
    });
    let opts23 = opts
        .set_dialect(AcceptedDialect {
            stepping: Some(23),
            strict: true,
            int_fix: false,
        })
        .set_optimize(true);
    let mut allocator = Allocator::new();
    let mut symbols = HashMap::new();
    let compiled21;
    let compiled23;

    // Get strict cl21 compile
    {
        let mut wrapper21 = CompileContextWrapper::new(
            &mut allocator,
            runner.clone(),
            &mut symbols,
            get_optimizer(&generated.loc(), opts21.clone()).expect("should be ok dialect"),
        );
        eprintln!("21 compile");
        compiled21 = Rc::new(
            compile_from_compileform(&mut wrapper21.context, opts21, generated.clone())
                .expect("compiled"),
        )
    }

    // Get cl23 compile
    {
        let mut wrapper23 = CompileContextWrapper::new(
            &mut allocator,
            runner.clone(),
            &mut symbols,
            get_optimizer(&generated.loc(), opts23.clone()).expect("should be ok dialect"),
        );
        eprintln!("23 compile");
        compiled23 = Rc::new(
            compile_from_compileform(&mut wrapper23.context, opts23, generated.clone())
                .expect("compiled"),
        )
    }

    eprintln!("generate tree numbers");
    let check_numbers: BTreeSet<u16> = tcse
        .numbers
        .iter()
        .filter(|_| {
            let check: u8 = rng.random();
            check & 15 >= 1
        })
        .copied()
        .collect();
    eprintln!("generate tree");
    let tree = Rc::new(tcse.tree.generate(generated.loc(), &move |tn| {
        check_numbers.contains(&tn.number)
    }));

    eprintln!("compiled21 {compiled21}");
    eprintln!("compiled23 {compiled23}");
    eprintln!("tree       {tree}");

    let mut allocator = Allocator::new();
    let clvm21 = run(
        &mut allocator,
        runner.clone(),
        opts.prim_map(),
        compiled21.clone(),
        tree.clone(),
        None,
        Some(TEST_TIMEOUT),
    );
    let clvm23 = run(
        &mut allocator,
        runner,
        opts.prim_map(),
        compiled23.clone(),
        tree.clone(),
        None,
        Some(TEST_TIMEOUT),
    );
    if let Ok(res21) = clvm21.as_ref() {
        eprintln!("cl21 {res21}");
    }
    if let Ok(res23) = clvm23.as_ref() {
        eprintln!("cl23 {res23}");
    }
    if clvm21.is_err() || clvm23.is_err() {
        // Precise errors might change due to differences in ordering and
        // locations and such.
        assert_eq!(clvm21.is_err(), clvm23.is_err());
        return;
    }
    assert_eq!(clvm21, clvm23);
}

#[test]
fn test_generate_tricky_cse() {
    for i in 0..16 {
        test_generated_cse(10000 + i);
    }
}
