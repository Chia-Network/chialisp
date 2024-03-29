use sha2::{Digest, Sha256};
use std::rc::Rc;

use clvmr::allocator::Allocator;

use num_bigint::ToBigInt;

use crate::classic::clvm::__type_compatibility__::{bi_one, Bytes, BytesFromType, Stream};
use crate::classic::clvm::serialize::{sexp_from_stream, SimpleCreateCLVMObject};
use crate::classic::clvm_tools::binutils::disassemble;
use crate::compiler::clvm::sha256tree;
use crate::compiler::prims::{primapply, primcons, primquote};
use crate::compiler::sexp::{enlist, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::tests::classic::run::{do_basic_brun, do_basic_run};
use crate::util::u8_from_number;

// Tests of a strict compile of singleton_top_layer.clsp
//
// Includes ported code from singleton_top_layer.py and
// my test-singleton.py
// const SINGLETON_MOD_HASH: &[u8] = &[
//     0x24, 0xe0, 0x44, 0x10, 0x1e, 0x57, 0xb3, 0xd8, 0xc9, 0x08, 0xb8, 0xa3, 0x8a, 0xd5, 0x78, 0x48, 0xaf, 0xd2, 0x9d, 0x3e, 0xec, 0xc4, 0x39, 0xdb, 0xa4, 0x5f, 0x44, 0x12, 0xdf, 0x49, 0x54, 0xfd
// ];
const SINGLETON_LAUNCHER_BYTES: &[u8] = &[
    0xff, 0x02, 0xff, 0xff, 0x01, 0xff, 0x04, 0xff, 0xff, 0x04, 0xff, 0x04, 0xff, 0xff, 0x04, 0xff,
    0x05, 0xff, 0xff, 0x04, 0xff, 0x0b, 0xff, 0x80, 0x80, 0x80, 0x80, 0xff, 0xff, 0x04, 0xff, 0xff,
    0x04, 0xff, 0x0a, 0xff, 0xff, 0x04, 0xff, 0xff, 0x02, 0xff, 0x0e, 0xff, 0xff, 0x04, 0xff, 0x02,
    0xff, 0xff, 0x04, 0xff, 0xff, 0x04, 0xff, 0x05, 0xff, 0xff, 0x04, 0xff, 0x0b, 0xff, 0xff, 0x04,
    0xff, 0x17, 0xff, 0x80, 0x80, 0x80, 0x80, 0xff, 0x80, 0x80, 0x80, 0x80, 0xff, 0x80, 0x80, 0x80,
    0xff, 0x80, 0x80, 0x80, 0xff, 0xff, 0x04, 0xff, 0xff, 0x01, 0xff, 0x33, 0xff, 0x3c, 0xff, 0x02,
    0xff, 0xff, 0x03, 0xff, 0xff, 0x07, 0xff, 0x05, 0x80, 0xff, 0xff, 0x01, 0xff, 0x0b, 0xff, 0xff,
    0x01, 0x02, 0xff, 0xff, 0x02, 0xff, 0x0e, 0xff, 0xff, 0x04, 0xff, 0x02, 0xff, 0xff, 0x04, 0xff,
    0x09, 0xff, 0x80, 0x80, 0x80, 0x80, 0xff, 0xff, 0x02, 0xff, 0x0e, 0xff, 0xff, 0x04, 0xff, 0x02,
    0xff, 0xff, 0x04, 0xff, 0x0d, 0xff, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0xff, 0x01, 0xff, 0x0b,
    0xff, 0xff, 0x01, 0x01, 0xff, 0x05, 0x80, 0x80, 0xff, 0x01, 0x80, 0xff, 0x01, 0x80, 0x80,
];

lazy_static! {
    pub static ref SINGLETON_LAUNCHER: String = {
        let mut stream = Stream::new(Some(Bytes::new(Some(BytesFromType::Raw(
            SINGLETON_LAUNCHER_BYTES.to_vec(),
        )))));
        let mut allocator = Allocator::new();
        let code = sexp_from_stream(
            &mut allocator,
            &mut stream,
            Box::new(SimpleCreateCLVMObject {}),
        )
        .expect("should be able to parse binary");
        disassemble(&mut allocator, code.1, Some(0))
    };
}

const SINGLETON_LAUNCHER_HASH: &[u8] = &[
    0xef, 0xf0, 0x75, 0x22, 0x49, 0x50, 0x60, 0xc0, 0x66, 0xf6, 0x6f, 0x32, 0xac, 0xc2, 0xa7, 0x7e,
    0x3a, 0x3e, 0x73, 0x7a, 0xca, 0x8b, 0xae, 0xa4, 0xd1, 0xa6, 0x4e, 0xa4, 0xcd, 0xc1, 0x3d, 0xa9,
];
const FIRST_COIN_PARENT: &[u8] = &[
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
];
const CREATE_COIN: usize = 51;

#[derive(Clone, Debug)]
struct Coin {
    parent_coin_id: Vec<u8>,
    puzzle_hash: Vec<u8>,
    amount: u64,
}

fn std_hash(form: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(&form);
    hasher.finalize().to_vec()
}

// Ported code from singleton_top_layer.py
fn generate_launcher_coin(parent_hash: &[u8], amount: u64) -> Coin {
    Coin {
        parent_coin_id: parent_hash.to_vec(),
        puzzle_hash: SINGLETON_LAUNCHER_HASH.to_vec(),
        amount,
    }
}

impl Coin {
    fn name(&self) -> Vec<u8> {
        let mut form = self.parent_coin_id.clone();
        form.append(&mut self.puzzle_hash.clone());
        let mut size_atom_bytes = u8_from_number(self.amount.to_bigint().unwrap());
        form.append(&mut size_atom_bytes);
        std_hash(&form)
    }
}

fn curry(the_mod: Rc<SExp>, new_args: &[Rc<SExp>]) -> Rc<SExp> {
    let mut fixed_args = SExp::Integer(the_mod.loc(), bi_one());
    for arg in new_args.iter().rev() {
        fixed_args = primcons(
            the_mod.loc(),
            Rc::new(primquote(the_mod.loc(), arg.clone())),
            Rc::new(fixed_args),
        );
    }
    Rc::new(primapply(
        the_mod.loc(),
        Rc::new(primquote(the_mod.loc(), the_mod)),
        Rc::new(fixed_args),
    ))
}

struct LineageProof {
    parent_name: Vec<u8>,
    inner_puzzle_hash: Option<Vec<u8>>,
    amount: u64,
}

struct SingletonLaunch {
    inner_puzzle: Rc<SExp>,
    inner_puzzle_hash: Vec<u8>,
    inner_puzzle_hash_atom: Rc<SExp>,
    curried_singleton: Rc<SExp>,
    curried_singleton_hash_atom: Rc<SExp>,
    amount: u64,
    amount_atom: Rc<SExp>,
}

impl SingletonLaunch {
    fn new(launcher_coin: Coin, program_file: &str, inner_puzzle: Rc<SExp>, amount: u64) -> Self {
        let compiled = do_basic_run(&vec![
            "run".to_string(),
            "-i".to_string(),
            "resources/tests/strict/includes".to_string(),
            "-i".to_string(),
            "resources/tests/strict".to_string(), // defmac_assert.clib
            program_file.to_string(),
        ]);
        let amount_atom = Rc::new(SExp::Integer(inner_puzzle.loc(), bi_one()));
        let parsed_compiled =
            parse_sexp(inner_puzzle.loc(), compiled.bytes()).expect("should parse")[0].clone();
        let singleton_mod_hash = sha256tree(parsed_compiled.clone());
        let curried_singleton = curry(
            parsed_compiled.clone(),
            &[
                Rc::new(SExp::Cons(
                    inner_puzzle.loc(),
                    Rc::new(SExp::Atom(inner_puzzle.loc(), singleton_mod_hash.clone())),
                    Rc::new(SExp::Cons(
                        inner_puzzle.loc(),
                        Rc::new(SExp::Atom(inner_puzzle.loc(), launcher_coin.name())),
                        Rc::new(SExp::Atom(
                            inner_puzzle.loc(),
                            SINGLETON_LAUNCHER_HASH.to_vec(),
                        )),
                    )),
                )),
                inner_puzzle.clone(),
            ],
        );
        let inner_puzzle_hash = sha256tree(inner_puzzle.clone());
        let inner_puzzle_hash_atom =
            Rc::new(SExp::Atom(inner_puzzle.loc(), inner_puzzle_hash.clone()));
        let curried_singleton_hash = sha256tree(curried_singleton.clone());
        SingletonLaunch {
            inner_puzzle_hash,
            inner_puzzle_hash_atom,
            curried_singleton_hash_atom: Rc::new(SExp::Atom(
                inner_puzzle.loc(),
                curried_singleton_hash.clone(),
            )),
            curried_singleton,
            amount,
            amount_atom,
            inner_puzzle,
        }
    }

    fn loc(&self) -> Srcloc {
        self.inner_puzzle.loc()
    }

    fn launcher_solution(&self) -> Rc<SExp> {
        Rc::new(enlist(
            self.loc(),
            &[
                self.curried_singleton_hash_atom.clone(),
                self.amount_atom.clone(),
                Rc::new(SExp::Nil(self.loc())),
            ],
        ))
    }

    fn lineage_proof(&self, parent_coin: &Coin) -> LineageProof {
        let inner_puzzle_hash = if parent_coin.puzzle_hash == SINGLETON_LAUNCHER_HASH {
            None
        } else {
            Some(self.inner_puzzle_hash.clone())
        };
        LineageProof {
            parent_name: parent_coin.parent_coin_id.clone(),
            inner_puzzle_hash,
            amount: parent_coin.amount,
        }
    }

    fn solution_for_singleton(
        &self,
        lineage_proof: &LineageProof,
        inner_solution: Rc<SExp>,
    ) -> Rc<SExp> {
        let parent_name = Rc::new(SExp::Atom(self.loc(), lineage_proof.parent_name.clone()));
        let amount_atom = Rc::new(SExp::Integer(
            self.loc(),
            lineage_proof.amount.to_bigint().unwrap(),
        ));

        let parent_info = if let Some(_inner_puzzle_hash) = &lineage_proof.inner_puzzle_hash {
            enlist(
                self.loc(),
                &[
                    parent_name,
                    self.inner_puzzle_hash_atom.clone(),
                    amount_atom.clone(),
                ],
            )
        } else {
            enlist(self.loc(), &[parent_name, amount_atom.clone()])
        };
        Rc::new(enlist(
            self.loc(),
            &[Rc::new(parent_info), amount_atom, inner_solution],
        ))
    }

    fn get_next_coin(&self, parent: &Coin) -> Coin {
        Coin {
            parent_coin_id: parent.name(),
            puzzle_hash: sha256tree(self.curried_singleton.clone()),
            amount: self.amount,
        }
    }
}

// singleton_top_layer in cl23 (just replacing the assert macro).
#[test]
fn test_singleton_top_layer_23() {
    let program_file = "resources/tests/strict/singleton_top_layer.clsp";
    let loc = Srcloc::start(program_file);
    let inner_puzzle = Rc::new(SExp::Integer(loc.clone(), 5_u32.to_bigint().unwrap()));
    let launcher_coin = generate_launcher_coin(FIRST_COIN_PARENT, 1);
    let launch = SingletonLaunch::new(launcher_coin.clone(), program_file, inner_puzzle, 1);

    let launcher_solution = launch.launcher_solution();
    let bootstrap_singleton = do_basic_brun(&vec![
        "brun".to_string(),
        "--operators-version".to_string(),
        "0".to_string(),
        SINGLETON_LAUNCHER.to_string(),
        launcher_solution.to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(bootstrap_singleton, "((51 0xe435c0e8158464c825adb748900108adba4877bcd960c485f3effd3b8438f7e9 1) (60 0x8117fb4c6939a70936ab1d4f6ffe1daf24333978f0302c0464cbdd98fc346f65))");

    // (((51 0xZZZZ... 1)))
    let inner_solution = Rc::new(enlist(
        launch.loc(),
        &[Rc::new(enlist(
            launch.loc(),
            &[Rc::new(enlist(
                launch.loc(),
                &[
                    Rc::new(SExp::Integer(
                        launch.loc(),
                        CREATE_COIN.to_bigint().unwrap(),
                    )),
                    launch.inner_puzzle_hash_atom.clone(),
                    launch.amount_atom.clone(),
                ],
            ))],
        ))],
    ));
    let lineage_proof = launch.lineage_proof(&launcher_coin);
    let solution_for_singleton =
        launch.solution_for_singleton(&lineage_proof, inner_solution.clone());
    let singleton_eve_run = do_basic_brun(&vec![
        "brun".to_string(),
        "--operators-version".to_string(),
        "0".to_string(),
        launch.curried_singleton.to_string(),
        solution_for_singleton.to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(singleton_eve_run, "((70 0xf5ab253587c5eee301cfe877321e662612f85f993a31616df45ef0f80a686fe8) (51 0xe435c0e8158464c825adb748900108adba4877bcd960c485f3effd3b8438f7e9 1))");
    let parent_of_eve_spend_coin = launch.get_next_coin(&launcher_coin);
    let next_lineage_proof = launch.lineage_proof(&parent_of_eve_spend_coin);
    let solution_for_next_singleton =
        launch.solution_for_singleton(&next_lineage_proof, inner_solution.clone());
    let singleton_next_run = do_basic_brun(&vec![
        "brun".to_string(),
        "--operators-version".to_string(),
        "0".to_string(),
        launch.curried_singleton.to_string(),
        solution_for_next_singleton.to_string(),
    ])
    .trim()
    .to_string();

    assert_eq!(singleton_next_run, "((70 0xff1dd6715c5de8101added4088b446308432f6f96f7ad95add6ec884834d7206) (51 0xe435c0e8158464c825adb748900108adba4877bcd960c485f3effd3b8438f7e9 1))");
}
