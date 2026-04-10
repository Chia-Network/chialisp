use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use clvmr::allocator::Allocator;
use num_bigint::ToBigInt;

use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use crate::compiler::cldb::{CldbNoOverride, CldbRun, CldbRunEnv};
use crate::compiler::clvm::start_step;
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::prims;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;

/*
pub struct ClvmHamt {
    runner: Rc<dyn TRunProgram>,
    prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    program_name: String,
    program_lines: Rc<Vec<String>>,
    program: Rc<SExp>,
    symbols: HashMap<String, String>,
    hamt: Rc<SExp>,
}

impl ClvmHamt {
    pub fn load(program_name: &str) -> Self {
        let program_text = fs::read_to_string(program_name).expect("should exist");
        let mut allocator = Allocator::new();
        let runner = Rc::new(DefaultProgramRunner::new());
        let opts = Rc::new(DefaultCompilerOpts::new(program_name))
            .set_dialect(AcceptedDialect {
                stepping: Some(1000000),
                strict: true,
                int_fix: true,
            })
            .set_optimize(false);
        let mut prim_map = HashMap::new();
        for p in prims::prims().iter() {
            prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
        }

        let mut symbols = HashMap::new();
        let program = compile_file(
            &mut allocator,
            runner.clone(),
            opts,
            &program_text,
            &mut symbols,
        )
        .expect("should compile");
        let program_lines: Vec<String> = program_text.lines().map(|x| x.to_string()).collect();

        let loc = Srcloc::start(&program_name);
        let nil = Rc::new(SExp::Nil(loc.clone()));
        let mut instance = ClvmHamt {
            runner: runner.clone(),
            prim_map: Rc::new(prim_map),
            program_name: program_name.to_string(),
            program_lines: Rc::new(program_lines),
            program: Rc::new(program),
            symbols,
            hamt: nil.clone(),
        };

        let start_args = Rc::new(SExp::Cons(loc.clone(), nil.clone(), nil.clone()));
        let start_hamt = instance.run_with_args(start_args);
        instance.hamt = start_hamt;

        instance
    }

    pub fn get(&self, idx: usize) -> Rc<SExp> {
        let nil = Rc::new(SExp::Nil(self.program.loc()));
        let args = Rc::new(SExp::Cons(
            self.program.loc(),
            self.hamt.clone(),
            Rc::new(SExp::Cons(
                self.program.loc(),
                Rc::new(SExp::Integer(self.program.loc(), idx.to_bigint().unwrap())),
                nil.clone(),
            )),
        ));
        let res = self.run_with_args(args);
        res
    }

    pub fn put(&mut self, idx: usize, elt: Rc<SExp>) {
        let nil = Rc::new(SExp::Nil(self.program.loc()));
        let args = Rc::new(SExp::Cons(
            self.program.loc(),
            self.hamt.clone(),
            Rc::new(SExp::Cons(
                self.program.loc(),
                Rc::new(SExp::Integer(self.program.loc(), idx.to_bigint().unwrap())),
                Rc::new(SExp::Cons(self.program.loc(), elt.clone(), nil.clone())),
            )),
        ));
        let new_hamt = self.run_with_args(args);
        self.hamt = new_hamt;
    }

    fn run_with_args(&self, args: Rc<SExp>) -> Rc<SExp> {
        let mut allocator = Allocator::new();
        let step = start_step(self.program.clone(), args);
        let cldbenv = CldbRunEnv::new(
            Some(self.program_name.to_string()),
            self.program_lines.clone(),
            Box::new(CldbNoOverride::new_symbols(self.symbols.clone())),
        );
        let mut cldbrun = CldbRun::new(
            self.runner.clone(),
            self.prim_map.clone(),
            Box::new(cldbenv),
            step,
        );

        loop {
            if cldbrun.is_ended() {
                return cldbrun.final_result().expect("should return a value");
            }

            let _ = cldbrun.step(&mut allocator);
        }
    }
}

#[test]
fn test_hamt_update_and_retrieve() {
    let mut hamt = ClvmHamt::load("resources/tests/strict/hamt.clsp");
    let test_array: Vec<Rc<SExp>> = (0..40)
        .map(|i| Rc::new(SExp::Integer(hamt.program.loc(), i.to_bigint().unwrap())))
        .collect();
    // Check that every element of the hamt is nil.
    let nil = Rc::new(SExp::Nil(hamt.program.loc()));
    for i in 0..20 {
        assert_eq!(hamt.get(i), nil);
    }

    // Put items 1,2,3,4 in at 4,9,14,19
    for i in 1..5 {
        hamt.put(i * 5 - 1, test_array[i].clone());
    }

    // Check
    for i in 0..20 {
        if (i + 1) % 5 == 0 {
            assert_eq!(hamt.get(i), test_array[(i + 1) / 5]);
        } else {
            assert_eq!(hamt.get(i), nil);
        }
    }

    // Put 5,6,7,8 items in at 0,5,10,15
    for i in 0..4 {
        hamt.put(i * 5, test_array[i + 5].clone());
    }

    // Check
    for i in 0..20 {
        if i % 5 == 0 {
            assert_eq!(hamt.get(i), test_array[(i / 5) + 5]);
        } else if ((i + 1) % 5) == 0 {
            assert_eq!(hamt.get(i), test_array[(i + 1) / 5]);
        } else {
            assert_eq!(hamt.get(i), nil);
        }
    }

    // Put in all the odd numbers
    for i in 0..20 {
        hamt.put(i, test_array[1 + (i * 2)].clone());
    }

    // Check
    for i in 0..20 {
        assert_eq!(hamt.get(i), test_array[1 + (i * 2)]);
    }

    // Put in all the even numbers
    for i in 0..20 {
        hamt.put(i, test_array[i * 2].clone());
    }

    // Check
    for i in 0..20 {
        assert_eq!(hamt.get(i), test_array[i * 2]);
    }
}
*/
