use num_bigint::ToBigInt;
use std::rc::Rc;

use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::{Srcloc, Until};

mod assign;
mod cldb;
mod clvm;
mod compiler;
mod evaluate;
mod fuzz;
mod fuzz_assign;
mod optimizer;
mod preprocessor;
mod repl;
mod restargs;
mod runtypes;
mod singleton;
mod srcloc;
mod usecheck;

#[test]
fn test_sexp_parse_print() {
    let start = Srcloc::start(&"test.cl".to_string());
    let mut end = start.clone();
    end.col = 2;
    end.until = Some(Until { line: 1, col: 8 });

    let mut atom_loc = start.clone();
    atom_loc.col = 2;
    atom_loc.until = Some(Until { line: 1, col: 4 });

    let mut num_loc = start.clone();
    num_loc.col = 7;

    let my_result: Result<Vec<SExp>, (Srcloc, String)> = Ok(vec![SExp::Cons(
        end,
        Rc::new(SExp::Atom(atom_loc, vec!['h' as u8, 'i' as u8])),
        Rc::new(SExp::Integer(num_loc, 3_i32.to_bigint().unwrap())),
    )]);

    let parse_result = parse_sexp(start.clone(), "(hi . 3)".bytes());
    assert_eq!(format!("{:?}", parse_result), format!("{:?}", my_result));

    assert_eq!(parse_result.unwrap()[0].to_string(), "(hi . 3)".to_string())
}
