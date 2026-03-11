extern crate clvmr as clvm_rs;

use std::collections::HashMap;
use std::env;
use std::rc::Rc;

use clvm_rs::allocator::Allocator;

use chialisp::compiler::compiler::DefaultCompilerOpts;
use chialisp::compiler::evaluate::{Evaluator, EVAL_STACK_LIMIT};
use chialisp::compiler::frontend::frontend;
use chialisp::compiler::sexp::parse_sexp;
use chialisp::compiler::srcloc::Srcloc;

use chialisp::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use chialisp::util::ErrInto;

fn main() {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts = Rc::new(DefaultCompilerOpts::new("*program*"));
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("give a chialisp program to minify");
        return;
    }

    let loc = Srcloc::start("*program*");
    let _ = parse_sexp(loc, args[1].bytes())
        .err_into()
        .and_then(|parsed_program| frontend(opts.clone(), &parsed_program))
        .and_then(|program| {
            let e = Evaluator::new(opts.clone(), runner.clone(), program.helpers.clone());
            e.shrink_bodyform(
                &mut allocator,
                program.args.clone(),
                &HashMap::new(),
                program.exp,
                false,
                Some(EVAL_STACK_LIMIT),
            )
        })
        .map(|result| {
            println!("shrunk: {}", result.to_sexp());
        })
        .map_err(|e| {
            println!("failed: {e:?}");
        });
}
