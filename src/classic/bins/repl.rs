extern crate clvmr as clvm_rs;

use std::io::{self, BufRead, Write};

use std::rc::Rc;

use clvm_rs::allocator::Allocator;

use chialisp::compiler::compiler::DefaultCompilerOpts;
use chialisp::compiler::repl::Repl;

use chialisp::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;

fn main() {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let opts = Rc::new(DefaultCompilerOpts::new("*program*"));
    let stdin = io::stdin();
    let mut repl = Repl::new(opts, runner);

    print!(">>> ");
    io::stdout().flush().unwrap();

    for l in stdin.lock().lines() {
        match l {
            Err(_) => break,
            Ok(line) => {
                let _ = repl
                    .process_line(&mut allocator, line)
                    .map(|result| {
                        if let Some(result) = result {
                            print!("{}\n>>> ", result.to_sexp());
                        } else {
                            print!("... ");
                        }
                    })
                    .map_err(|e| {
                        print!("failed: {e:?}\n>>> ");
                    });
            }
        }
        io::stdout().flush().unwrap();
    }
}
