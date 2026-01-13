extern crate clvmr as clvm_rs;

use std::env;
use std::rc::Rc;

use chialisp::compiler::compiler::DefaultCompilerOpts;
use chialisp::compiler::frontend::frontend;
use chialisp::compiler::sexp::parse_sexp;
use chialisp::compiler::srcloc::Srcloc;

use chialisp::util::ErrInto;

fn main() {
    let opts = Rc::new(DefaultCompilerOpts::new("*program*"));
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Give a chialisp program to convert to json AST");
        return;
    }

    let loc = Srcloc::start("*program*");
    let result = parse_sexp(loc, args[1].bytes())
        .err_into()
        .and_then(|parsed_program| frontend(opts.clone(), &parsed_program));
    match result {
        Ok(program) => match serde_json::to_string(&program) {
            Ok(output) => println!("{output}"),
            Err(e) => {
                println!("{e:?}");
            }
        },
        Err(e) => {
            println!("{e:?}");
        }
    }
}
