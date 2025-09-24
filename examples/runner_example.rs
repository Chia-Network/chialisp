extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;

use clvm_tools_rs::classic::clvm::__type_compatibility__::Stream;
use clvm_tools_rs::classic::clvm::serialize::sexp_to_stream;
use clvm_tools_rs::classic::clvm_tools::binutils::assemble_from_ir;
use clvm_tools_rs::classic::clvm_tools::clvmc::compile_clvm_text;
use clvm_tools_rs::classic::clvm_tools::ir::reader::read_ir;
use clvm_tools_rs::classic::clvm_tools::stages::stage_0::TRunProgram;
use clvm_tools_rs::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, RunProgramOption};
use clvm_tools_rs::compiler::compiler::DefaultCompilerOpts;
use clvm_tools_rs::compiler::comptypes::CompilerOpts;
use clvmr::allocator::Allocator;

fn main() {
    // Example Chialisp program: factorial function
    let clsp =
        "(mod (n) (defun factorial (x) (if (= x 0) 1 (* x (factorial (- x 1))))) (factorial n))";
    let filename = "*inline*";
    let search_paths: Vec<String> = vec![];

    // 1) Allocator and compiler opts
    let mut allocator = Allocator::new();
    let opts = Rc::new(DefaultCompilerOpts::new(filename)).set_search_paths(&search_paths);

    // 2) Compile Chialisp -> CLVM node
    let mut symbols: BTreeMap<String, String> = BTreeMap::new();
    let compiled_prog = compile_clvm_text(&mut allocator, opts, &mut symbols, clsp, filename, true)
        .expect("compile failed");

    // 3) Build arguments (n = 5)
    let args_ir = read_ir("(5)").expect("arg parse failed");
    let args = assemble_from_ir(&mut allocator, Rc::new(args_ir)).expect("assemble args failed");

    // 4) Run
    let runner = DefaultProgramRunner::new();
    let result = runner
        .run_program(
            &mut allocator,
            compiled_prog,
            args,
            Some(RunProgramOption::default()),
        )
        .expect("run failed");

    // 5) Print result
    let mut out = Stream::new(None);
    sexp_to_stream(&mut allocator, result.1, &mut out);
    println!("{}", out.get_value().hex());
}
