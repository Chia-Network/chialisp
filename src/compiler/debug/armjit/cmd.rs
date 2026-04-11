use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use argh::FromArgs;
use clvmr::{run_program, Allocator, NodePtr};

use crate::classic::clvm_tools::binutils::assemble;
use crate::classic::clvm_tools::comp_input::RunAndCompileInputData;
use crate::classic::clvm_tools::ir::reader::read_ir;
use crate::classic::clvm_tools::stages::run;
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use crate::classic::clvm_tools::stages::stage_2::operators::run_program_for_search_paths;
use crate::classic::platform::argparse::ArgumentValue;
use crate::compiler::clvm::convert_from_clvm_rs;
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::debug::armjit::code::{Program, TARGET_ADDR};
use crate::compiler::debug::armjit::emu::Emu;
use crate::compiler::debug::armjit::emu_stub::{run_stub, start_stub};
use crate::compiler::debug::build_symbol_table_mut;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::sexp::{decode_string, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;

/// Translate a chialisp program to debug as an arm elf executable.
#[derive(FromArgs)]
pub struct Args {
    /// include paths
    #[argh(option, short = 'i')]
    pub include: Vec<String>,

    #[argh(option, short = 'o', description = "output file")]
    pub output: String,

    /// file name
    #[argh(positional)]
    pub filename: String,

    /// initial env
    #[argh(positional)]
    pub env: String,
}

pub fn spin_up_emulation(
    elf_bin: &[u8],
    symbols: Rc<HashMap<String, String>>,
) -> Result<(), String> {
    // Tiny start.
    let mut emu = Emu::new(elf_bin, TARGET_ADDR, symbols)
        .map_err(|e| format!("could not create emulator: {e:?}"))?;
    let connection = start_stub().map_err(|e| format!("could not start gdb service: {e:?}"))?;
    run_stub(connection, &mut emu).map_err(|e| format!("could not run program for gdb: {e:?}"))
}

pub fn compile_to_arm_elf(args: &Args) -> Result<(Vec<u8>, HashMap<String, String>), String> {
    let search_paths = args.include.clone();

    let argfile = if let Ok(res) = fs::read_to_string(&args.filename) {
        res
    } else {
        eprintln!("error reading {}", args.filename);
        std::process::exit(1);
    };

    let srcloc = Srcloc::start(&args.filename);
    let mut allocator = Allocator::new();
    let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
    let mut symbol_table = HashMap::new();

    let mut allocator = Allocator::new();
    let mut arguments: HashMap<String, ArgumentValue> = HashMap::default();
    arguments.insert(
        "path_or_code".to_string(),
        ArgumentValue::ArgString(Some(args.filename.clone()), argfile.clone()),
    );
    arguments.insert(
        "env".to_string(),
        ArgumentValue::ArgString(None, args.env.clone()),
    );
    arguments.insert(
        "include".to_string(),
        ArgumentValue::ArgArray(
            args.include
                .iter()
                .map(|i| ArgumentValue::ArgString(None, i.clone()))
                .collect(),
        ),
    );

    let parsed = RunAndCompileInputData::new(&mut allocator, &arguments)?;
    let special_runner =
        run_program_for_search_paths(&parsed.use_filename(), &parsed.search_paths, true, 0);

    let opts = DefaultCompilerOpts::new(&args.filename).set_search_paths(&parsed.search_paths);

    let compiled = if parsed.dialect.stepping.is_some() {
        let compiled = compile_file(&mut allocator, runner, opts, &argfile, &mut symbol_table)
            .map_err(|e| format!("failed to compile chialisp: {e:?}"))?;
        build_symbol_table_mut(&mut symbol_table, &compiled);
        eprintln!("compiled {compiled}");
        Rc::new(compiled)
    } else {
        let compile_invoke_code = run(&mut allocator);
        let assembled_sexp = assemble(&mut allocator, &argfile)
            .map_err(|e| format!("failed to assemble clvm {e:?}"))?;
        let input_sexp = allocator
            .new_pair(assembled_sexp, NodePtr::NIL)
            .map_err(|e| format!("failed to allocate compiler args {e:?}"))?;
        special_runner.set_compiler_opts(Some(opts));
        let run_program_output = special_runner
            .run_program(&mut allocator, compile_invoke_code, input_sexp, None)
            .map_err(|e| format!("failed to run classic compiler: {e:?}"))?;
        symbol_table = special_runner.get_compiles();
        convert_from_clvm_rs(
            &mut allocator,
            Srcloc::start(&args.filename),
            run_program_output.1,
        )
        .map_err(|e| format!("failed to convert clvm {e:?}"))?
    };

    let env_node =
        assemble(&mut allocator, &args.env).map_err(|e| format!("failed to read env: {e:?}"))?;
    let env = convert_from_clvm_rs(&mut allocator, Srcloc::start("*env*"), env_node)
        .map_err(|e| format!("failed to convert env program: {e:?}"))?;
    let symbols = Rc::new(symbol_table.clone());

    let program = Program::new(
        &args.filename,
        &args.output,
        compiled,
        env,
        TARGET_ADDR,
        symbols.clone(),
    )
    .expect("should generate");

    let output = program
        .to_elf(&args.output)
        .map_err(|e| format!("failed to create elf output: {e:?}"))?;

    Ok((output, symbol_table))
}
