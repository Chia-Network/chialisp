use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Formatter;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::swap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::mpsc;
use std::thread;
use std::rc::Rc;

use argh::FromArgs;
use armv4t_emu::{reg, Cpu, Mode, Memory};
use clvmr::Allocator;
use elf_rs::{Elf, ElfFile, ProgramHeaderFlags, ProgramType, SectionHeaderFlags, SectionType};
use faerie::{ArtifactBuilder, Decl, Link, SectionKind};
use gimli::{Encoding, Format, LineEncoding, LittleEndian};
use gimli::write::{Address, DirectoryId, Dwarf, FileId, LineProgram, LineString, Section, Sections, UnitId, Unit};
use subprocess::{Popen, PopenConfig};
use target_lexicon::triple;
use tempfile::NamedTempFile;

use clvm_tools_rs::classic::clvm::casts::bigint_to_bytes_clvm;
use clvm_tools_rs::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use clvm_tools_rs::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};

use clvm_tools_rs::compiler::clvm::{sha256tree, truthy};
use clvm_tools_rs::compiler::comptypes::CompilerOpts;
use clvm_tools_rs::compiler::compiler::{DefaultCompilerOpts, compile_file};
use clvm_tools_rs::compiler::debug::build_symbol_table_mut;
use clvm_tools_rs::compiler::debug::armjit::code::{Program, TARGET_ADDR};
use clvm_tools_rs::compiler::debug::armjit::emu::Emu;
use clvm_tools_rs::compiler::debug::armjit::emu_stub::{run_stub, start_stub};
use clvm_tools_rs::compiler::debug::armjit::load::ElfLoader;
use clvm_tools_rs::compiler::debug::armjit::memory::PagedMemory;
use clvm_tools_rs::compiler::dialect::AcceptedDialect;
use clvm_tools_rs::compiler::sexp::{Atom, decode_string, NodeSel, SelectNode, SExp, ThisNode, parse_sexp};
use clvm_tools_rs::compiler::srcloc::Srcloc;

/// Translate a chialisp program to debug as an arm elf executable.
#[derive(FromArgs)]
struct Args {
    /// include paths
    #[argh(option, short='i')]
    pub include: Vec<String>,

    #[argh(option, short='o', description="output file")]
    pub output: String,

    /// file name
    #[argh(positional)]
    pub filename: String,

    /// initial env
    #[argh(positional)]
    pub env: String,
}

fn spin_up_emulation(signal_emu_startup_complete: Sender<()>, elf_bin: &[u8]) {
    // Tiny start.
    let mut emu = Emu::new(elf_bin, TARGET_ADDR).expect("should have elf");
    let mut connection = start_stub().expect("should start service");
    signal_emu_startup_complete.send(()).expect("should send");
    run_stub(connection, &mut emu).expect("should run"); // Until exit.
}

fn main() {
    let args: Args = argh::from_env();

    let mut search_paths = args.include.clone();

    let argfile =
        if let Ok(res) = fs::read(&args.filename) {
            res
        } else {
            panic!("error reading {}", args.filename);
        };

    let srcloc = Srcloc::start(&args.filename);
    let mut allocator = Allocator::new();
    let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
    let mut symbol_table = HashMap::new();
    let opts = Rc::new(DefaultCompilerOpts::new(&args.filename))
        .set_dialect(AcceptedDialect {
            stepping: Some(21),
            strict: true,
        })
        .set_optimize(false)
        .set_search_paths(&search_paths)
        .set_frontend_opt(false);
    let compiled =
        match compile_file(
            &mut allocator,
            runner,
            opts,
            &decode_string(&argfile),
            &mut symbol_table,
        ) {
            Ok(res) => res,
            Err(e) => panic!("{:?}", e)
        };
    build_symbol_table_mut(&mut symbol_table, &compiled);
    eprintln!("symbols {symbol_table:?}");
    let mut env =
        match parse_sexp(srcloc.clone(), args.env.bytes()) {
            Ok(res) => res,
            Err(e) => panic!("{:?}", e)
        };

    if env.is_empty() {
        env.push(Rc::new(SExp::Nil(srcloc.clone())));
    }

    let program = Program::new(
        &args.filename,
        Rc::new(compiled),
        env[0].clone(),
        symbol_table
    ).expect("should generate");

    let mut elf_out = program.to_elf(&args.output).expect("should be writable");
    // copy all in-memory sections from the ELF file into system RAM
    let mut elf_loader = ElfLoader::new(&elf_out, TARGET_ADDR).expect("should load");
    fs::write(&args.output, &elf_out).expect("should be able to write file");
    let (signal_emu_startup_complete, emu_startup_complete) = mpsc::channel();

    // Spin up our emulation.
    let t = thread::spawn(move || {
        let elf_out = elf_out;
        spin_up_emulation(signal_emu_startup_complete, &elf_out)
    });

    let _ = emu_startup_complete.recv().expect("should be able to start emu");
    // Startup done, so we can spawn gdb.
    let mut p = Popen::create(&["sleep", "5000"], PopenConfig {
        .. Default::default()
    }).expect("should be able to start gdb");
    t.join().expect("thread should join successfully");
}
