use std::fs;
use std::rc::Rc;

use chialisp::compiler::debug::armjit::cmd::{compile_to_arm_elf, spin_up_emulation, Args};

fn run_arm_conversion() -> Result<(), String> {
    let args: Args = argh::from_env();

    let arm_elf = compile_to_arm_elf(&args)?;

    // copy all in-memory sections from the ELF file into system RAM
    (|| {
        fs::write(&args.output, &arm_elf.object_file)?;
        fs::write(format!("{}.clsp", args.output), &arm_elf.synthetic_source)
    })()
    .map_err(|e| format!("could not write elf file: {e:?}"))?;

    spin_up_emulation(
        &arm_elf.object_file,
        Rc::new(arm_elf.symbol_table),
        Some(9001),
    )?;
    Ok(())
}

fn main() {
    if let Err(e) = run_arm_conversion() {
        eprintln!("Error {e}");
        std::process::exit(1);
    }
}
