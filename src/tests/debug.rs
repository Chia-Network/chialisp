use std::fs;
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::mpsc::channel;
use std::thread;
use subprocess::Exec;
use tempfile::NamedTempFile;

use crate::compiler::debug::armjit::cmd::{compile_to_arm_elf, Args};
use crate::compiler::debug::armjit::code::TARGET_ADDR;
use crate::compiler::debug::armjit::emu::Emu;
use crate::compiler::debug::armjit::emu_stub::run_stub;

#[cfg(target_os = "linux")]
fn run_test_program_with_argument(file: &str, gdb_script: &str, env: &str) -> String {
    let temp_output = NamedTempFile::with_suffix("elf").unwrap();
    let args = Args {
        include: vec![],
        output: temp_output.path().to_string_lossy().to_string(),
        filename: file.to_string(),
        env: env.to_string(),
    };
    let (sender, receiver) = channel();
    let t = thread::spawn(move || {
        let output = compile_to_arm_elf(&args).expect("should compile");
        eprintln!("compile_to_arm_elf succeeded");
        fs::write(&args.output, &output.object_file).expect("should be able to write file");
        eprintln!("elf file written to {}", args.output);
        // Tiny start.
        let mut emu = Emu::new(
            &output.object_file,
            TARGET_ADDR,
            Rc::new(output.symbol_table),
        )
        .unwrap();
        let sockaddr = format!("127.0.0.1:0");
        let sock = TcpListener::bind(sockaddr).unwrap();
        let local_addr = sock.local_addr().unwrap();
        sender.send(local_addr).unwrap();
        eprintln!("Waiting for a GDB connection on {:?}...", local_addr);
        let (stream, _addr) = sock.accept().unwrap();
        run_stub(Box::new(stream), &mut emu)
            .map_err(|e| format!("could not run program for gdb: {e:?}"))
            .unwrap();
    });
    let addr = receiver.recv().unwrap();
    eprintln!("connect gdb to {addr}");
    let gdb_run_stdout = Exec::cmd(gdb_script)
        .args(&[addr.to_string()])
        .capture()
        .expect("should complete")
        .stdout_str();
    eprintln!(">> {gdb_run_stdout}");
    t.join().expect("should finish");
    fs::remove_file(temp_output.path()).expect("should work");
    gdb_run_stdout
}

#[cfg(target_os = "linux")]
#[test]
fn test_smoke_arm_debug() {
    let gdb_run = run_test_program_with_argument(
        "resources/tests/simple_deinline_case_23.clsp",
        "./resources/tests/test_sdc_gdb.sh",
        "(5)",
    );
    assert!(gdb_run.contains("CLVM: 6000030"));
}

#[cfg(target_os = "linux")]
#[test]
fn test_smoke_arm_debug_exception() {
    let gdb_run = run_test_program_with_argument(
        "resources/tests/simple_deinline_case_23.clsp",
        "./resources/tests/test_sdc_gdb.sh",
        "()",
    );
    assert!(gdb_run.contains("SIGABRT"));
}
