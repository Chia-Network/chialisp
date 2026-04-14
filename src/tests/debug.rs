use std::fs;
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::mpsc::channel;
use std::thread;
use subprocess::Exec;

use crate::compiler::debug::armjit::cmd::{compile_to_arm_elf, Args};
use crate::compiler::debug::armjit::code::TARGET_ADDR;
use crate::compiler::debug::armjit::emu::Emu;
use crate::compiler::debug::armjit::emu_stub::run_stub;

#[cfg(target_os = "linux")]
#[test]
fn test_smoke_arm_debug() {
    let args = Args {
        include: vec![],
        output: "sdc.elf".to_string(),
        filename: "resources/tests/simple_deinline_case_23.clsp".to_string(),
        env: "(5)".to_string(),
    };
    let (sender, receiver) = channel();
    let t = thread::spawn(move || {
        let (output, symbols) = compile_to_arm_elf(&args).expect("should compile");
        fs::write(&args.output, &output).expect("should be able to write file");
        // Tiny start.
        let mut emu = Emu::new(&output, TARGET_ADDR, Rc::new(symbols)).unwrap();
        let sockaddr = format!("127.0.0.1:0");
        let sock = TcpListener::bind(sockaddr).unwrap();
        let local_addr = sock.local_addr().unwrap();
        sender.send(local_addr).unwrap();
        eprintln!("Waiting for a GDB connection on {:?}...", local_addr);
        let (stream, _addr) = sock.accept().unwrap();
        run_stub(Box::new(stream), &mut emu).map_err(|e| format!("could not run program for gdb: {e:?}")).unwrap();
    });
    let addr = receiver.recv().unwrap();
    eprintln!("connect gdb to {addr}");
    let gdb_run_stdout = Exec::cmd(format!("./resources/tests/test_sdc_gdb.sh"))
        .args(&[addr.to_string()])
        .capture()
        .expect("should complete")
        .stdout_str();
    eprintln!(">> {gdb_run_stdout}");
    t.join().expect("should finish");
    fs::remove_file("sdc.elf").expect("should work");
    assert!(gdb_run_stdout.contains("CLVM: 6000030"));
}
