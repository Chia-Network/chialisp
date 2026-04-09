use std::fs;
use std::rc::Rc;
use std::thread;
use subprocess::Exec;
use crate::compiler::debug::armjit::cmd::{Args, compile_to_arm_elf, spin_up_emulation};

#[cfg(target_os = "linux")]
#[test]
fn test_smoke_arm_debug() {
    let args = Args {
        include: vec![],
        output: "sdc.elf".to_string(),
        filename: "resources/tests/simple_deinline_case_23.clsp".to_string(),
        env: "(5)".to_string(),
    };
    let t = thread::spawn(move || {
        let (output, symbols) = compile_to_arm_elf(&args).expect("should compile");
        fs::write(&args.output, &output).expect("should be able to write file");
        spin_up_emulation(&output, Rc::new(symbols)).expect("should run");
    });
    let gdb_run_stdout = Exec::cmd("./resources/tests/test_sdc_gdb.sh").capture().expect("should complete").stdout_str();
    eprintln!(">> {gdb_run_stdout}");
    t.join().expect("should finish");
    fs::remove_file("sdc.elf").expect("should work");
    assert!(gdb_run_stdout.contains("CLVM: 6000030"));
}
