use chialisp::classic::clvm_tools::cmds::opd_status;
use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    ExitCode::from(opd_status(&args) as u8)
}
