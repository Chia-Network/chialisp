use chialisp::classic::clvm_tools::cmds::cldb;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let status = cldb(&args);
    if status.should_exit_with_error() {
        std::process::exit(1);
    }
}
