use chialisp::classic::clvm_tools::cmds::run;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    run(&args);
}
