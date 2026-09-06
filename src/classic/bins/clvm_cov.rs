use chialisp::classic::clvm_tools::clvm_cov::clvm_cov;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let code = clvm_cov(&args);
    if code != 0 {
        std::process::exit(code);
    }
}
