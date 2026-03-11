use std::path::PathBuf;

use crate::classic::platform::argparse::{ArgumentValue, ArgumentValueConv};

pub mod argparse;
pub mod distutils;

pub struct PathJoin {}

impl ArgumentValueConv for PathJoin {
    fn convert(&self, arg: &str) -> Result<ArgumentValue, String> {
        let mut p = PathBuf::new();
        p.push(arg);
        Ok(ArgumentValue::ArgString(
            None,
            p.as_path().to_str().unwrap().to_string(),
        ))
    }
}
