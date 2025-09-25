#![no_std]
extern crate alloc;

// Remove failed global imports

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate indoc;

#[macro_use]
extern crate do_notation;

// extern crate tempfile; // Removed for no_std

extern crate clvmr as clvm_rs;

pub mod util;

pub mod classic;
pub mod compiler;
