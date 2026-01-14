#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

// Minimal no_std compatible monadic do-notation macro (replaces do_notation crate)
// Must be defined before other modules that use it
#[macro_use]
mod macros;

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate indoc;

extern crate clvmr as clvm_rs;

pub mod util;

pub mod classic;
pub mod compiler;
