//! Chialisp compiler and other clvm tools.
//!
//! ./src/classic -- overall port of python tools.
//! ./src/classic/clvm_tools/stages/stage_2 -- port of original clvm compiler.
//!
//! An approach to clvm compilation which introduces line numbers, strict use of atoms,
//! macros which can tell the difference between identifiers and quoted values, better error
//! reporting and other features.
//!
//! ./src/compiler -- a more ergonomic chialisp with more features.
#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate indoc;

#[macro_use]
extern crate do_notation;

#[cfg(all(not(test), not(target_family = "wasm"), feature = "extension-module"))]
extern crate pyo3;

extern crate tempfile;

extern crate clvmr as clvm_rs;

pub mod util;

pub mod classic;
pub mod compiler;

// Python impl
#[cfg(all(not(test), not(target_family = "wasm"), feature = "extension-module"))]
mod py;

#[cfg(test)]
mod tests;
