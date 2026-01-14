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

// Re-export clvmr for zkVM usage
pub use clvmr;

// Re-export key types for convenience
pub use clvmr::{
    Allocator, ChiaDialect, CryptoHandlers, NodePtr, OpHandler,
    node_from_bytes, node_to_bytes, node_to_bytes_limit, run_program,
};

// Re-export veil adapter types
pub use veil_adapter::{VeilEvaluator, Hasher, BlsVerifier, EcdsaVerifier};

pub mod util;
pub mod veil_adapter;

pub mod classic;
pub mod compiler;
