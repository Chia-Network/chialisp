#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod allocator;
pub mod chia_dialect;
pub mod core_ops;
pub mod cost;
pub mod crypto_handlers;
pub mod dialect;
pub mod error;
pub mod more_ops;
pub mod number;
pub mod op_utils;
pub mod reduction;
pub mod run_program;
pub mod traverse_path;

// These modules require std for HashMap
#[cfg(feature = "std")]
pub mod f_table;
#[cfg(feature = "std")]
pub mod runtime_dialect;

// Serde module - some parts require std
#[cfg(feature = "std")]
pub mod serde;

#[cfg(feature = "bls-ops")]
pub mod bls_ops;

#[cfg(feature = "secp-ops")]
pub mod secp_ops;

#[cfg(feature = "keccak-ops")]
pub mod keccak256_ops;

pub use allocator::{Allocator, Atom, NodePtr, ObjectType, SExp};
pub use chia_dialect::ChiaDialect;
pub use crypto_handlers::{CryptoHandlers, OpHandler};
pub use run_program::run_program;

pub use chia_dialect::{ENABLE_KECCAK_OPS_OUTSIDE_GUARD, LIMIT_HEAP, MEMPOOL_MODE, NO_UNKNOWN_OPS};

#[cfg(feature = "counters")]
pub use run_program::run_program_with_counters;

#[cfg(feature = "pre-eval")]
pub use run_program::run_program_with_pre_eval;

#[cfg(feature = "counters")]
pub use run_program::Counters;

#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(all(test, feature = "std"))]
mod test_ops;
