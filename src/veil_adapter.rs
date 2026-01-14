//! Veil-compatible adapter for clvmr
//!
//! This module provides a bridge between Veil's clvm_zk_core interface
//! and the clvmr runtime. It allows Veil's zkVM guests to use clvmr
//! instead of the custom ClvmEvaluator.
//!
//! # Usage
//!
//! ```ignore
//! use clvm_tools_rs::veil_adapter::{VeilEvaluator, Hasher, BlsVerifier, EcdsaVerifier};
//!
//! fn my_hasher(data: &[u8]) -> [u8; 32] { ... }
//! fn my_bls_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, &'static str> { ... }
//! fn my_ecdsa_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, &'static str> { ... }
//!
//! let evaluator = VeilEvaluator::new(my_hasher, my_bls_verify, my_ecdsa_verify);
//! let (result_bytes, cost) = evaluator.run_program(&program_bytes, &args_bytes, max_cost)?;
//! ```

extern crate alloc;

use alloc::vec::Vec;

use crate::clvmr::{
    Allocator, ChiaDialect, CryptoHandlers, NodePtr,
    node_from_bytes, node_to_bytes, run_program,
    cost::Cost,
    reduction::{Response, Reduction},
    error::EvalErr,
};

/// Hasher function type (matches Veil's clvm_zk_core)
pub type Hasher = fn(&[u8]) -> [u8; 32];

/// BLS verifier function type (matches Veil's clvm_zk_core)  
pub type BlsVerifier = fn(&[u8], &[u8], &[u8]) -> Result<bool, &'static str>;

/// ECDSA verifier function type (matches Veil's clvm_zk_core)
pub type EcdsaVerifier = fn(&[u8], &[u8], &[u8]) -> Result<bool, &'static str>;

// For zkVM (single-threaded), we use a simple static to pass crypto callbacks
// to OpHandler functions (which are fn pointers that can't capture state)
static mut CURRENT_BLS_VERIFIER: Option<BlsVerifier> = None;
static mut CURRENT_ECDSA_VERIFIER: Option<EcdsaVerifier> = None;

/// Veil-compatible CLVM evaluator backed by clvmr
pub struct VeilEvaluator {
    hasher: Hasher,
    bls_verifier: BlsVerifier,
    ecdsa_verifier: EcdsaVerifier,
}

impl VeilEvaluator {
    /// Create a new VeilEvaluator with injected crypto functions
    pub fn new(hasher: Hasher, bls_verifier: BlsVerifier, ecdsa_verifier: EcdsaVerifier) -> Self {
        Self {
            hasher,
            bls_verifier,
            ecdsa_verifier,
        }
    }

    /// Run a CLVM program with the given arguments
    ///
    /// # Arguments
    /// * `program` - Serialized CLVM program bytecode
    /// * `args` - Serialized CLVM arguments
    /// * `max_cost` - Maximum cost allowed for execution
    ///
    /// # Returns
    /// * `Ok((result_bytes, cost))` - Serialized result and cost consumed
    /// * `Err(msg)` - Error message if execution fails
    pub fn run_program(
        &self,
        program: &[u8],
        args: &[u8],
        max_cost: u64,
    ) -> Result<(Vec<u8>, u64), &'static str> {
        // Store verifiers in thread-local for OpHandler access
        // SAFETY: zkVM is single-threaded
        unsafe {
            CURRENT_BLS_VERIFIER = Some(self.bls_verifier);
            CURRENT_ECDSA_VERIFIER = Some(self.ecdsa_verifier);
        }

        let mut allocator = Allocator::new();

        // Deserialize program and args
        let program_node = node_from_bytes(&mut allocator, program)
            .map_err(|_| "failed to deserialize program")?;
        let args_node = node_from_bytes(&mut allocator, args)
            .map_err(|_| "failed to deserialize args")?;

        // Create dialect with crypto handlers
        let handlers = CryptoHandlers::new()
            .with_bls_verify(bls_verify_handler)
            .with_secp256k1_verify(ecdsa_verify_handler);

        let dialect = ChiaDialect::new_with_handlers(0, handlers);

        // Run the program
        let reduction = run_program(&mut allocator, &dialect, program_node, args_node, max_cost)
            .map_err(|_| "CLVM execution failed")?;

        // Serialize result
        let result_bytes = node_to_bytes(&allocator, reduction.1)
            .map_err(|_| "failed to serialize result")?;

        // Clear verifiers
        unsafe {
            CURRENT_BLS_VERIFIER = None;
            CURRENT_ECDSA_VERIFIER = None;
        }

        Ok((result_bytes, reduction.0))
    }

    /// Get the hasher function
    pub fn hasher(&self) -> Hasher {
        self.hasher
    }
}

/// OpHandler wrapper for BLS verification (opcode 59)
///
/// Extracts (pubkey, message, signature) from CLVM args and calls the injected verifier
fn bls_verify_handler(
    allocator: &mut Allocator,
    args: NodePtr,
    _max_cost: Cost,
) -> Response {
    // Get the verifier
    let verifier = unsafe {
        CURRENT_BLS_VERIFIER.ok_or(EvalErr::Unimplemented(args))?
    };

    // Extract arguments: (pubkey message signature)
    let (pk, msg, sig) = extract_three_atoms(allocator, args)?;

    // Call verifier
    match verifier(&pk, &msg, &sig) {
        Ok(true) => {
            // Return nil on success (standard CLVM convention)
            Ok(Reduction(0, allocator.nil()))
        }
        Ok(false) => {
            // Verification failed
            Err(EvalErr::BLSVerifyFailed(args))
        }
        Err(_) => {
            Err(EvalErr::BLSVerifyFailed(args))
        }
    }
}

/// OpHandler wrapper for ECDSA verification (secp256k1)
///
/// Extracts (pubkey, message, signature) from CLVM args and calls the injected verifier
fn ecdsa_verify_handler(
    allocator: &mut Allocator,
    args: NodePtr,
    _max_cost: Cost,
) -> Response {
    // Get the verifier
    let verifier = unsafe {
        CURRENT_ECDSA_VERIFIER.ok_or(EvalErr::Unimplemented(args))?
    };

    // Extract arguments: (pubkey message signature)
    let (pk, msg, sig) = extract_three_atoms(allocator, args)?;

    // Call verifier
    match verifier(&pk, &msg, &sig) {
        Ok(true) => {
            // Return nil on success
            Ok(Reduction(0, allocator.nil()))
        }
        Ok(false) => {
            Err(EvalErr::Secp256Failed(args))
        }
        Err(_) => {
            Err(EvalErr::Secp256Failed(args))
        }
    }
}

/// Extract three atom arguments from a CLVM list
fn extract_three_atoms(
    allocator: &Allocator,
    args: NodePtr,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), EvalErr> {
    use crate::clvmr::SExp;

    // args should be (pk . (msg . (sig . nil)))
    let (first, rest1) = match allocator.sexp(args) {
        SExp::Pair(f, r) => (f, r),
        _ => return Err(EvalErr::Invalid(args)),
    };

    let (second, rest2) = match allocator.sexp(rest1) {
        SExp::Pair(f, r) => (f, r),
        _ => return Err(EvalErr::Invalid(args)),
    };

    let (third, _) = match allocator.sexp(rest2) {
        SExp::Pair(f, r) => (f, r),
        _ => return Err(EvalErr::Invalid(args)),
    };

    // Extract bytes from atoms
    let pk = match allocator.sexp(first) {
        SExp::Atom => allocator.atom(first).as_ref().to_vec(),
        _ => return Err(EvalErr::Invalid(args)),
    };

    let msg = match allocator.sexp(second) {
        SExp::Atom => allocator.atom(second).as_ref().to_vec(),
        _ => return Err(EvalErr::Invalid(args)),
    };

    let sig = match allocator.sexp(third) {
        SExp::Atom => allocator.atom(third).as_ref().to_vec(),
        _ => return Err(EvalErr::Invalid(args)),
    };

    Ok((pk, msg, sig))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_hasher(data: &[u8]) -> [u8; 32] {
        let mut result = [0u8; 32];
        if !data.is_empty() {
            result[0] = data[0];
        }
        result
    }

    fn dummy_bls_verify(_pk: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, &'static str> {
        Ok(true) // Always succeeds for testing
    }

    fn dummy_ecdsa_verify(_pk: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, &'static str> {
        Ok(true) // Always succeeds for testing
    }

    #[test]
    fn test_simple_program() {
        let evaluator = VeilEvaluator::new(dummy_hasher, dummy_bls_verify, dummy_ecdsa_verify);

        // Program: (q . 42)
        let program = vec![0xff, 0x01, 0x2a];
        // Args: nil
        let args = vec![0x80];

        let (result, cost) = evaluator.run_program(&program, &args, 1000000).unwrap();

        // Should return 42
        assert_eq!(result, vec![0x2a]);
        println!("VeilEvaluator test passed! Result: {:?}, Cost: {}", result, cost);
    }

    #[test]
    fn test_addition() {
        let evaluator = VeilEvaluator::new(dummy_hasher, dummy_bls_verify, dummy_ecdsa_verify);

        // Program: (+ (q . 2) (q . 3))
        let program = vec![
            0xff, 0x10,       // (+
            0xff,             //   (
            0xff, 0x01, 0x02, //     (q . 2)
            0xff,             //     .
            0xff, 0x01, 0x03, //     (q . 3)
            0x80,             //   )
        ];
        let args = vec![0x80];

        let (result, cost) = evaluator.run_program(&program, &args, 1000000).unwrap();

        assert_eq!(result, vec![0x05]);
        println!("VeilEvaluator addition test passed! 2+3={:?}, Cost: {}", result, cost);
    }
}
