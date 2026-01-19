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
    Allocator, ChiaDialect, CryptoHandlers,
    node_from_bytes, node_to_bytes, run_program,
};

/// Hasher function type (matches Veil's clvm_zk_core)
pub type Hasher = fn(&[u8]) -> [u8; 32];

/// BLS verifier function type (matches Veil's clvm_zk_core)  
pub type BlsVerifier = fn(&[u8], &[u8], &[u8]) -> Result<bool, &'static str>;

/// ECDSA verifier function type (matches Veil's clvm_zk_core)
pub type EcdsaVerifier = fn(&[u8], &[u8], &[u8]) -> Result<bool, &'static str>;

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
        let mut allocator = Allocator::new();

        // Deserialize program and args
        let program_node = node_from_bytes(&mut allocator, program)
            .map_err(|_| "failed to deserialize program")?;
        let args_node = node_from_bytes(&mut allocator, args)
            .map_err(|_| "failed to deserialize args")?;

        // Create dialect with crypto handlers - functions passed directly!
        let handlers = CryptoHandlers::new()
            .with_sha256(self.hasher)
            .with_bls_verify(self.bls_verifier)
            .with_secp256k1_verify(self.ecdsa_verifier);

        let dialect = ChiaDialect::new_with_handlers(0, handlers);

        // Run the program
        let reduction = run_program(&mut allocator, &dialect, program_node, args_node, max_cost)
            .map_err(|_| "CLVM execution failed")?;

        // Serialize result
        let result_bytes = node_to_bytes(&allocator, reduction.1)
            .map_err(|_| "failed to serialize result")?;

        Ok((result_bytes, reduction.0))
    }

    /// Get the hasher function
    pub fn hasher(&self) -> Hasher {
        self.hasher
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::sync::atomic::{AtomicBool, Ordering};

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

        let (result, _cost) = evaluator.run_program(&program, &args, 1000000).unwrap();

        // Should return 42
        assert_eq!(result, vec![0x2a]);
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

        let (result, _cost) = evaluator.run_program(&program, &args, 1000000).unwrap();

        assert_eq!(result, vec![0x05]);
    }

    #[test]
    fn test_sha256_uses_injected_hasher() {
        // Custom hasher that returns a predictable result
        fn custom_hasher(data: &[u8]) -> [u8; 32] {
            let mut result = [0xABu8; 32];
            // Put input length in first byte to verify we got the right data
            result[0] = data.len() as u8;
            if !data.is_empty() {
                result[1] = data[0];
            }
            result
        }

        let evaluator = VeilEvaluator::new(custom_hasher, dummy_bls_verify, dummy_ecdsa_verify);

        // Program: (sha256 (q . "hi"))
        // sha256 = opcode 11 (0x0b)
        // "hi" = 0x6869
        let program = vec![
            0xff, 0x0b,             // (sha256
            0xff,                   //   (
            0xff, 0x01,             //     (q .
            0x82, 0x68, 0x69,       //       "hi")  -- 0x82 = 2-byte atom
            0x80,                   //   )
        ];
        let args = vec![0x80];

        let (result, _cost) = evaluator.run_program(&program, &args, 1000000).unwrap();

        // Result should be 32 bytes with our custom pattern
        assert_eq!(result.len(), 33); // 32 bytes + 1 byte length prefix in CLVM serialization
        // The actual atom is result[1..] since result[0] is the length indicator
        assert_eq!(result[1], 2);     // data length was 2 ("hi")
        assert_eq!(result[2], 0x68);  // first byte of "hi"
        assert_eq!(result[3], 0xAB);  // rest filled with 0xAB
    }

    // Static flags to track if handlers were called
    static BLS_HANDLER_CALLED: AtomicBool = AtomicBool::new(false);
    static ECDSA_HANDLER_CALLED: AtomicBool = AtomicBool::new(false);

    #[test]
    fn test_bls_verify_uses_injected_handler() {
        BLS_HANDLER_CALLED.store(false, Ordering::SeqCst);

        // Custom BLS verifier that tracks calls
        fn tracking_bls_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, &'static str> {
            BLS_HANDLER_CALLED.store(true, Ordering::SeqCst);
            // Verify we received the expected arguments
            assert_eq!(pk, &[0x01, 0x02, 0x03]); // pubkey
            assert_eq!(msg, &[0x04, 0x05]);       // message
            assert_eq!(sig, &[0x06, 0x07, 0x08]); // signature
            Ok(true)
        }

        let evaluator = VeilEvaluator::new(dummy_hasher, tracking_bls_verify, dummy_ecdsa_verify);

        // Program: (bls_verify (q . <pk>) (q . <msg>) (q . <sig>))
        // bls_verify = opcode 59 (0x3b)
        // We need: (59 (q . pk) (q . msg) (q . sig))
        let program = vec![
            0xff, 0x3b,                   // (bls_verify
            0xff,                         //   (
            0xff, 0x01,                   //     (q .
            0x83, 0x01, 0x02, 0x03,       //       pk = 3 bytes)
            0xff,                         //     .
            0xff, 0x01,                   //     (q .
            0x82, 0x04, 0x05,             //       msg = 2 bytes)
            0xff,                         //     .
            0xff, 0x01,                   //     (q .
            0x83, 0x06, 0x07, 0x08,       //       sig = 3 bytes)
            0x80,                         //     . nil
        ];
        let args = vec![0x80];

        let result = evaluator.run_program(&program, &args, 1000000);

        // Verify the handler was called
        assert!(BLS_HANDLER_CALLED.load(Ordering::SeqCst), "BLS handler was not called!");
        assert!(result.is_ok(), "BLS verify should succeed");
    }

    #[test]
    fn test_ecdsa_verify_uses_injected_handler() {
        ECDSA_HANDLER_CALLED.store(false, Ordering::SeqCst);

        // Custom ECDSA verifier that tracks calls
        fn tracking_ecdsa_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, &'static str> {
            ECDSA_HANDLER_CALLED.store(true, Ordering::SeqCst);
            // Verify we received the expected arguments
            assert_eq!(pk, &[0x02, 0xAA, 0xBB]); // pubkey
            assert_eq!(msg, &[0xCC, 0xDD]);       // message
            assert_eq!(sig, &[0xEE, 0xFF]);       // signature
            Ok(true)
        }

        let evaluator = VeilEvaluator::new(dummy_hasher, dummy_bls_verify, tracking_ecdsa_verify);

        // Program: (secp256k1_verify (q . <pk>) (q . <msg>) (q . <sig>))
        // secp256k1_verify = 4-byte opcode 0x13d61f00
        let program = vec![
            0xff,                               // (
            0x84, 0x13, 0xd6, 0x1f, 0x00,       // secp256k1_verify opcode (4 bytes, 0x84 prefix)
            0xff,                               //   (
            0xff, 0x01,                         //     (q .
            0x83, 0x02, 0xAA, 0xBB,             //       pk = 3 bytes)
            0xff,                               //     .
            0xff, 0x01,                         //     (q .
            0x82, 0xCC, 0xDD,                   //       msg = 2 bytes)
            0xff,                               //     .
            0xff, 0x01,                         //     (q .
            0x82, 0xEE, 0xFF,                   //       sig = 2 bytes)
            0x80,                               //     . nil
        ];
        let args = vec![0x80];

        let result = evaluator.run_program(&program, &args, 1000000);

        // Verify the handler was called
        assert!(ECDSA_HANDLER_CALLED.load(Ordering::SeqCst), "ECDSA handler was not called!");
        assert!(result.is_ok(), "ECDSA verify should succeed");
    }
}
