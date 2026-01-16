//! Injectable crypto operation handlers for zkVM compatibility
//!
//! This module provides a way to inject custom implementations of cryptographic
//! operations (SHA256, BLS, SECP256k1/r1) at runtime. This is essential for zkVM
//! environments where native crypto libraries aren't available, but precompiles are.
//!
//! # Example
//!
//! ```ignore
//! use clvmr::crypto_handlers::CryptoHandlers;
//! use clvmr::chia_dialect::ChiaDialect;
//!
//! // For host execution with native crypto
//! let dialect = ChiaDialect::new(0);
//!
//! // For zkVM execution with custom handlers
//! fn my_hasher(data: &[u8]) -> [u8; 32] { /* ... */ }
//! fn my_bls_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, &'static str> { /* ... */ }
//!
//! let handlers = CryptoHandlers::new()
//!     .with_sha256(my_hasher)
//!     .with_bls_verify(my_bls_verify);
//! let dialect = ChiaDialect::new_with_handlers(0, handlers);
//! ```

use crate::allocator::{Allocator, NodePtr};
use crate::cost::Cost;
use crate::reduction::Response;

/// Type alias for CLVM operator functions (used for complex BLS ops)
pub type OpHandler = fn(&mut Allocator, NodePtr, Cost) -> Response;

/// SHA256 hasher function type
pub type Sha256Fn = fn(&[u8]) -> [u8; 32];

/// Signature verifier function type (for BLS and ECDSA)
/// Arguments: (public_key, message, signature)
/// Returns: Ok(true) if valid, Ok(false) if invalid, Err on error
pub type VerifierFn = fn(&[u8], &[u8], &[u8]) -> Result<bool, &'static str>;

/// Container for injectable cryptographic operation handlers.
///
/// Each handler is optional - if None, the dialect will either use the
/// native implementation (if the corresponding feature is enabled) or
/// return an error.
#[derive(Clone, Default)]
pub struct CryptoHandlers {
    // SHA256 (opcode 11) - direct hasher function
    pub sha256: Option<Sha256Fn>,

    // BLS verify (opcode 59) - direct verifier function  
    pub bls_verify: Option<VerifierFn>,

    // SECP256k1 verify (4-byte opcode) - direct verifier function
    pub secp256k1_verify: Option<VerifierFn>,

    // SECP256r1 verify (4-byte opcode) - direct verifier function
    pub secp256r1_verify: Option<VerifierFn>,

    // Other BLS operations (opcodes 29-30, 49-58) - still use OpHandler
    // These are less commonly needed for basic zkVM use cases
    pub point_add: Option<OpHandler>,
    pub pubkey_for_exp: Option<OpHandler>,
    pub bls_g1_subtract: Option<OpHandler>,
    pub bls_g1_multiply: Option<OpHandler>,
    pub bls_g1_negate: Option<OpHandler>,
    pub bls_g2_add: Option<OpHandler>,
    pub bls_g2_subtract: Option<OpHandler>,
    pub bls_g2_multiply: Option<OpHandler>,
    pub bls_g2_negate: Option<OpHandler>,
    pub bls_map_to_g1: Option<OpHandler>,
    pub bls_map_to_g2: Option<OpHandler>,
    pub bls_pairing_identity: Option<OpHandler>,
}

impl CryptoHandlers {
    /// Create a new empty CryptoHandlers with no handlers set.
    pub fn new() -> Self {
        Self::default()
    }

    // Builder methods for the main crypto functions

    pub fn with_sha256(mut self, hasher: Sha256Fn) -> Self {
        self.sha256 = Some(hasher);
        self
    }

    pub fn with_bls_verify(mut self, verifier: VerifierFn) -> Self {
        self.bls_verify = Some(verifier);
        self
    }

    pub fn with_secp256k1_verify(mut self, verifier: VerifierFn) -> Self {
        self.secp256k1_verify = Some(verifier);
        self
    }

    pub fn with_secp256r1_verify(mut self, verifier: VerifierFn) -> Self {
        self.secp256r1_verify = Some(verifier);
        self
    }

    // Builder methods for other BLS operations (still use OpHandler)

    pub fn with_point_add(mut self, handler: OpHandler) -> Self {
        self.point_add = Some(handler);
        self
    }

    pub fn with_pubkey_for_exp(mut self, handler: OpHandler) -> Self {
        self.pubkey_for_exp = Some(handler);
        self
    }

    pub fn with_bls_g1_subtract(mut self, handler: OpHandler) -> Self {
        self.bls_g1_subtract = Some(handler);
        self
    }

    pub fn with_bls_g1_multiply(mut self, handler: OpHandler) -> Self {
        self.bls_g1_multiply = Some(handler);
        self
    }

    pub fn with_bls_g1_negate(mut self, handler: OpHandler) -> Self {
        self.bls_g1_negate = Some(handler);
        self
    }

    pub fn with_bls_g2_add(mut self, handler: OpHandler) -> Self {
        self.bls_g2_add = Some(handler);
        self
    }

    pub fn with_bls_g2_subtract(mut self, handler: OpHandler) -> Self {
        self.bls_g2_subtract = Some(handler);
        self
    }

    pub fn with_bls_g2_multiply(mut self, handler: OpHandler) -> Self {
        self.bls_g2_multiply = Some(handler);
        self
    }

    pub fn with_bls_g2_negate(mut self, handler: OpHandler) -> Self {
        self.bls_g2_negate = Some(handler);
        self
    }

    pub fn with_bls_map_to_g1(mut self, handler: OpHandler) -> Self {
        self.bls_map_to_g1 = Some(handler);
        self
    }

    pub fn with_bls_map_to_g2(mut self, handler: OpHandler) -> Self {
        self.bls_map_to_g2 = Some(handler);
        self
    }

    pub fn with_bls_pairing_identity(mut self, handler: OpHandler) -> Self {
        self.bls_pairing_identity = Some(handler);
        self
    }
}
