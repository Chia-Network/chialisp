//! Injectable crypto operation handlers for zkVM compatibility
//!
//! This module provides a way to inject custom implementations of cryptographic
//! operations (BLS, SECP256k1/r1) at runtime. This is essential for zkVM environments
//! where native crypto libraries aren't available, but precompiles are.
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
//! let handlers = CryptoHandlers::new()
//!     .with_bls_verify(my_bls_precompile)
//!     .with_secp256k1_verify(my_secp_precompile);
//! let dialect = ChiaDialect::new_with_handlers(0, handlers);
//! ```

use crate::allocator::{Allocator, NodePtr};
use crate::cost::Cost;
use crate::reduction::Response;

/// Type alias for CLVM operator functions
pub type OpHandler = fn(&mut Allocator, NodePtr, Cost) -> Response;

/// Container for injectable cryptographic operation handlers.
///
/// Each handler is optional - if None, the dialect will either use the
/// native implementation (if the corresponding feature is enabled) or
/// return an error.
#[derive(Clone, Default)]
pub struct CryptoHandlers {
    // BLS operations (opcodes 29-30, 49-59)
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
    pub bls_verify: Option<OpHandler>,

    // SECP operations (4-byte opcodes)
    pub secp256k1_verify: Option<OpHandler>,
    pub secp256r1_verify: Option<OpHandler>,
}

impl CryptoHandlers {
    /// Create a new empty CryptoHandlers with no handlers set.
    pub fn new() -> Self {
        Self::default()
    }

    // Builder methods for setting handlers

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

    pub fn with_bls_verify(mut self, handler: OpHandler) -> Self {
        self.bls_verify = Some(handler);
        self
    }

    pub fn with_secp256k1_verify(mut self, handler: OpHandler) -> Self {
        self.secp256k1_verify = Some(handler);
        self
    }

    pub fn with_secp256r1_verify(mut self, handler: OpHandler) -> Self {
        self.secp256r1_verify = Some(handler);
        self
    }

    /// Set all BLS handlers at once
    pub fn with_all_bls(
        mut self,
        point_add: OpHandler,
        pubkey_for_exp: OpHandler,
        g1_subtract: OpHandler,
        g1_multiply: OpHandler,
        g1_negate: OpHandler,
        g2_add: OpHandler,
        g2_subtract: OpHandler,
        g2_multiply: OpHandler,
        g2_negate: OpHandler,
        map_to_g1: OpHandler,
        map_to_g2: OpHandler,
        pairing_identity: OpHandler,
        bls_verify: OpHandler,
    ) -> Self {
        self.point_add = Some(point_add);
        self.pubkey_for_exp = Some(pubkey_for_exp);
        self.bls_g1_subtract = Some(g1_subtract);
        self.bls_g1_multiply = Some(g1_multiply);
        self.bls_g1_negate = Some(g1_negate);
        self.bls_g2_add = Some(g2_add);
        self.bls_g2_subtract = Some(g2_subtract);
        self.bls_g2_multiply = Some(g2_multiply);
        self.bls_g2_negate = Some(g2_negate);
        self.bls_map_to_g1 = Some(map_to_g1);
        self.bls_map_to_g2 = Some(map_to_g2);
        self.bls_pairing_identity = Some(pairing_identity);
        self.bls_verify = Some(bls_verify);
        self
    }

    /// Set all SECP handlers at once
    pub fn with_all_secp(
        mut self,
        secp256k1_verify: OpHandler,
        secp256r1_verify: OpHandler,
    ) -> Self {
        self.secp256k1_verify = Some(secp256k1_verify);
        self.secp256r1_verify = Some(secp256r1_verify);
        self
    }
}
