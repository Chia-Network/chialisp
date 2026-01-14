use crate::allocator::{Allocator, NodePtr};
use crate::core_ops::{op_cons, op_eq, op_first, op_if, op_listp, op_raise, op_rest};
use crate::cost::Cost;
use crate::crypto_handlers::CryptoHandlers;
use crate::dialect::{Dialect, OperatorSet};
use crate::error::EvalErr;
use crate::more_ops::{
    op_add, op_all, op_any, op_ash, op_coinid, op_concat, op_div, op_divmod, op_gr, op_gr_bytes,
    op_logand, op_logior, op_lognot, op_logxor, op_lsh, op_mod, op_modpow, op_multiply, op_not,
    op_point_add, op_pubkey_for_exp, op_sha256, op_strlen, op_substr, op_subtract, op_unknown,
};
use crate::reduction::Response;

#[cfg(feature = "bls-ops")]
use crate::bls_ops::{
    op_bls_g1_multiply, op_bls_g1_negate, op_bls_g1_subtract, op_bls_g2_add, op_bls_g2_multiply,
    op_bls_g2_negate, op_bls_g2_subtract, op_bls_map_to_g1, op_bls_map_to_g2,
    op_bls_pairing_identity, op_bls_verify,
};

#[cfg(feature = "secp-ops")]
use crate::secp_ops::{op_secp256k1_verify, op_secp256r1_verify};

#[cfg(feature = "keccak-ops")]
use crate::keccak256_ops::op_keccak256;

// unknown operators are disallowed
// (otherwise they are no-ops with well defined cost)
pub const NO_UNKNOWN_OPS: u32 = 0x0002;

// When set, limits the number of atom-bytes allowed to be allocated, as well as
// the number of pairs
pub const LIMIT_HEAP: u32 = 0x0004;

// enables the keccak256 op *outside* the softfork guard.
// This is a hard-fork and should only be enabled when it activates
pub const ENABLE_KECCAK_OPS_OUTSIDE_GUARD: u32 = 0x0100;

pub const DISABLE_OP: u32 = 0x200;

// The default mode when running grnerators in mempool-mode (i.e. the stricter
// mode)
pub const MEMPOOL_MODE: u32 = NO_UNKNOWN_OPS | LIMIT_HEAP | DISABLE_OP;

fn unknown_operator(
    allocator: &mut Allocator,
    o: NodePtr,
    args: NodePtr,
    flags: u32,
    max_cost: Cost,
) -> Response {
    if (flags & NO_UNKNOWN_OPS) != 0 {
        Err(EvalErr::Unimplemented(o))?
    } else {
        op_unknown(allocator, o, args, max_cost)
    }
}

/// The Chia dialect for CLVM execution.
/// 
/// Supports injectable crypto handlers for zkVM compatibility.
/// When handlers are provided, they take precedence over native implementations.
pub struct ChiaDialect {
    flags: u32,
    /// Optional crypto handlers for injectable BLS/SECP operations
    crypto_handlers: Option<CryptoHandlers>,
}

impl ChiaDialect {
    /// Create a new ChiaDialect with default (native) crypto implementations.
    pub fn new(flags: u32) -> ChiaDialect {
        ChiaDialect { 
            flags,
            crypto_handlers: None,
        }
    }

    /// Create a new ChiaDialect with custom crypto handlers.
    /// 
    /// This is useful for zkVM environments where native crypto libraries
    /// aren't available, but precompiles are.
    /// 
    /// # Example
    /// 
    /// ```ignore
    /// use clvmr::{ChiaDialect, CryptoHandlers};
    /// 
    /// let handlers = CryptoHandlers::new()
    ///     .with_bls_verify(my_bls_precompile)
    ///     .with_secp256k1_verify(my_secp_precompile);
    /// 
    /// let dialect = ChiaDialect::new_with_handlers(0, handlers);
    /// ```
    pub fn new_with_handlers(flags: u32, handlers: CryptoHandlers) -> ChiaDialect {
        ChiaDialect {
            flags,
            crypto_handlers: Some(handlers),
        }
    }

    /// Get the crypto handlers, if any.
    pub fn crypto_handlers(&self) -> Option<&CryptoHandlers> {
        self.crypto_handlers.as_ref()
    }

}

impl Dialect for ChiaDialect {
    fn op(
        &self,
        allocator: &mut Allocator,
        o: NodePtr,
        argument_list: NodePtr,
        max_cost: Cost,
        extension: OperatorSet,
    ) -> Response {
        let flags = self.flags
            | match extension {
                OperatorSet::Default => 0,
                OperatorSet::Bls => 0,
                OperatorSet::Keccak => ENABLE_KECCAK_OPS_OUTSIDE_GUARD,
            };

        let op_len = allocator.atom_len(o);
        
        // Handle 4-byte opcodes (SECP operations)
        if op_len == 4 {
            let b = allocator.atom(o);
            let opcode = u32::from_be_bytes(b.as_ref().try_into().unwrap());

            // Check for injected SECP handlers first
            if let Some(ref handlers) = self.crypto_handlers {
                match opcode {
                    0x13d61f00 => {
                        if let Some(handler) = handlers.secp256k1_verify {
                            return handler(allocator, argument_list, max_cost);
                        }
                    }
                    0x1c3a8f00 => {
                        if let Some(handler) = handlers.secp256r1_verify {
                            return handler(allocator, argument_list, max_cost);
                        }
                    }
                    _ => {}
                }
            }

            // Fall back to native SECP if feature enabled
            #[cfg(feature = "secp-ops")]
            {
                let f = match opcode {
                    0x13d61f00 => op_secp256k1_verify,
                    0x1c3a8f00 => op_secp256r1_verify,
                    _ => {
                        return unknown_operator(allocator, o, argument_list, flags, max_cost);
                    }
                };
                return f(allocator, argument_list, max_cost);
            }

            #[cfg(not(feature = "secp-ops"))]
            {
                return unknown_operator(allocator, o, argument_list, flags, max_cost);
            }
        }
        
        if op_len != 1 {
            return unknown_operator(allocator, o, argument_list, flags, max_cost);
        }
        
        let Some(op) = allocator.small_number(o) else {
            return unknown_operator(allocator, o, argument_list, flags, max_cost);
        };
        
        // Core operators (always available)
        match op {
            // 1 = quote
            // 2 = apply
            3 => return op_if(allocator, argument_list, max_cost),
            4 => return op_cons(allocator, argument_list, max_cost),
            5 => return op_first(allocator, argument_list, max_cost),
            6 => return op_rest(allocator, argument_list, max_cost),
            7 => return op_listp(allocator, argument_list, max_cost),
            8 => return op_raise(allocator, argument_list, max_cost),
            9 => return op_eq(allocator, argument_list, max_cost),
            10 => return op_gr_bytes(allocator, argument_list, max_cost),
            11 => return op_sha256(allocator, argument_list, max_cost),
            12 => return op_substr(allocator, argument_list, max_cost),
            13 => return op_strlen(allocator, argument_list, max_cost),
            14 => return op_concat(allocator, argument_list, max_cost),
            // 15 ---
            16 => return op_add(allocator, argument_list, max_cost),
            17 => return op_subtract(allocator, argument_list, max_cost),
            18 => return op_multiply(allocator, argument_list, max_cost),
            19 => return op_div(allocator, argument_list, max_cost),
            20 => return op_divmod(allocator, argument_list, max_cost),
            21 => return op_gr(allocator, argument_list, max_cost),
            22 => return op_ash(allocator, argument_list, max_cost),
            23 => return op_lsh(allocator, argument_list, max_cost),
            24 => return op_logand(allocator, argument_list, max_cost),
            25 => return op_logior(allocator, argument_list, max_cost),
            26 => return op_logxor(allocator, argument_list, max_cost),
            27 => return op_lognot(allocator, argument_list, max_cost),
            // 28 ---
            
            // point_add (29) - check for injected handler
            29 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.point_add {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                return op_point_add(allocator, argument_list, max_cost);
            }
            
            // pubkey_for_exp (30) - check for injected handler
            30 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.pubkey_for_exp {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                return op_pubkey_for_exp(allocator, argument_list, max_cost);
            }
            
            // 31 ---
            32 => return op_not(allocator, argument_list, max_cost),
            33 => return op_any(allocator, argument_list, max_cost),
            34 => return op_all(allocator, argument_list, max_cost),
            // 35 ---
            // 36 = softfork
            48 => return op_coinid(allocator, argument_list, max_cost),
            
            // BLS operators (49-59) - check for injected handlers first
            49 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g1_subtract {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g1_subtract(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            50 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g1_multiply {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g1_multiply(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            51 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g1_negate {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g1_negate(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            52 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g2_add {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g2_add(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            53 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g2_subtract {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g2_subtract(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            54 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g2_multiply {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g2_multiply(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            55 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_g2_negate {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_g2_negate(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            56 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_map_to_g1 {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_map_to_g1(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            57 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_map_to_g2 {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_map_to_g2(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            58 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_pairing_identity {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_pairing_identity(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            59 => {
                if let Some(ref handlers) = self.crypto_handlers {
                    if let Some(handler) = handlers.bls_verify {
                        return handler(allocator, argument_list, max_cost);
                    }
                }
                #[cfg(feature = "bls-ops")]
                return op_bls_verify(allocator, argument_list, max_cost);
                #[cfg(not(feature = "bls-ops"))]
                return Err(EvalErr::Unimplemented(o));
            }
            
            60 => {
                if (flags & DISABLE_OP) != 0 {
                    return Err(EvalErr::Unimplemented(o));
                } else {
                    return op_modpow(allocator, argument_list, max_cost);
                }
            }
            61 => return op_mod(allocator, argument_list, max_cost),
            
            // Keccak256 (62)
            #[cfg(feature = "keccak-ops")]
            62 if (flags & ENABLE_KECCAK_OPS_OUTSIDE_GUARD) != 0 => {
                return op_keccak256(allocator, argument_list, max_cost);
            }
            
            _ => {}
        }
        
        unknown_operator(allocator, o, argument_list, flags, max_cost)
    }

    fn quote_kw(&self) -> u32 {
        1
    }
    fn apply_kw(&self) -> u32 {
        2
    }
    fn softfork_kw(&self) -> u32 {
        36
    }

    fn softfork_extension(&self, ext: u32) -> OperatorSet {
        match ext {
            0 => OperatorSet::Bls,
            1 => OperatorSet::Keccak,
            _ => OperatorSet::Default,
        }
    }

    fn allow_unknown_ops(&self) -> bool {
        (self.flags & NO_UNKNOWN_OPS) == 0
    }
}
