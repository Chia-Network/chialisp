//! RISC Zero guest program for CLVM execution
//!
//! This guest uses VeilEvaluator with RISC0-accelerated crypto,
//! demonstrating compatibility with Veil's interface.

#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use risc0_zkvm::guest::env;
use risc0_zkvm::sha::{Impl, Sha256 as RiscSha256};

// Import VeilEvaluator from clvm_tools_rs
use clvm_tools_rs::VeilEvaluator;

// BLS imports for signature verification
use bls12_381::hash_to_curve::{ExpandMsgXmd, HashToCurve};
use bls12_381::{pairing, G1Affine, G1Projective, G2Affine};
use sha2::Sha256;

risc0_zkvm::guest::entry!(main);

/// BLS domain separation tag (matches Veil)
const BLS_DST: &[u8] = b"CLVM_ZK_BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_";

/// Input to the CLVM guest program
#[derive(serde::Deserialize, borsh::BorshDeserialize)]
pub struct ClvmInput {
    /// Serialized CLVM program bytecode
    pub program: Vec<u8>,
    /// Serialized CLVM environment/arguments
    pub args: Vec<u8>,
    /// Maximum cost allowed
    pub max_cost: u64,
}

/// Output from the CLVM guest program  
#[derive(serde::Serialize, borsh::BorshSerialize)]
pub struct ClvmOutput {
    /// Serialized result
    pub result: Vec<u8>,
    /// Cost consumed
    pub cost: u64,
}

// RISC0-accelerated SHA256 hasher
fn risc0_hasher(data: &[u8]) -> [u8; 32] {
    let digest = Impl::hash_bytes(data);
    digest
        .as_bytes()
        .try_into()
        .expect("SHA-256 digest should be 32 bytes")
}

// RISC0-accelerated BLS verification
fn risc0_verify_bls(
    public_key_bytes: &[u8],
    message_bytes: &[u8],
    signature_bytes: &[u8],
) -> Result<bool, &'static str> {
    // Pad public key to 96 bytes (G2 compressed)
    let mut pk_padded = [0u8; 96];
    if public_key_bytes.len() > 96 {
        return Err("invalid public key size - too large");
    }
    let pk_start = 96 - public_key_bytes.len();
    pk_padded[pk_start..].copy_from_slice(public_key_bytes);

    // Pad signature to 48 bytes (G1 compressed)
    let mut sig_padded = [0u8; 48];
    if signature_bytes.len() > 48 {
        return Err("invalid signature size - too large");
    }
    let sig_start = 48 - signature_bytes.len();
    sig_padded[sig_start..].copy_from_slice(signature_bytes);

    // Parse public key
    let public_key = G2Affine::from_compressed(&pk_padded);
    let public_key = if public_key.is_some().into() {
        public_key.unwrap()
    } else {
        return Err("invalid BLS public key format");
    };

    // Parse signature
    let signature = G1Affine::from_compressed(&sig_padded);
    let signature = if signature.is_some().into() {
        signature.unwrap()
    } else {
        return Err("invalid BLS signature format");
    };

    // Hash message to curve
    let message_parts = [message_bytes];
    let message_point = <G1Projective as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(
        message_parts.iter().copied(),
        BLS_DST,
    );

    // Verify: e(sig, g2) == e(H(msg), pk)
    let g2_generator = G2Affine::generator();
    let lhs = pairing(&signature, &g2_generator);
    let rhs = pairing(&message_point.into(), &public_key);

    Ok(lhs == rhs)
}

// ECDSA verification using RISC0 hasher
fn risc0_verify_ecdsa(
    public_key_bytes: &[u8],
    message_bytes: &[u8],
    signature_bytes: &[u8],
) -> Result<bool, &'static str> {
    use k256::ecdsa::{signature::Verifier, Signature, VerifyingKey};

    // Parse public key
    if public_key_bytes.len() != 33 && public_key_bytes.len() != 65 {
        return Err("invalid public key size");
    }
    let verifying_key = VerifyingKey::from_sec1_bytes(public_key_bytes)
        .map_err(|_| "invalid public key format")?;

    // Parse signature (64 bytes compact format)
    let signature = if signature_bytes.len() == 64 {
        Signature::try_from(signature_bytes).map_err(|_| "invalid signature format")?
    } else if signature_bytes.len() < 64 {
        let mut padded = [0u8; 64];
        padded[..signature_bytes.len()].copy_from_slice(signature_bytes);
        Signature::try_from(padded.as_slice()).map_err(|_| "invalid signature format")?
    } else {
        return Err("signature too long");
    };

    // Hash message if not already 32 bytes
    let message_hash = if message_bytes.len() == 32 {
        message_bytes.to_vec()
    } else {
        risc0_hasher(message_bytes).to_vec()
    };

    // Verify
    match verifying_key.verify(&message_hash, &signature) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

fn main() {
    // Read input from host
    let input: ClvmInput = env::read();
    
    // Create VeilEvaluator with RISC0 crypto
    let evaluator = VeilEvaluator::new(
        risc0_hasher,
        risc0_verify_bls,
        risc0_verify_ecdsa,
    );
    
    // Run the program using VeilEvaluator (backed by clvmr)
    let (result_bytes, cost) = evaluator
        .run_program(&input.program, &input.args, input.max_cost)
        .expect("CLVM execution failed");
    
    // Commit output
    let output = ClvmOutput {
        result: result_bytes,
        cost,
    };
    
    env::commit(&output);
}
