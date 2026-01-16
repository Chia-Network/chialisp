//! RISC Zero host for CLVM execution testing
//!
//! This crate provides the host-side implementation for running CLVM
//! programs inside the RISC Zero zkVM.

pub mod methods;

use risc0_zkvm::{default_prover, ExecutorEnv};
use serde::{Deserialize, Serialize};

pub use methods::CLVM_RISC0_GUEST_ELF;
pub use methods::CLVM_RISC0_GUEST_ID;

/// Input to the CLVM guest program
#[derive(Clone, Serialize, Deserialize, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct ClvmInput {
    /// Serialized CLVM program bytecode
    pub program: Vec<u8>,
    /// Serialized CLVM environment/arguments
    pub args: Vec<u8>,
    /// Maximum cost allowed
    pub max_cost: u64,
}

/// Output from the CLVM guest program
#[derive(Clone, Serialize, Deserialize, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct ClvmOutput {
    /// Serialized result
    pub result: Vec<u8>,
    /// Cost consumed
    pub cost: u64,
}

/// Run a CLVM program in the RISC Zero zkVM and return the proof
pub fn prove_clvm_execution(input: ClvmInput) -> Result<(ClvmOutput, Vec<u8>), String> {
    // Build executor environment with input
    let env = ExecutorEnv::builder()
        .write(&input)
        .map_err(|e| format!("Failed to write input: {}", e))?
        .build()
        .map_err(|e| format!("Failed to build env: {}", e))?;

    // Get the prover
    let prover = default_prover();

    // Generate the proof (local proving, no GPU needed)
    let prove_info = prover
        .prove(env, CLVM_RISC0_GUEST_ELF)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    // Extract the receipt
    let receipt = prove_info.receipt;

    // Decode the output from the journal
    let output: ClvmOutput = receipt
        .journal
        .decode()
        .map_err(|e| format!("Failed to decode output: {}", e))?;

    // Serialize the receipt as proof bytes
    let proof_bytes = borsh::to_vec(&receipt)
        .map_err(|e| format!("Failed to serialize proof: {}", e))?;

    Ok((output, proof_bytes))
}

/// Run a CLVM program and generate a real proof (local, no GPU)
pub fn run_clvm_with_proof(input: ClvmInput) -> Result<(ClvmOutput, risc0_zkvm::Receipt), String> {
    // Build executor environment with input
    let env = ExecutorEnv::builder()
        .write(&input)
        .map_err(|e| format!("Failed to write input: {}", e))?
        .build()
        .map_err(|e| format!("Failed to build env: {}", e))?;

    // Get the prover
    let prover = default_prover();

    // Generate real proof (local proving)
    let prove_info = prover
        .prove(env, CLVM_RISC0_GUEST_ELF)
        .map_err(|e| format!("Proving failed: {}", e))?;

    let receipt = prove_info.receipt;

    // Decode the output
    let output: ClvmOutput = receipt
        .journal
        .decode()
        .map_err(|e| format!("Failed to decode output: {}", e))?;

    Ok((output, receipt))
}

/// Verify a CLVM execution proof
pub fn verify_clvm_proof(receipt: &risc0_zkvm::Receipt) -> Result<ClvmOutput, String> {
    // Verify the receipt against our guest ID
    receipt
        .verify(CLVM_RISC0_GUEST_ID)
        .map_err(|e| format!("Verification failed: {}", e))?;

    // Decode and return the output
    receipt
        .journal
        .decode()
        .map_err(|e| format!("Failed to decode output: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nil_program_with_proof() {
        // nil program returns nil
        let input = ClvmInput {
            program: vec![0x80], // nil
            args: vec![0x80],    // nil  
            max_cost: 1000000,
        };

        let (output, receipt) = run_clvm_with_proof(input).expect("CLVM proving failed");
        
        // Verify the proof
        let verified_output = verify_clvm_proof(&receipt).expect("Verification failed");
        
        assert_eq!(output.result, vec![0x80]);
        assert_eq!(verified_output.result, output.result);
        println!("nil test PROVED and VERIFIED! Cost: {}", output.cost);
    }

    #[test]
    fn test_quote_42_with_proof() {
        // Program: (q . 42) - quote returns 42
        let input = ClvmInput {
            program: vec![0xff, 0x01, 0x2a], // (q . 42)
            args: vec![0x80],                 // nil args
            max_cost: 1000000,
        };

        let (output, receipt) = run_clvm_with_proof(input).expect("CLVM proving failed");
        
        // Verify the proof
        let verified_output = verify_clvm_proof(&receipt).expect("Verification failed");
        
        assert_eq!(output.result, vec![0x2a]);
        assert_eq!(verified_output.result, output.result);
        println!("quote 42 PROVED and VERIFIED! Result: {:?}, Cost: {}", output.result, output.cost);
    }

    #[test]
    fn test_addition_with_proof() {
        // Program: (+ (q . 2) (q . 3)) = 5
        let input = ClvmInput {
            program: vec![
                0xff, 0x10,       // (+ . 
                0xff,             //   (
                0xff, 0x01, 0x02, //     (q . 2)
                0xff,             //     .
                0xff, 0x01, 0x03, //     (q . 3)
                0x80,             //     . nil
            ],
            args: vec![0x80],     // nil args
            max_cost: 1000000,
        };

        let (output, receipt) = run_clvm_with_proof(input).expect("CLVM proving failed");
        
        // Verify the proof
        let verified_output = verify_clvm_proof(&receipt).expect("Verification failed");
        
        assert_eq!(output.result, vec![0x05]);
        assert_eq!(verified_output.result, output.result);
        println!("Addition PROVED and VERIFIED! 2 + 3 = {:?}, Cost: {}", output.result, output.cost);
    }

    #[test]
    fn test_sha256_with_proof() {
        // Program: (sha256 (q . "test"))
        // sha256 = opcode 11 (0x0b)
        // "test" = 0x74657374
        let input = ClvmInput {
            program: vec![
                0xff, 0x0b,                   // (sha256
                0xff,                         //   (
                0xff, 0x01,                   //     (q .
                0x84, 0x74, 0x65, 0x73, 0x74, //       "test" = 4 bytes)
                0x80,                         //   )
            ],
            args: vec![0x80],
            max_cost: 1000000,
        };

        let (output, receipt) = run_clvm_with_proof(input).expect("CLVM proving failed");
        
        // Verify the proof
        let verified_output = verify_clvm_proof(&receipt).expect("Verification failed");
        
        // SHA256("test") = 0x9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
        // Result is 32 bytes + 1 byte length prefix (0xa0 = 32 in CLVM serialization)
        assert_eq!(output.result.len(), 33);
        assert_eq!(output.result[0], 0xa0); // 32-byte atom prefix
        assert_eq!(verified_output.result, output.result);
        
        // Verify it's the correct SHA256 hash
        let expected_hash = hex::decode("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08").unwrap();
        assert_eq!(&output.result[1..], expected_hash.as_slice());
        
        println!("SHA256 PROVED and VERIFIED! Hash: {:?}, Cost: {}", hex::encode(&output.result[1..]), output.cost);
    }
}
