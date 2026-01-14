//! RISC Zero host for CLVM execution testing
//!
//! This crate provides the host-side implementation for running CLVM
//! programs inside the RISC Zero zkVM.

pub mod methods;

use risc0_zkvm::{default_prover, ExecutorEnv, ProverOpts, VerifierContext};
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

    // Generate the proof
    let prove_info = prover
        .prove_with_ctx(
            env,
            &VerifierContext::default(),
            CLVM_RISC0_GUEST_ELF,
            &ProverOpts::groth16(),
        )
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

/// Run a CLVM program in dev mode (fast, no real proof)
pub fn run_clvm_dev(input: ClvmInput) -> Result<ClvmOutput, String> {
    // Build executor environment with input
    let env = ExecutorEnv::builder()
        .write(&input)
        .map_err(|e| format!("Failed to write input: {}", e))?
        .build()
        .map_err(|e| format!("Failed to build env: {}", e))?;

    // Get the prover
    let prover = default_prover();

    // Run in dev mode (fast execution, fake proof)
    let prove_info = prover
        .prove_with_ctx(
            env,
            &VerifierContext::default(),
            CLVM_RISC0_GUEST_ELF,
            &ProverOpts::fast(),
        )
        .map_err(|e| format!("Execution failed: {}", e))?;

    // Decode the output
    let output: ClvmOutput = prove_info
        .receipt
        .journal
        .decode()
        .map_err(|e| format!("Failed to decode output: {}", e))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nil_program() {
        // nil program returns nil
        // Program: 0x80 (nil)
        // Args: 0x80 (nil)
        
        let input = ClvmInput {
            program: vec![0x80], // nil
            args: vec![0x80],    // nil  
            max_cost: 1000000,
        };

        let output = run_clvm_dev(input).expect("CLVM execution failed");
        
        // nil returns nil
        assert_eq!(output.result, vec![0x80]);
        println!("nil test passed! Cost: {}", output.cost);
    }

    #[test]
    fn test_quote_42() {
        // Program: (q . 42) - quote returns 42
        // Serialized as: ff 01 2a
        // 0xff = cons marker
        // 0x01 = quote opcode (1)
        // 0x2a = 42
        
        let input = ClvmInput {
            program: vec![0xff, 0x01, 0x2a], // (q . 42)
            args: vec![0x80],                 // nil args
            max_cost: 1000000,
        };

        let output = run_clvm_dev(input).expect("CLVM execution failed");
        
        // Should return 42
        assert_eq!(output.result, vec![0x2a]);
        println!("quote 42 test passed! Result: {:?}, Cost: {}", output.result, output.cost);
    }

    #[test]
    fn test_addition() {
        // Program: (+ (q . 2) (q . 3)) = 5
        // Opcode for + is 16
        // Serialized as nested cons:
        // (16 . ((q . 2) . ((q . 3) . nil)))
        // = ff 10 ff ff 01 02 ff ff 01 03 80
        
        let input = ClvmInput {
            program: vec![
                0xff, 0x10,       // (+ . 
                0xff,             //   (
                0xff, 0x01, 0x02, //     (q . 2)
                0xff,             //     .
                0xff, 0x01, 0x03, //     (q . 3)
                0x80,             //     . nil
                                  //   )
                                  // )
            ],
            args: vec![0x80],     // nil args
            max_cost: 1000000,
        };

        let output = run_clvm_dev(input).expect("CLVM execution failed");
        
        // Should return 5
        assert_eq!(output.result, vec![0x05]);
        println!("Addition test passed! 2 + 3 = {:?}, Cost: {}", output.result, output.cost);
    }
}
