//! RISC Zero guest program for CLVM execution
//!
//! This guest receives serialized CLVM bytecode, deserializes it,
//! executes it using clvmr, and commits the result.

#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use risc0_zkvm::guest::env;

// Import from clvm_tools_rs
use clvm_tools_rs::{
    Allocator, ChiaDialect, node_from_bytes, node_to_bytes, run_program,
};

risc0_zkvm::guest::entry!(main);

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

fn main() {
    // Read input from host
    let input: ClvmInput = env::read();
    
    // Create allocator for CLVM execution
    let mut allocator = Allocator::new();
    
    // Deserialize program and args
    let program = node_from_bytes(&mut allocator, &input.program)
        .expect("Failed to deserialize program");
    let args = node_from_bytes(&mut allocator, &input.args)
        .expect("Failed to deserialize args");
    
    // Create dialect (no crypto handlers for now - stubbed)
    let dialect = ChiaDialect::new(0);
    
    // Run the program
    let reduction = run_program(
        &mut allocator,
        &dialect,
        program,
        args,
        input.max_cost,
    ).expect("CLVM execution failed");
    
    // Serialize the result
    let result_bytes = node_to_bytes(&allocator, reduction.1)
        .expect("Failed to serialize result");
    
    // Commit output
    let output = ClvmOutput {
        result: result_bytes,
        cost: reduction.0,
    };
    
    env::commit(&output);
}
