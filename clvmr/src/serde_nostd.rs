//! no_std compatible CLVM serialization/deserialization
//!
//! This module provides serialization and deserialization for CLVM programs
//! without requiring std::io. It operates directly on byte slices.

use alloc::vec;
use alloc::vec::Vec;

use crate::allocator::{Allocator, NodePtr, NodeVisitor, len_for_value};
use crate::error::{EvalErr, Result};

const CONS_BOX_MARKER: u8 = 0xff;

/// A simple cursor for reading bytes without std::io
struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(EvalErr::SerializationError);
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        Ok(byte)
    }

    fn read_bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.pos + count > self.data.len() {
            return Err(EvalErr::SerializationError);
        }
        let bytes = &self.data[self.pos..self.pos + count];
        self.pos += count;
        Ok(bytes)
    }

    #[allow(dead_code)]
    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
}

/// Parse an atom from the byte stream (no_std version)
fn parse_atom(allocator: &mut Allocator, first_byte: u8, reader: &mut ByteReader) -> Result<NodePtr> {
    if first_byte == 0x80 {
        // Special case for nil (empty atom)
        return allocator.new_atom(&[]);
    }
    
    if first_byte < 0x80 {
        // Single byte atom (value < 128)
        return allocator.new_small_number(first_byte as u32);
    }
    
    // Multi-byte atom - decode size
    let (_size_bytes, atom_size) = decode_size(first_byte, reader)?;
    
    if atom_size == 0 {
        return allocator.new_atom(&[]);
    }
    
    // Read the atom data
    let atom_data = reader.read_bytes(atom_size)?;
    allocator.new_atom(atom_data)
}

/// Decode size from atom header (no_std version)
fn decode_size(first_byte: u8, reader: &mut ByteReader) -> Result<(usize, usize)> {
    // Count leading 1 bits to determine size encoding
    if (first_byte & 0xC0) == 0x80 {
        // 0b10xxxxxx - 1 byte size, up to 63 bytes
        let size = (first_byte & 0x3F) as usize;
        return Ok((1, size));
    }
    
    if (first_byte & 0xE0) == 0xC0 {
        // 0b110xxxxx - 2 byte size
        let b2 = reader.read_byte()?;
        let size = (((first_byte & 0x1F) as usize) << 8) | (b2 as usize);
        return Ok((2, size));
    }
    
    if (first_byte & 0xF0) == 0xE0 {
        // 0b1110xxxx - 3 byte size
        let b2 = reader.read_byte()?;
        let b3 = reader.read_byte()?;
        let size = (((first_byte & 0x0F) as usize) << 16) 
                 | ((b2 as usize) << 8) 
                 | (b3 as usize);
        return Ok((3, size));
    }
    
    if (first_byte & 0xF8) == 0xF0 {
        // 0b11110xxx - 4 byte size
        let b2 = reader.read_byte()?;
        let b3 = reader.read_byte()?;
        let b4 = reader.read_byte()?;
        let size = (((first_byte & 0x07) as usize) << 24)
                 | ((b2 as usize) << 16)
                 | ((b3 as usize) << 8)
                 | (b4 as usize);
        return Ok((4, size));
    }
    
    if (first_byte & 0xFC) == 0xF8 {
        // 0b111110xx - 5 byte size
        let b2 = reader.read_byte()?;
        let b3 = reader.read_byte()?;
        let b4 = reader.read_byte()?;
        let b5 = reader.read_byte()?;
        let size = (((first_byte & 0x03) as usize) << 32)
                 | ((b2 as usize) << 24)
                 | ((b3 as usize) << 16)
                 | ((b4 as usize) << 8)
                 | (b5 as usize);
        return Ok((5, size));
    }
    
    Err(EvalErr::SerializationError)
}

#[repr(u8)]
enum ParseOp {
    SExp,
    Cons,
}

/// Deserialize a CLVM node from bytes (no_std compatible)
/// 
/// This is the main entry point for deserializing CLVM programs in no_std environments.
/// 
/// # Example
/// 
/// ```ignore
/// use clvmr::{Allocator, node_from_bytes_nostd};
/// 
/// let mut allocator = Allocator::new();
/// let program_bytes = &[0x80]; // nil
/// let node = node_from_bytes_nostd(&mut allocator, program_bytes)?;
/// ```
pub fn node_from_bytes(allocator: &mut Allocator, bytes: &[u8]) -> Result<NodePtr> {
    let mut reader = ByteReader::new(bytes);
    let mut values: Vec<NodePtr> = Vec::new();
    let mut ops = vec![ParseOp::SExp];
    
    while let Some(op) = ops.pop() {
        match op {
            ParseOp::SExp => {
                let first_byte = reader.read_byte()?;
                if first_byte == CONS_BOX_MARKER {
                    ops.push(ParseOp::Cons);
                    ops.push(ParseOp::SExp);
                    ops.push(ParseOp::SExp);
                } else {
                    values.push(parse_atom(allocator, first_byte, &mut reader)?);
                }
            }
            ParseOp::Cons => {
                let v2 = values.pop().ok_or(EvalErr::SerializationError)?;
                let v1 = values.pop().ok_or(EvalErr::SerializationError)?;
                values.push(allocator.new_pair(v1, v2)?);
            }
        }
    }
    
    values.pop().ok_or(EvalErr::SerializationError)
}

/// Write an atom to the output buffer (no_std version)
fn write_atom(output: &mut Vec<u8>, atom: &[u8]) -> Result<()> {
    let len = atom.len();
    
    if len == 0 {
        // Nil (empty atom)
        output.push(0x80);
    } else if len == 1 && atom[0] < 0x80 {
        // Single byte atom (value < 128)
        output.push(atom[0]);
    } else if len < 0x40 {
        // 1-byte size prefix (up to 63 bytes)
        output.push(0x80 | (len as u8));
        output.extend_from_slice(atom);
    } else if len < 0x2000 {
        // 2-byte size prefix
        output.push(0xC0 | ((len >> 8) as u8));
        output.push((len & 0xFF) as u8);
        output.extend_from_slice(atom);
    } else if len < 0x100000 {
        // 3-byte size prefix
        output.push(0xE0 | ((len >> 16) as u8));
        output.push(((len >> 8) & 0xFF) as u8);
        output.push((len & 0xFF) as u8);
        output.extend_from_slice(atom);
    } else if len < 0x8000000 {
        // 4-byte size prefix
        output.push(0xF0 | ((len >> 24) as u8));
        output.push(((len >> 16) & 0xFF) as u8);
        output.push(((len >> 8) & 0xFF) as u8);
        output.push((len & 0xFF) as u8);
        output.extend_from_slice(atom);
    } else {
        // 5-byte size prefix (very large atoms)
        output.push(0xF8 | ((len >> 32) as u8));
        output.push(((len >> 24) & 0xFF) as u8);
        output.push(((len >> 16) & 0xFF) as u8);
        output.push(((len >> 8) & 0xFF) as u8);
        output.push((len & 0xFF) as u8);
        output.extend_from_slice(atom);
    }
    
    Ok(())
}

/// Serialize a CLVM node to bytes (no_std compatible)
/// 
/// This is the main entry point for serializing CLVM programs in no_std environments.
/// 
/// # Example
/// 
/// ```ignore
/// use clvmr::{Allocator, node_to_bytes_nostd};
/// 
/// let mut allocator = Allocator::new();
/// let node = allocator.nil();
/// let bytes = node_to_bytes_nostd(&allocator, node)?;
/// assert_eq!(bytes, vec![0x80]);
/// ```
pub fn node_to_bytes(allocator: &Allocator, node: NodePtr) -> Result<Vec<u8>> {
    node_to_bytes_limit(allocator, node, 2_000_000)
}

/// Serialize a CLVM node to bytes with a size limit (no_std compatible)
pub fn node_to_bytes_limit(allocator: &Allocator, node: NodePtr, limit: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut values: Vec<NodePtr> = vec![node];
    
    while let Some(v) = values.pop() {
        if output.len() > limit {
            return Err(EvalErr::OutOfMemory);
        }
        
        match allocator.node(v) {
            NodeVisitor::Buffer(buf) => write_atom(&mut output, buf)?,
            NodeVisitor::U32(val) => {
                let buf = val.to_be_bytes();
                let len = len_for_value(val);
                write_atom(&mut output, &buf[4 - len..])?;
            }
            NodeVisitor::Pair(left, right) => {
                output.push(CONS_BOX_MARKER);
                values.push(right);
                values.push(left);
            }
        }
    }
    
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_nil() {
        let mut a = Allocator::new();
        let nil = a.nil();
        let bytes = node_to_bytes(&a, nil).unwrap();
        assert_eq!(bytes, vec![0x80]);
        
        let restored = node_from_bytes(&mut a, &bytes).unwrap();
        assert_eq!(nil, restored);
    }

    #[test]
    fn test_roundtrip_small_number() {
        let mut a = Allocator::new();
        let num = a.new_small_number(42).unwrap();
        let bytes = node_to_bytes(&a, num).unwrap();
        assert_eq!(bytes, vec![42]);
        
        let restored = node_from_bytes(&mut a, &bytes).unwrap();
        let bytes2 = node_to_bytes(&a, restored).unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn test_roundtrip_pair() {
        let mut a = Allocator::new();
        let left = a.new_small_number(1).unwrap();
        let right = a.new_small_number(2).unwrap();
        let pair = a.new_pair(left, right).unwrap();
        
        let bytes = node_to_bytes(&a, pair).unwrap();
        assert_eq!(bytes, vec![0xff, 1, 2]);
        
        let restored = node_from_bytes(&mut a, &bytes).unwrap();
        let bytes2 = node_to_bytes(&a, restored).unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn test_roundtrip_atom() {
        let mut a = Allocator::new();
        let atom = a.new_atom(&[0xde, 0xad, 0xbe, 0xef]).unwrap();
        
        let bytes = node_to_bytes(&a, atom).unwrap();
        let restored = node_from_bytes(&mut a, &bytes).unwrap();
        let bytes2 = node_to_bytes(&a, restored).unwrap();
        assert_eq!(bytes, bytes2);
    }
}
