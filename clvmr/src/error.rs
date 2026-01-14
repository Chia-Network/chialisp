use alloc::string::String;
use core::fmt;

use crate::NodePtr;

pub type Result<T> = core::result::Result<T, EvalErr>;

#[cfg(feature = "std")]
impl From<std::io::Error> for EvalErr {
    fn from(_: std::io::Error) -> Self {
        EvalErr::SerializationError
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvalErr {
    SerializationError,
    SerializationBackreferenceError,
    OutOfMemory,
    PathIntoAtom,
    TooManyPairs,
    TooManyAtoms,
    CostExceeded,
    UnknownSoftforkExtension,
    SoftforkCostMismatch,
    InternalError(NodePtr, String),
    Raise(NodePtr),
    InvalidNilTerminator(NodePtr),
    DivisionByZero(NodePtr),
    ValueStackLimitReached(NodePtr),
    EnvironmentStackLimitReached(NodePtr),
    ShiftTooLarge(NodePtr),
    Reserved(NodePtr),
    Invalid(NodePtr),
    Unimplemented(NodePtr),
    InvalidOpArg(NodePtr, String),
    InvalidAllocArg(NodePtr, String),
    BLSPairingIdentityFailed(NodePtr),
    BLSVerifyFailed(NodePtr),
    Secp256Failed(NodePtr),
}

impl fmt::Display for EvalErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalErr::SerializationError => write!(f, "bad encoding"),
            EvalErr::SerializationBackreferenceError => {
                write!(f, "invalid backreference during deserialisation")
            }
            EvalErr::OutOfMemory => write!(f, "Out of Memory"),
            EvalErr::PathIntoAtom => write!(f, "path into atom"),
            EvalErr::TooManyPairs => write!(f, "too many pairs"),
            EvalErr::TooManyAtoms => write!(f, "Too Many Atoms"),
            EvalErr::CostExceeded => write!(f, "cost exceeded or below zero"),
            EvalErr::UnknownSoftforkExtension => write!(f, "unknown softfork extension"),
            EvalErr::SoftforkCostMismatch => write!(f, "softfork specified cost mismatch"),
            EvalErr::InternalError(_, msg) => write!(f, "Internal Error: {}", msg),
            EvalErr::Raise(_) => write!(f, "clvm raise"),
            EvalErr::InvalidNilTerminator(_) => {
                write!(f, "Invalid Nil Terminator in operand list")
            }
            EvalErr::DivisionByZero(_) => write!(f, "Division by zero"),
            EvalErr::ValueStackLimitReached(_) => write!(f, "Value Stack Limit Reached"),
            EvalErr::EnvironmentStackLimitReached(_) => {
                write!(f, "Environment Stack Limit Reached")
            }
            EvalErr::ShiftTooLarge(_) => write!(f, "Shift too large"),
            EvalErr::Reserved(_) => write!(f, "Reserved operator"),
            EvalErr::Invalid(_) => write!(f, "invalid operator"),
            EvalErr::Unimplemented(_) => write!(f, "unimplemented operator"),
            EvalErr::InvalidOpArg(_, msg) => write!(f, "InvalidOperatorArg: {}", msg),
            EvalErr::InvalidAllocArg(_, msg) => write!(f, "InvalidAllocatorArg: {}", msg),
            EvalErr::BLSPairingIdentityFailed(_) => write!(f, "bls_pairing_identity failed"),
            EvalErr::BLSVerifyFailed(_) => write!(f, "bls_verify failed"),
            EvalErr::Secp256Failed(_) => write!(f, "Secp256 Verify Error: failed"),
        }
    }
}

impl EvalErr {
    fn node(&self) -> Option<NodePtr> {
        match self {
            EvalErr::InternalError(node, _) => Some(*node),
            EvalErr::Raise(node) => Some(*node),
            EvalErr::InvalidNilTerminator(node) => Some(*node),
            EvalErr::DivisionByZero(node) => Some(*node),
            EvalErr::ValueStackLimitReached(node) => Some(*node),
            EvalErr::EnvironmentStackLimitReached(node) => Some(*node),
            EvalErr::ShiftTooLarge(node) => Some(*node),
            EvalErr::Reserved(node) => Some(*node),
            EvalErr::Invalid(node) => Some(*node),
            EvalErr::Unimplemented(node) => Some(*node),
            EvalErr::InvalidOpArg(node, _) => Some(*node),
            EvalErr::InvalidAllocArg(node, _) => Some(*node),
            EvalErr::BLSPairingIdentityFailed(node) => Some(*node),
            EvalErr::BLSVerifyFailed(node) => Some(*node),
            EvalErr::Secp256Failed(node) => Some(*node),
            _ => None,
        }
    }

    pub fn node_ptr(&self) -> NodePtr {
        // This is a convenience function to get the node pointer
        self.node().unwrap_or_default()
    }
}
