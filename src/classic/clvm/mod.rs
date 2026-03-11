use std::collections::HashMap;

pub mod __type_compatibility__;
pub mod as_rust;
pub mod casts;
pub mod costs;
pub mod serialize;
pub mod sexp;
pub mod syntax_error;

/// This specifies the latest "version" of the operator set.  This will increase
/// each time a new set of operators is included in the primitive set in clvm.
/// We keep track of what was added when so users can specify what version of the
/// tools' output they're expecting when it matters.
pub const OPERATORS_LATEST_VERSION: usize = 2;

struct KwAtomPair {
    v: &'static [u8],
    n: &'static str,
    version: usize,
}

const KW_PAIRS: [KwAtomPair; 49] = [
    KwAtomPair {
        v: &[0x01],
        n: "q",
        version: 0,
    },
    KwAtomPair {
        v: &[0x02],
        n: "a",
        version: 0,
    },
    KwAtomPair {
        v: &[0x03],
        n: "i",
        version: 0,
    },
    KwAtomPair {
        v: &[0x04],
        n: "c",
        version: 0,
    },
    KwAtomPair {
        v: &[0x05],
        n: "f",
        version: 0,
    },
    KwAtomPair {
        v: &[0x06],
        n: "r",
        version: 0,
    },
    KwAtomPair {
        v: &[0x07],
        n: "l",
        version: 0,
    },
    KwAtomPair {
        v: &[0x08],
        n: "x",
        version: 0,
    },
    KwAtomPair {
        v: &[0x09],
        n: "=",
        version: 0,
    },
    KwAtomPair {
        v: &[0x0a],
        n: ">s",
        version: 0,
    },
    KwAtomPair {
        v: &[0x0b],
        n: "sha256",
        version: 0,
    },
    KwAtomPair {
        v: &[0x0c],
        n: "substr",
        version: 0,
    },
    KwAtomPair {
        v: &[0x0d],
        n: "strlen",
        version: 0,
    },
    KwAtomPair {
        v: &[0x0e],
        n: "concat",
        version: 0,
    },
    KwAtomPair {
        v: &[0x10],
        n: "+",
        version: 0,
    },
    KwAtomPair {
        v: &[0x11],
        n: "-",
        version: 0,
    },
    KwAtomPair {
        v: &[0x12],
        n: "*",
        version: 0,
    },
    KwAtomPair {
        v: &[0x13],
        n: "/",
        version: 0,
    },
    KwAtomPair {
        v: &[0x14],
        n: "divmod",
        version: 0,
    },
    KwAtomPair {
        v: &[0x15],
        n: ">",
        version: 0,
    },
    KwAtomPair {
        v: &[0x16],
        n: "ash",
        version: 0,
    },
    KwAtomPair {
        v: &[0x17],
        n: "lsh",
        version: 0,
    },
    KwAtomPair {
        v: &[0x18],
        n: "logand",
        version: 0,
    },
    KwAtomPair {
        v: &[0x19],
        n: "logior",
        version: 0,
    },
    KwAtomPair {
        v: &[0x1a],
        n: "logxor",
        version: 0,
    },
    KwAtomPair {
        v: &[0x1b],
        n: "lognot",
        version: 0,
    },
    KwAtomPair {
        v: &[0x1d],
        n: "point_add",
        version: 0,
    },
    KwAtomPair {
        v: &[0x1e],
        n: "pubkey_for_exp",
        version: 0,
    },
    KwAtomPair {
        v: &[0x20],
        n: "not",
        version: 0,
    },
    KwAtomPair {
        v: &[0x21],
        n: "any",
        version: 0,
    },
    KwAtomPair {
        v: &[0x22],
        n: "all",
        version: 0,
    },
    KwAtomPair {
        v: &[0x24],
        n: "softfork",
        version: 0,
    },
    KwAtomPair {
        v: &[0x30],
        n: "coinid",
        version: 1,
    },
    KwAtomPair {
        v: &[0x31],
        n: "g1_subtract",
        version: 1,
    },
    KwAtomPair {
        v: &[0x32],
        n: "g1_multiply",
        version: 1,
    },
    KwAtomPair {
        v: &[0x33],
        n: "g1_negate",
        version: 1,
    },
    KwAtomPair {
        v: &[0x34],
        n: "g2_add",
        version: 1,
    },
    KwAtomPair {
        v: &[0x35],
        n: "g2_subtract",
        version: 1,
    },
    KwAtomPair {
        v: &[0x36],
        n: "g2_multiply",
        version: 1,
    },
    KwAtomPair {
        v: &[0x37],
        n: "g2_negate",
        version: 1,
    },
    KwAtomPair {
        v: &[0x38],
        n: "g1_map",
        version: 1,
    },
    KwAtomPair {
        v: &[0x39],
        n: "g2_map",
        version: 1,
    },
    KwAtomPair {
        v: &[0x3a],
        n: "bls_pairing_identity",
        version: 1,
    },
    KwAtomPair {
        v: &[0x3b],
        n: "bls_verify",
        version: 1,
    },
    KwAtomPair {
        v: &[0x3c],
        n: "modpow",
        version: 1,
    },
    KwAtomPair {
        v: &[0x3d],
        n: "%",
        version: 1,
    },
    KwAtomPair {
        v: &[0x3e],
        n: "keccak256",
        version: 2,
    },
    KwAtomPair {
        v: &[0x13, 0xd6, 0x1f, 0x00],
        n: "secp256k1_verify",
        version: 1,
    },
    KwAtomPair {
        v: &[0x1c, 0x3a, 0x8f, 0x00],
        n: "secp256r1_verify",
        version: 1,
    },
];

lazy_static! {
    pub static ref KEYWORD_FROM_ATOM_0: HashMap<Vec<u8>, String> = {
        let mut result = HashMap::new();
        for pair in KW_PAIRS.iter().filter(|p| p.version == 0) {
            result.insert(pair.v.to_vec(), pair.n.to_string());
        }
        result
    };
    pub static ref KEYWORD_TO_ATOM_0: HashMap<String, Vec<u8>> = {
        let mut result = HashMap::new();
        for pair in KW_PAIRS.iter().filter(|p| p.version == 0) {
            result.insert(pair.n.to_string(), pair.v.to_vec());
        }
        result
    };
    pub static ref KEYWORD_FROM_ATOM_1: HashMap<Vec<u8>, String> = {
        let mut result = HashMap::new();
        for pair in KW_PAIRS.iter().filter(|p| p.version <= 1) {
            result.insert(pair.v.to_vec(), pair.n.to_string());
        }
        result
    };
    pub static ref KEYWORD_TO_ATOM_1: HashMap<String, Vec<u8>> = {
        let mut result = HashMap::new();
        for pair in KW_PAIRS.iter().filter(|p| p.version <= 1) {
            result.insert(pair.n.to_string(), pair.v.to_vec());
        }
        result
    };
    pub static ref KEYWORD_FROM_ATOM_2: HashMap<Vec<u8>, String> = {
        let mut result = HashMap::new();
        for pair in KW_PAIRS.iter().filter(|p| p.version <= 2) {
            result.insert(pair.v.to_vec(), pair.n.to_string());
        }
        result
    };
    pub static ref KEYWORD_TO_ATOM_2: HashMap<String, Vec<u8>> = {
        let mut result = HashMap::new();
        for pair in KW_PAIRS.iter().filter(|p| p.version <= 2) {
            result.insert(pair.n.to_string(), pair.v.to_vec());
        }
        result
    };
}

pub fn keyword_from_atom(version: usize) -> &'static HashMap<Vec<u8>, String> {
    match version {
        0 => &KEYWORD_FROM_ATOM_0,
        1 => &KEYWORD_FROM_ATOM_1,
        _ => &KEYWORD_FROM_ATOM_2,
    }
}

pub fn keyword_to_atom(version: usize) -> &'static HashMap<String, Vec<u8>> {
    match version {
        0 => &KEYWORD_TO_ATOM_0,
        1 => &KEYWORD_TO_ATOM_1,
        _ => &KEYWORD_TO_ATOM_2,
    }
}
