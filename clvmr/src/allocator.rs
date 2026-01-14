use alloc::vec::Vec;
use core::borrow::Borrow;
use core::fmt;
use core::hash::Hash;
use core::hash::Hasher;
use core::ops::Deref;

use crate::error::{EvalErr, Result};
use crate::number::{Number, number_from_u8};

#[cfg(feature = "bls-ops")]
use chia_bls::{G1Element, G2Element};

#[cfg(feature = "allocator-debug")]
use rand::RngCore;

const MAX_NUM_ATOMS: usize = 62500000;
const MAX_NUM_PAIRS: usize = 62500000;
const NODE_PTR_IDX_BITS: u32 = 26;
const NODE_PTR_IDX_MASK: u32 = (1 << NODE_PTR_IDX_BITS) - 1;

#[cfg(feature = "allocator-debug")]
#[derive(Clone, Copy)]
struct AllocatorReference {
    // the low 24 bits are fingerprint
    // the top 8 bits are version
    fingerprint: u32,
}

#[cfg(feature = "allocator-debug")]
#[derive(Clone, Copy)]
pub struct NodePtr(u32, AllocatorReference);

#[cfg(feature = "allocator-debug")]
impl Hash for NodePtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[cfg(feature = "allocator-debug")]
impl PartialEq for NodePtr {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

#[cfg(feature = "allocator-debug")]
impl Eq for NodePtr {}

#[cfg(not(feature = "allocator-debug"))]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodePtr(u32);

impl fmt::Debug for NodePtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NodePtr")
            .field(&self.object_type())
            .field(&self.index())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectType {
    // The low bits form an index into the pair_vec
    Pair,
    // The low bits form an index into the atom_vec
    Bytes,
    // The low bits are the atom itself (unsigned integer, 26 bits)
    SmallAtom,
}

// The top 6 bits of the NodePtr indicate what type of object it is
impl NodePtr {
    pub const NIL: Self = Self::new(ObjectType::SmallAtom, 0);

    #[cfg(not(feature = "allocator-debug"))]
    const fn new(object_type: ObjectType, index: usize) -> Self {
        debug_assert!(index <= NODE_PTR_IDX_MASK as usize);
        NodePtr(((object_type as u32) << NODE_PTR_IDX_BITS) | (index as u32))
    }

    #[cfg(feature = "allocator-debug")]
    const fn new(object_type: ObjectType, index: usize) -> Self {
        debug_assert!(index <= NODE_PTR_IDX_MASK as usize);
        NodePtr(
            ((object_type as u32) << NODE_PTR_IDX_BITS) | (index as u32),
            AllocatorReference {
                fingerprint: u32::MAX,
            },
        )
    }

    #[cfg(feature = "allocator-debug")]
    const fn new_debug(object_type: ObjectType, index: usize, ar: AllocatorReference) -> Self {
        debug_assert!(index <= NODE_PTR_IDX_MASK as usize);
        NodePtr(
            ((object_type as u32) << NODE_PTR_IDX_BITS) | (index as u32),
            ar,
        )
    }

    pub fn is_atom(self) -> bool {
        matches!(
            self.object_type(),
            ObjectType::Bytes | ObjectType::SmallAtom
        )
    }

    pub fn is_pair(self) -> bool {
        self.object_type() == ObjectType::Pair
    }

    /// This is an advanced API that exposes implementation details.
    /// Returns the internal representation of this node
    pub fn object_type(self) -> ObjectType {
        match self.0 >> NODE_PTR_IDX_BITS {
            0 => ObjectType::Pair,
            1 => ObjectType::Bytes,
            2 => ObjectType::SmallAtom,
            _ => unreachable!(),
        }
    }

    /// This is an advanced API that exposes implementation details.
    /// Returns a dense index of low numbers for the specific ObjectType
    pub fn index(self) -> u32 {
        self.0 & NODE_PTR_IDX_MASK
    }
}

impl Default for NodePtr {
    fn default() -> Self {
        Self::NIL
    }
}

#[derive(PartialEq, Debug)]
pub enum SExp {
    Atom,
    Pair(NodePtr, NodePtr),
}

#[derive(Clone, Copy, Debug)]
struct AtomBuf {
    start: u32,
    end: u32,
}

impl AtomBuf {
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IntPair {
    first: NodePtr,
    rest: NodePtr,
}

// this represents a specific (former) state of an allocator. This can be used
// to restore an allocator to a previous state. It cannot be used to re-create
// the state from some other allocator.
pub struct Checkpoint {
    u8s: usize,
    pairs: usize,
    atoms: usize,
    ghost_atoms: usize,
    ghost_pairs: usize,
    ghost_heap: usize,
}

pub enum NodeVisitor<'a> {
    Buffer(&'a [u8]),
    U32(u32),
    Pair(NodePtr, NodePtr),
}

#[derive(Debug, Clone, Copy, Eq)]
pub enum Atom<'a> {
    Borrowed(&'a [u8]),
    U32([u8; 4], usize),
}

impl Hash for Atom<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl PartialEq for Atom<'_> {
    fn eq(&self, other: &Atom) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl AsRef<[u8]> for Atom<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::U32(bytes, len) => &bytes[4 - len..],
        }
    }
}

impl Deref for Atom<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl Borrow<[u8]> for Atom<'_> {
    fn borrow(&self) -> &[u8] {
        self.as_ref()
    }
}

#[derive(Debug)]
pub struct Allocator {
    // this is effectively a grow-only stack where atoms are allocated. Atoms
    // are immutable, so once they are created, they will stay around until the
    // program completes
    u8_vec: Vec<u8>,

    // storage for all pairs (positive indices)
    pair_vec: Vec<IntPair>,

    // storage for all atoms (negative indices).
    // node index -1 refers to index 0 in this vector, -2 refers to 1 and so
    // on.
    atom_vec: Vec<AtomBuf>,

    // the atom_vec may not grow past this
    heap_limit: usize,

    // the ghost counters are pretend atoms/pairs, that were optimized out. We
    // still account for them to not affect the limits of atoms and pairs. Those
    // limits must stay the same for consensus purpose.
    // For example, a "small atom", which is allocated in-place in the NodePtr.
    ghost_atoms: usize,
    ghost_pairs: usize,
    ghost_heap: usize,

    #[cfg(feature = "allocator-debug")]
    // fingerprints are 24 bits
    fingerprint: u32,

    // the number of atoms and pairs at different versions
    #[cfg(feature = "allocator-debug")]
    versions: Vec<(u32, u32)>,
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

pub fn fits_in_small_atom(v: &[u8]) -> Option<u32> {
    if !v.is_empty()
        && (v.len() > 4
        || (v.len() == 1 && v[0] == 0)
        // a 1-byte buffer of 0 is not the canonical representation of 0
        || (v[0] & 0x80) != 0
        // if the top bit is set, it's a negative number (i.e. not positive)
        || (v[0] == 0 && (v[1] & 0x80) == 0)
        // if the buffer is 4 bytes, the top byte can't use more than 2 bits.
        // otherwise the integer won't fit in 26 bits
        || (v.len() == 4 && v[0] > 0x03))
    {
        // if the top byte is a 0 but the top bit of the next byte is not set,
        // that's a redundant leading zero. i.e. not canonical representation
        None
    } else {
        let mut ret: u32 = 0;
        for b in v {
            ret <<= 8;
            ret |= *b as u32;
        }
        Some(ret)
    }
}

pub fn len_for_value(val: u32) -> usize {
    if val == 0 {
        0
    } else if val < 0x80 {
        1
    } else if val < 0x8000 {
        2
    } else if val < 0x800000 {
        3
    } else if val < 0x80000000 {
        4
    } else {
        5
    }
}

impl Allocator {
    pub fn new() -> Self {
        Self::new_limited(u32::MAX as usize)
    }

    pub fn new_limited(heap_limit: usize) -> Self {
        // we have a maximum of 4 GiB heap, because pointers are 32 bit unsigned
        assert!(heap_limit <= u32::MAX as usize);

        let mut r = Self {
            u8_vec: Vec::new(),
            pair_vec: Vec::new(),
            atom_vec: Vec::new(),
            // subtract 1 to compensate for the one() we used to allocate unconfitionally
            heap_limit: heap_limit - 1,
            // initialize this to 2 to behave as if we had allocated atoms for
            // nil() and one(), like we used to
            ghost_atoms: 2,
            ghost_pairs: 0,
            ghost_heap: 0,

            // fingerprints are 24 bits
            #[cfg(feature = "allocator-debug")]
            fingerprint: rand::thread_rng().next_u32() & 0xffffff,

            #[cfg(feature = "allocator-debug")]
            versions: Vec::new(),
        };
        r.u8_vec.reserve(1024 * 1024);
        r.atom_vec.reserve(256);
        r.pair_vec.reserve(256);
        r
    }

    #[cfg(feature = "allocator-debug")]
    fn validate_node(&self, n: NodePtr) {
        if n.1.fingerprint == u32::MAX {
            assert!(matches!(n.object_type(), ObjectType::SmallAtom));
            return;
        }

        assert_eq!(
            n.1.fingerprint & 0xffffff,
            self.fingerprint,
            "using a NodePtr on the wrong Allocator"
        );
        // if n.1.version is equal to self.versions.len() it means no
        // restore_checkpoint() has been called since this NodePtr was created
        let version = (n.1.fingerprint >> 24) as usize;
        if version < self.versions.len() {
            // self.versions contains the number of atoms (.0) and pairs (.1) at
            // the specific version
            match n.object_type() {
                ObjectType::Bytes => {
                    assert!(
                        n.index() < self.versions[version].0,
                        "NodePtr (atom) was invalidated by restore_checkpoint()"
                    );
                }
                ObjectType::Pair => {
                    assert!(
                        n.index() < self.versions[version].1,
                        "NodePtr (pair) was invalidated by restore_checkpoint()"
                    );
                }
                ObjectType::SmallAtom => {}
            }
        }
    }

    #[inline(always)]
    #[cfg(not(feature = "allocator-debug"))]
    fn mk_node(&self, t: ObjectType, idx: usize) -> NodePtr {
        NodePtr::new(t, idx)
    }

    #[inline(always)]
    #[cfg(feature = "allocator-debug")]
    fn mk_node(&self, t: ObjectType, idx: usize) -> NodePtr {
        assert!((self.fingerprint & 0xff000000) == 0);
        assert!(self.versions.len() <= 255);
        NodePtr::new_debug(
            t,
            idx,
            AllocatorReference {
                fingerprint: self.fingerprint | (self.versions.len() as u32) << 24,
            },
        )
    }

    // create a checkpoint for the current state of the allocator. This can be
    // used to go back to an earlier allocator state by passing the Checkpoint
    // to restore_checkpoint().
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            u8s: self.u8_vec.len(),
            pairs: self.pair_vec.len(),
            atoms: self.atom_vec.len(),
            ghost_atoms: self.ghost_atoms,
            ghost_pairs: self.ghost_pairs,
            ghost_heap: self.ghost_heap,
        }
    }

    pub fn restore_checkpoint(&mut self, cp: &Checkpoint) {
        // if any of these asserts fire, it means we're trying to restore to
        // a state that has already been "long-jumped" passed (via another
        // restore to an earlier state). You can only restore backwards in time,
        // not forwards.
        assert!(self.u8_vec.len() >= cp.u8s);
        assert!(self.pair_vec.len() >= cp.pairs);
        assert!(self.atom_vec.len() >= cp.atoms);
        self.u8_vec.truncate(cp.u8s);
        self.pair_vec.truncate(cp.pairs);
        self.atom_vec.truncate(cp.atoms);
        self.ghost_atoms = cp.ghost_atoms;
        self.ghost_pairs = cp.ghost_pairs;
        self.ghost_heap = cp.ghost_heap;

        // This invalidates all NodePtrs with higher index than this, with a
        // lower version than self.versions.len()
        #[cfg(feature = "allocator-debug")]
        self.versions
            .push((self.atom_vec.len() as u32, self.pair_vec.len() as u32));
    }

    pub fn new_atom(&mut self, v: &[u8]) -> Result<NodePtr> {
        let start = self.u8_vec.len() as u32;
        if (self.heap_limit - start as usize - self.ghost_heap) < v.len() {
            return Err(EvalErr::OutOfMemory);
        }
        let idx = self.atom_vec.len();
        self.check_atom_limit()?;
        if let Some(ret) = fits_in_small_atom(v) {
            self.ghost_atoms += 1;
            Ok(self.mk_node(ObjectType::SmallAtom, ret as usize))
        } else {
            self.u8_vec.extend_from_slice(v);
            let end = self.u8_vec.len() as u32;
            self.atom_vec.push(AtomBuf { start, end });
            Ok(self.mk_node(ObjectType::Bytes, idx))
        }
    }

    pub fn new_small_number(&mut self, v: u32) -> Result<NodePtr> {
        debug_assert!(v <= NODE_PTR_IDX_MASK);
        self.check_atom_limit()?;
        self.ghost_atoms += 1;
        Ok(self.mk_node(ObjectType::SmallAtom, v as usize))
    }

    pub fn new_number(&mut self, v: Number) -> Result<NodePtr> {
        use num_traits::ToPrimitive;
        if let Some(val) = v.to_u32() {
            if val <= NODE_PTR_IDX_MASK {
                return self.new_small_number(val);
            }
        }
        let bytes: Vec<u8> = v.to_signed_bytes_be();
        let mut slice = bytes.as_slice();

        // make number minimal by removing leading zeros
        while (!slice.is_empty()) && (slice[0] == 0) {
            if slice.len() > 1 && (slice[1] & 0x80 == 0x80) {
                break;
            }
            slice = &slice[1..];
        }
        self.new_atom(slice)
    }

    #[cfg(feature = "bls-ops")]
    pub fn new_g1(&mut self, g1: G1Element) -> Result<NodePtr> {
        self.new_atom(&g1.to_bytes())
    }

    #[cfg(feature = "bls-ops")]
    pub fn new_g2(&mut self, g2: G2Element) -> Result<NodePtr> {
        self.new_atom(&g2.to_bytes())
    }

    pub fn new_pair(&mut self, first: NodePtr, rest: NodePtr) -> Result<NodePtr> {
        #[cfg(feature = "allocator-debug")]
        {
            self.validate_node(first);
            self.validate_node(rest);
        }
        let idx = self.pair_vec.len();
        if idx >= MAX_NUM_PAIRS - self.ghost_pairs {
            return Err(EvalErr::TooManyPairs);
        }
        self.pair_vec.push(IntPair { first, rest });
        Ok(self.mk_node(ObjectType::Pair, idx))
    }

    // this code is used when we are simulating pairs with a vec locally
    // in the deserialize_br code
    // we must maintain parity with the old deserialize_br code so need to track the skipped pairs
    pub fn add_ghost_pair(&mut self, amount: usize) -> Result<()> {
        if MAX_NUM_PAIRS - self.ghost_pairs - self.pair_vec.len() < amount {
            return Err(EvalErr::TooManyPairs);
        }
        self.ghost_pairs += amount;
        Ok(())
    }

    // this code is used when we actually create the pairs that were previously skipped ghost pairs
    pub fn remove_ghost_pair(&mut self, amount: usize) -> Result<()> {
        // currently let this panic with overflow if we go below 0 to debug if/where it happens
        debug_assert!(self.ghost_pairs >= amount);
        self.ghost_pairs -= amount;
        Ok(())
    }

    pub fn add_ghost_atom(&mut self, amount: usize) -> Result<()> {
        if MAX_NUM_ATOMS - self.ghost_atoms - self.atom_vec.len() < amount {
            return Err(EvalErr::TooManyAtoms);
        }
        self.ghost_atoms += amount;
        Ok(())
    }
    pub fn new_substr(&mut self, node: NodePtr, start: u32, end: u32) -> Result<NodePtr> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        self.check_atom_limit()?;

        fn bounds_check(node: NodePtr, start: u32, end: u32, len: u32) -> Result<()> {
            if start > len {
                Err(EvalErr::InvalidAllocArg(
                    node,
                    alloc::format!("substr start out of bounds: {start} > {len}"),
                ))?;
            }
            if end > len {
                Err(EvalErr::InvalidAllocArg(
                    node,
                    alloc::format!("substr end out of bounds: {end} > {len}"),
                ))?;
            }
            if end < start {
                Err(EvalErr::InvalidAllocArg(
                    node,
                    alloc::format!("substr invalid bounds: {end} < {start}"),
                ))?;
            }
            Ok(())
        }

        match node.object_type() {
            ObjectType::Pair => Err(EvalErr::InternalError(
                node,
                "substr expected atom, got pair".into(),
            ))?,
            ObjectType::Bytes => {
                let atom = self.atom_vec[node.index() as usize];
                let atom_len = atom.end - atom.start;
                bounds_check(node, start, end, atom_len)?;
                let idx = self.atom_vec.len();
                self.atom_vec.push(AtomBuf {
                    start: atom.start + start,
                    end: atom.start + end,
                });
                Ok(self.mk_node(ObjectType::Bytes, idx))
            }
            ObjectType::SmallAtom => {
                let val = node.index();
                let len = len_for_value(val) as u32;
                bounds_check(node, start, end, len)?;
                let buf: [u8; 4] = val.to_be_bytes();
                let buf = &buf[4 - len as usize..];
                let substr = &buf[start as usize..end as usize];
                if let Some(new_val) = fits_in_small_atom(substr) {
                    self.ghost_atoms += 1;
                    Ok(self.mk_node(ObjectType::SmallAtom, new_val as usize))
                } else {
                    let start = self.u8_vec.len();
                    let end = start + substr.len();
                    self.u8_vec.extend_from_slice(substr);
                    let idx = self.atom_vec.len();
                    self.atom_vec.push(AtomBuf {
                        start: start as u32,
                        end: end as u32,
                    });
                    Ok(self.mk_node(ObjectType::Bytes, idx))
                }
            }
        }
    }

    pub fn new_concat(&mut self, new_size: usize, nodes: &[NodePtr]) -> Result<NodePtr> {
        #[cfg(feature = "allocator-debug")]
        {
            for n in nodes {
                self.validate_node(*n);
            }
        }

        self.check_atom_limit()?;
        let start = self.u8_vec.len();
        if self.heap_limit - start - self.ghost_heap < new_size {
            return Err(EvalErr::OutOfMemory);
        }

        if nodes.is_empty() {
            if 0 != new_size {
                return Err(EvalErr::InternalError(
                    self.nil(),
                    "concat passed invalid new_size".into(),
                ))?;
            }
            // pretend that we created a new atom and allocated new_size bytes on the heap
            self.ghost_atoms += 1;
            return Ok(self.nil());
        }

        if nodes.len() == 1 {
            if self.atom_len(nodes[0]) != new_size {
                return Err(EvalErr::InternalError(
                    self.nil(),
                    "concat passed invalid new_size".into(),
                ))?;
            }
            // pretend that we created a new atom and allocated new_size bytes on the heap
            self.ghost_heap += new_size;
            self.ghost_atoms += 1;
            return Ok(nodes[0]);
        }

        self.u8_vec.reserve(new_size);

        let mut counter: usize = 0;
        for node in nodes {
            match node.object_type() {
                ObjectType::Pair => {
                    self.u8_vec.truncate(start);
                    return Err(EvalErr::InternalError(
                        *node,
                        "concat expected atom, got pair".into(),
                    ))?;
                }
                ObjectType::Bytes => {
                    let term = self.atom_vec[node.index() as usize];
                    if counter + term.len() > new_size {
                        self.u8_vec.truncate(start);
                        return Err(EvalErr::InternalError(
                            *node,
                            "concat passed invalid new_size".into(),
                        ))?;
                    }
                    self.u8_vec
                        .extend_from_within(term.start as usize..term.end as usize);
                    counter += term.len();
                }
                ObjectType::SmallAtom => {
                    let val = node.index();
                    let len = len_for_value(val) as u32;
                    let buf: [u8; 4] = val.to_be_bytes();
                    let buf = &buf[4 - len as usize..];
                    self.u8_vec.extend_from_slice(buf);
                    counter += len as usize;
                }
            }
        }
        if counter != new_size {
            self.u8_vec.truncate(start);
            return Err(EvalErr::InternalError(
                self.nil(),
                "concat passed invalid new_size".into(),
            ))?;
        }
        let end = self.u8_vec.len() as u32;
        let idx = self.atom_vec.len();
        self.atom_vec.push(AtomBuf {
            start: start as u32,
            end,
        });
        Ok(self.mk_node(ObjectType::Bytes, idx))
    }

    pub fn atom_eq(&self, lhs: NodePtr, rhs: NodePtr) -> bool {
        #[cfg(feature = "allocator-debug")]
        {
            self.validate_node(lhs);
            self.validate_node(rhs);
        }
        let lhs_type = lhs.object_type();
        let rhs_type = rhs.object_type();

        match (lhs_type, rhs_type) {
            (ObjectType::Pair, _) | (_, ObjectType::Pair) => {
                panic!("atom_eq() called on pair");
            }
            (ObjectType::Bytes, ObjectType::Bytes) => {
                let lhs = self.atom_vec[lhs.index() as usize];
                let rhs = self.atom_vec[rhs.index() as usize];
                self.u8_vec[lhs.start as usize..lhs.end as usize]
                    == self.u8_vec[rhs.start as usize..rhs.end as usize]
            }
            (ObjectType::SmallAtom, ObjectType::SmallAtom) => lhs.index() == rhs.index(),
            (ObjectType::SmallAtom, ObjectType::Bytes) => {
                self.bytes_eq_int(self.atom_vec[rhs.index() as usize], lhs.index())
            }
            (ObjectType::Bytes, ObjectType::SmallAtom) => {
                self.bytes_eq_int(self.atom_vec[lhs.index() as usize], rhs.index())
            }
        }
    }

    fn bytes_eq_int(&self, atom: AtomBuf, val: u32) -> bool {
        let len = len_for_value(val) as u32;
        if (atom.end - atom.start) != len {
            return false;
        }
        if val == 0 {
            return true;
        }

        if self.u8_vec[atom.start as usize] & 0x80 != 0 {
            // SmallAtom only represents positive values
            // if the byte buffer is negative, they can't match
            return false;
        }

        // since we know the value of atom is small, we can turn it into a u32 and compare
        // against val
        let mut atom_val: u32 = 0;
        for i in atom.start..atom.end {
            atom_val <<= 8;
            atom_val |= self.u8_vec[i as usize] as u32;
        }
        val == atom_val
    }

    pub fn atom(&self, node: NodePtr) -> Atom<'_> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        let index = node.index();

        match node.object_type() {
            ObjectType::Bytes => {
                let atom = self.atom_vec[index as usize];
                Atom::Borrowed(&self.u8_vec[atom.start as usize..atom.end as usize])
            }
            ObjectType::SmallAtom => {
                let len = len_for_value(index);
                let bytes = index.to_be_bytes();
                Atom::U32(bytes, len)
            }
            _ => panic!("expected atom, got pair"),
        }
    }

    pub fn atom_len(&self, node: NodePtr) -> usize {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        let index = node.index();

        match node.object_type() {
            ObjectType::Bytes => {
                let atom = self.atom_vec[index as usize];
                (atom.end - atom.start) as usize
            }
            ObjectType::SmallAtom => len_for_value(index),
            _ => {
                panic!("expected atom, got pair");
            }
        }
    }

    pub fn small_number(&self, node: NodePtr) -> Option<u32> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        match node.object_type() {
            ObjectType::SmallAtom => Some(node.index()),
            ObjectType::Bytes => {
                let atom = self.atom_vec[node.index() as usize];
                let buf = &self.u8_vec[atom.start as usize..atom.end as usize];
                fits_in_small_atom(buf)
            }
            _ => None,
        }
    }

    pub fn number(&self, node: NodePtr) -> Number {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        let index = node.index();

        match node.object_type() {
            ObjectType::Bytes => {
                let atom = self.atom_vec[index as usize];
                number_from_u8(&self.u8_vec[atom.start as usize..atom.end as usize])
            }
            ObjectType::SmallAtom => Number::from(index),
            _ => {
                panic!("number() called on pair");
            }
        }
    }

    #[cfg(feature = "bls-ops")]
    pub fn g1(&self, node: NodePtr) -> Result<G1Element> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        let idx = match node.object_type() {
            ObjectType::Bytes => node.index(),
            ObjectType::SmallAtom => {
                return Err(EvalErr::InvalidAllocArg(
                    node,
                    "atom is not G1 size, 48 bytes".into(),
                ))?;
            }
            ObjectType::Pair => {
                return Err(EvalErr::InvalidAllocArg(
                    node,
                    "pair found, expected G1 point".into(),
                ))?;
            }
        };
        let atom = self.atom_vec[idx as usize];
        if atom.end - atom.start != 48 {
            return Err(EvalErr::InvalidAllocArg(
                node,
                "atom is not G1 size, 48 bytes".into(),
            ))?;
        }

        let array: &[u8; 48] = &self.u8_vec[atom.start as usize..atom.end as usize]
            .try_into()
            .map_err(|_| {
                EvalErr::InvalidAllocArg(node, "atom is not G1 size, 48 bytes".into())
            })?;
        G1Element::from_bytes(array)
            .map_err(|_| EvalErr::InvalidAllocArg(node, "atom is not a G1 point".into()))
    }

    #[cfg(feature = "bls-ops")]
    pub fn g2(&self, node: NodePtr) -> Result<G2Element> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        let idx = match node.object_type() {
            ObjectType::Bytes => node.index(),
            ObjectType::SmallAtom => {
                return Err(EvalErr::InvalidAllocArg(
                    node,
                    "atom is not G2 size, 96 bytes".into(),
                ))?;
            }
            ObjectType::Pair => {
                return Err(EvalErr::InvalidAllocArg(
                    node,
                    "pair found, expected G2 point".into(),
                ))?;
            }
        };

        let atom = self.atom_vec[idx as usize];

        let array: &[u8; 96] = &self.u8_vec[atom.start as usize..atom.end as usize]
            .try_into()
            .map_err(|_| {
                EvalErr::InvalidAllocArg(node, "atom is not G2 size, 96 bytes".into())
            })?;

        G2Element::from_bytes(array)
            .map_err(|_| EvalErr::InvalidAllocArg(node, "atom is not a G2 point".into()))
    }

    pub fn node(&self, node: NodePtr) -> NodeVisitor<'_> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        let index = node.index();

        match node.object_type() {
            ObjectType::Bytes => {
                let atom = self.atom_vec[index as usize];
                let buf = &self.u8_vec[atom.start as usize..atom.end as usize];
                NodeVisitor::Buffer(buf)
            }
            ObjectType::SmallAtom => NodeVisitor::U32(index),
            ObjectType::Pair => {
                let pair = self.pair_vec[index as usize];
                NodeVisitor::Pair(pair.first, pair.rest)
            }
        }
    }

    pub fn sexp(&self, node: NodePtr) -> SExp {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(node);

        match node.object_type() {
            ObjectType::Bytes | ObjectType::SmallAtom => SExp::Atom,
            ObjectType::Pair => {
                let pair = self.pair_vec[node.index() as usize];
                SExp::Pair(pair.first, pair.rest)
            }
        }
    }

    // this is meant to be used when iterating lists:
    // while let Some((i, rest)) = a.next(node) {
    //     node = rest;
    //     ...
    // }
    pub fn next(&self, n: NodePtr) -> Option<(NodePtr, NodePtr)> {
        #[cfg(feature = "allocator-debug")]
        self.validate_node(n);

        match self.sexp(n) {
            SExp::Pair(first, rest) => Some((first, rest)),
            SExp::Atom => None,
        }
    }

    pub fn nil(&self) -> NodePtr {
        self.mk_node(ObjectType::SmallAtom, 0)
    }

    pub fn one(&self) -> NodePtr {
        self.mk_node(ObjectType::SmallAtom, 1)
    }

    #[inline]
    fn check_atom_limit(&self) -> Result<()> {
        if self.atom_vec.len() + self.ghost_atoms == MAX_NUM_ATOMS {
            Err(EvalErr::TooManyAtoms)
        } else {
            Ok(())
        }
    }

    pub fn atom_count(&self) -> usize {
        self.atom_vec.len()
    }

    pub fn small_atom_count(&self) -> usize {
        self.ghost_atoms
    }

    pub fn pair_count(&self) -> usize {
        self.pair_vec.len() + self.ghost_pairs
    }

    pub fn pair_count_no_ghosts(&self) -> usize {
        self.pair_vec.len()
    }

    pub fn heap_size(&self) -> usize {
        self.u8_vec.len()
    }
}
