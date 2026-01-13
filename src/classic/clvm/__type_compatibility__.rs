use chia_bls::PublicKey;
use num_traits::{One, Zero};
use std::clone::Clone;
use std::cmp::Ordering;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

use sha2::Digest;
use sha2::Sha256;

use crate::util::Number;
//use crate::classic::clvm::__type_compatibility__::SyntaxErr;
use crate::classic::clvm::syntax_error::SyntaxErr;

pub fn to_hexstr(r: &[u8]) -> String {
    hex::encode(r)
}

pub fn char_to_string(ch: char) -> String {
    String::from_utf8(vec![ch as u8]).unwrap_or_default()
}

pub fn vec_to_string(r: &[u8]) -> String {
    String::from_utf8_lossy(r).as_ref().to_string()
}

/**
 * Get python's bytes.__repr__ style string.
 * @see https://github.com/python/cpython/blob/main/Objects/bytesobject.c#L1337
 * @param {Uint8Array} r - byteArray to stringify
 */
pub fn pybytes_repr(r: &[u8], dquoted: bool, full_repr: bool) -> String {
    let mut squotes = 0;
    let mut dquotes = 0;
    for b in r.iter() {
        let c = *b as char;
        match c {
            '\'' => squotes += 1,
            '\"' => dquotes += 1,
            _ => (),
        }
    }
    let mut quote = '\'';
    if squotes > 0 && dquotes == 0 || dquoted {
        quote = '\"';
    }

    let mut s = "".to_string();

    if !dquoted {
        s += "b";
    }

    s += char_to_string(quote).as_str();

    for b in r.iter() {
        let c = *b as char;
        if c == quote || (c == '\\' && full_repr) {
            s = (s + "\\") + char_to_string(c).as_str();
        } else if c == '\t' {
            s += "\\t";
        } else if c == '\n' {
            s += "\\n";
        } else if c == '\r' {
            s += "\\r";
        } else if c < ' ' || *b >= 0x7f {
            s += "\\x";
            s += hex::encode(vec![*b]).as_str();
        } else {
            s += char_to_string(c).as_str();
        }
    }

    s += char_to_string(quote).as_str();

    s
}

pub enum UnvalidatedBytesFromType {
    Hex(String),
}

pub enum BytesFromType {
    Raw(Vec<u8>),
    String(String),
    G1Element(PublicKey),
}

#[derive(Debug, Clone)]
pub struct Bytes {
    _b: Vec<u8>,
}

pub fn ordering_to_int(o: Ordering) -> i32 {
    match o {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

/**
 * Unlike python, there is no immutable byte type in javascript.
 */
impl Bytes {
    pub fn new(value: Option<BytesFromType>) -> Self {
        match value {
            None => Bytes { _b: vec![] },
            Some(BytesFromType::Raw(v)) => Bytes { _b: v },
            Some(BytesFromType::String(s)) => {
                let bytes = s.as_bytes();
                let mut bvec = vec![];
                for b in bytes {
                    bvec.push(*b);
                }
                Bytes::new(Some(BytesFromType::Raw(bvec)))
            }
            Some(BytesFromType::G1Element(g1)) => Bytes {
                _b: g1.to_bytes().to_vec(),
            },
        }
    }

    pub fn new_validated(value: Option<UnvalidatedBytesFromType>) -> Result<Self, SyntaxErr> {
        match value {
            Some(UnvalidatedBytesFromType::Hex(hstr)) => {
                #[allow(clippy::single_char_pattern)]
                let hex_stripped = hstr
                    .replace(" ", "")
                    .replace("\t", "")
                    .replace("\r", "")
                    .replace("\n", "");

                match hex::decode(&hex_stripped) {
                    Ok(d) => Ok(Bytes { _b: d }),
                    Err(e) => Err(SyntaxErr {
                        msg: format!("{e} in '{hex_stripped}'"),
                    }),
                }
            }
            None => Ok(Bytes { _b: vec![] }),
        }
    }

    pub fn length(&self) -> usize {
        self._b.len()
    }

    pub fn at(&self, i: usize) -> u8 {
        self._b[i]
    }

    pub fn raw(&self) -> Vec<u8> {
        self._b.clone()
    }

    pub fn concat(&self, b: &Bytes) -> Bytes {
        let mut this_bin = self._b.clone();
        let mut that_bin = b.raw();
        let mut concat_bin = Vec::<u8>::with_capacity(this_bin.len() + that_bin.len());
        concat_bin.append(&mut this_bin);
        concat_bin.append(&mut that_bin);
        Bytes::new(Some(BytesFromType::Raw(concat_bin)))
    }

    pub fn data(&self) -> &Vec<u8> {
        &self._b
    }

    pub fn to_formal_string(&self) -> String {
        pybytes_repr(&self._b, true, false)
    }

    pub fn pybytes(&self) -> String {
        pybytes_repr(&self._b, false, true)
    }

    pub fn hex(&self) -> String {
        to_hexstr(&self._b)
    }

    pub fn decode(&self) -> String {
        vec_to_string(&self._b)
    }

    pub fn startswith(&self, b: &Bytes) -> bool {
        for i in 0..min(b.length(), self._b.len()) - 1 {
            if b.at(i) != self._b[i] {
                return false;
            }
        }
        true
    }

    pub fn endswith(&self, b: &Bytes) -> bool {
        let blen = min(b.length(), self._b.len()) - 1;
        for i in 0..blen {
            if b.at(blen - i) != self._b[blen - i] {
                return false;
            }
        }
        true
    }
}

impl Display for Bytes {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        fmt.write_str(&pybytes_repr(&self._b, false, true))?;
        Ok(())
    }
}

pub fn sha256(value: Bytes) -> Bytes {
    let hashed = Sha256::digest(&value.data()[..]);
    let hashed_iter = hashed.into_iter();
    let newvec: Vec<u8> = hashed_iter.collect();
    Bytes::new(Some(BytesFromType::Raw(newvec)))
}

pub fn list<E, I>(vals: I) -> Vec<E>
where
    I: Iterator<Item = E>,
    E: Clone,
{
    vals.collect()
}

pub trait PythonStr {
    fn py_str(&self) -> String;
}

pub fn str<T>(thing: T) -> String
where
    T: PythonStr,
{
    thing.py_str()
}

pub trait PythonRepr {
    fn py_repr(&self) -> String;
}

pub fn repr<T>(thing: T) -> String
where
    T: PythonRepr,
{
    thing.py_repr()
}

#[derive(Debug)]
pub enum Tuple<T1, T2> {
    Tuple(T1, T2),
}

impl<T1, T2> Tuple<T1, T2> {
    pub fn first(&self) -> &T1 {
        match self {
            Tuple::Tuple(f, _) => f,
        }
    }

    pub fn rest(&self) -> &T2 {
        match self {
            Tuple::Tuple(_, r) => r,
        }
    }
}

impl<T1, T2> Display for Tuple<T1, T2>
where
    T1: PythonStr,
    T2: PythonStr,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        fmt.write_str("(")?;
        fmt.write_str(self.first().py_str().as_str())?;
        fmt.write_str(", ")?;
        fmt.write_str(self.rest().py_str().as_str())?;
        fmt.write_str(")")?;

        Ok(())
    }
}

pub fn t<T1, T2>(v1: T1, v2: T2) -> Tuple<T1, T2> {
    Tuple::Tuple(v1, v2)
}

impl<T1, T2> PythonStr for Tuple<T1, T2>
where
    T1: PythonStr,
    T2: PythonStr,
{
    fn py_str(&self) -> String {
        self.to_string()
    }
}

const BUF_ALLOC_MULTIPLIER: usize = 4;

pub type Record<K, V> = HashMap<K, V>;

#[derive(Debug)]
pub struct Stream {
    seek: usize,
    length: usize,
    buffer: Vec<u8>,
}

impl Stream {
    pub fn new(b: Option<Bytes>) -> Self {
        match b {
            None => Stream {
                seek: 0,
                length: 0,
                buffer: vec![],
            },
            Some(b) => {
                let data = b.data().to_vec();
                Stream {
                    seek: 0,
                    length: data.len(),
                    buffer: data,
                }
            }
        }
    }

    pub fn get_seek(&self) -> usize {
        self.seek
    }

    pub fn set_seek(&mut self, value: i64) {
        if value < 0 {
            self.seek = self.length - 1;
        } else if value as usize > self.length - 1 {
            self.seek = self.length;
        } else {
            self.seek = value as usize;
        }
    }

    pub fn get_length(&self) -> usize {
        self.length
    }

    fn re_allocate(&mut self, size: Option<usize>) {
        let mut s = match size {
            None => self.buffer.len() * BUF_ALLOC_MULTIPLIER,
            Some(s) => s,
        };

        /*
         * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Errors/Invalid_array_length
         */
        if s > 4294967295 {
            // 4294967295 = 2**32 - 1
            s = 4294967295;
        }

        let mut buf = Vec::<u8>::with_capacity(s);
        for by in &self.buffer {
            buf.push(*by);
        }
        self.buffer = buf;
    }

    pub fn write(&mut self, b: Bytes) -> usize {
        let new_length = max(self.buffer.len(), b.length() + self.seek);
        if new_length > self.buffer.len() {
            self.re_allocate(Some(new_length * BUF_ALLOC_MULTIPLIER));
        }

        if b.length() > 0 {
            self.buffer
                .resize(max(self.seek + b.length(), self.buffer.len()), 0);

            for i in 0..b.length() {
                self.buffer[i + self.seek] = b.at(i);
            }
        }

        self.length = new_length;
        self.seek += b.length(); // Don't move this line prior to `this._length = newLength`!
        b.length()
    }

    pub fn write_str(&mut self, s: &str) -> usize {
        self.write(Bytes::new(Some(BytesFromType::Raw(s.as_bytes().to_vec()))))
    }

    pub fn read(&mut self, size: usize) -> Bytes {
        if self.seek > self.length || size == 0 {
            return Bytes::new(None); // Return empty byte
        }

        let size = if self.seek + size <= self.length {
            size
        } else {
            self.length - self.seek
        };

        let mut u8 = Vec::<u8>::with_capacity(size);
        for i in 0..size {
            u8.push(self.buffer[self.seek + i]);
        }
        self.seek += size;
        Bytes::new(Some(BytesFromType::Raw(u8)))
    }

    pub fn get_value(&self) -> Bytes {
        Bytes::new(Some(BytesFromType::Raw(self.buffer.clone())))
    }
}

pub fn bi_zero() -> Number {
    Zero::zero()
}
pub fn bi_one() -> Number {
    One::one()
}

pub fn get_u32(v: &[u8], n: usize) -> u32 {
    let p1 = v[n] as u32;
    let p2 = v[n + 1] as u32;
    let p3 = v[n + 2] as u32;
    let p4 = v[n + 3] as u32;
    p1 | (p2 << 8) | (p3 << 16) | (p4 << 24)
}

pub fn set_u8(vec: &mut [u8], n: usize, v: u8) {
    vec[n] = v;
}

pub fn set_u32(vec: &mut [u8], n: usize, v: u32) {
    vec[n] = ((v >> 24) & 0xff) as u8;
    vec[n + 1] = ((v >> 16) & 0xff) as u8;
    vec[n + 2] = ((v >> 8) & 0xff) as u8;
    vec[n + 3] = (v & 0xff) as u8;
}
