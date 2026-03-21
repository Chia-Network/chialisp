#[cfg(test)]
use rand::distr::StandardUniform;
#[cfg(test)]
use rand::prelude::Distribution;
#[cfg(test)]
use rand::Rng;

use std::borrow::Borrow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use binascii::{bin2hex, hex2bin};
use num_traits::{zero, Num};

use serde::Serialize;

#[cfg(test)]
use crate::classic::clvm::__type_compatibility__::bi_one;
use crate::classic::clvm::__type_compatibility__::{bi_zero, Bytes, BytesFromType};
use crate::classic::clvm::casts::{bigint_from_bytes, bigint_to_bytes_clvm, TConvertOption};
#[cfg(any(test, feature = "fuzz"))]
use crate::compiler::fuzz::{ExprModifier, FuzzChoice};
use crate::compiler::prims::prims;
use crate::compiler::srcloc::Srcloc;
use crate::util::{number_from_u8, u8_from_number, Number};

/// This relates to automatic generation of random sexp.
pub const MAX_SEXP_COST: usize = 15;

/// The compiler's view of SExp.
///
/// These preserve some characteristics of the source text that aren't strictly
/// required for chialisp but are useful for ergonomics and compilation.  The
/// Srcloc especially is relied on by the vscode plugin, which uses the frontend
/// entrypoints here for parsing and to surface some kinds of errors.
#[derive(Clone, Debug, Serialize)]
pub enum SExp {
    /// A native nil value "()"
    Nil(Srcloc),
    /// A cons with a left and right child.  The srcloc should span the entire
    /// content of the list, but may not depending on the construction of the
    /// list.
    Cons(Srcloc, Rc<SExp>, Rc<SExp>),
    /// Contains an integer which is presented normalized.
    Integer(Srcloc, Number),
    /// Contains a quoted string or hex constant to reproduce exactly.
    QuotedString(Srcloc, u8, Vec<u8>),
    /// Contains an identifier like atom.
    Atom(Srcloc, Vec<u8>),
}

#[cfg(test)]
pub fn random_atom_name<R: Rng + ?Sized>(rng: &mut R, min_size: usize) -> Vec<u8> {
    let mut bytevec: Vec<u8> = Vec::new();
    let mut len = 0;
    loop {
        let mut n: u8 = rng.random();
        n %= 40;
        len += 1;
        if n < 26 || len < min_size {
            bytevec.push((n % 26) + 97); // lowercase a
        } else {
            break;
        }
    }
    bytevec
}

#[cfg(test)]
pub fn random_atom<R: Rng + ?Sized>(rng: &mut R) -> SExp {
    SExp::Atom(Srcloc::start("*rng*"), random_atom_name(rng, 1))
}

#[cfg(test)]
pub fn random_sexp<R: Rng + ?Sized>(rng: &mut R, remaining: usize) -> SExp {
    if remaining < 2 {
        random_atom(rng)
    } else {
        let loc = || Srcloc::start("*rng*");
        let alternative: usize = rng.random_range(0..=2);
        match alternative {
            0 => {
                // list
                let length = rng.random_range(1..=remaining);
                let costs = vec![remaining / length; length];
                let collected_list: Vec<Rc<SExp>> = costs
                    .iter()
                    .map(|c| Rc::new(random_sexp(rng, *c)))
                    .collect();
                enlist(loc(), &collected_list)
            }
            1 => {
                // cons
                let left_cost = rng.random_range(1..=remaining);
                let right_cost = remaining - left_cost;
                SExp::Cons(
                    loc(),
                    Rc::new(random_sexp(rng, left_cost)),
                    Rc::new(random_sexp(rng, right_cost)),
                )
            }
            _ => {
                // atom
                random_atom(rng)
            }
        }
    }
}

// Thanks: https://stackoverflow.com/questions/48490049/how-do-i-choose-a-random-value-from-an-enum
#[cfg(test)]
impl Distribution<SExp> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SExp {
        random_sexp(rng, MAX_SEXP_COST)
    }
}

impl Eq for SExp {}

impl PartialEq for SExp {
    fn eq(&self, other: &Self) -> bool {
        self.equal_to(other)
    }
}

impl Display for SExp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            SExp::Nil(_) => {
                formatter.write_str("()")?;
            }
            SExp::Cons(_, a, b) => {
                formatter.write_str("(")?;
                formatter.write_str(&list_no_parens(a, b))?;
                formatter.write_str(")")?;
            }
            SExp::Integer(_, v) => {
                formatter.write_str(&v.to_string())?;
            }
            SExp::QuotedString(_, q, s) => {
                if printable(s, true) {
                    formatter.write_str("\"")?;
                    formatter.write_str(&escape_quote(*q, s))?;
                    formatter.write_str("\"")?;
                } else {
                    let vlen = s.len() * 2;
                    let mut outbuf = vec![0; vlen];
                    bin2hex(s, &mut outbuf).map_err(|_e| std::fmt::Error)?;
                    formatter.write_str("0x")?;
                    formatter.write_str(
                        std::str::from_utf8(&outbuf).expect("only hex digits expected"),
                    )?;
                }
            }
            SExp::Atom(l, a) => {
                if a.is_empty() {
                    formatter.write_str("()")?;
                } else if printable(a, false) {
                    formatter.write_str(&decode_string(a))?;
                } else {
                    formatter
                        .write_str(&SExp::Integer(l.clone(), number_from_u8(a)).to_string())?;
                }
            }
        }

        Ok(())
    }
}

impl Hash for SExp {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            SExp::Nil(l) => {
                SExp::Atom(l.clone(), Vec::new()).hash(state);
            }
            SExp::Cons(_, a, b) => {
                a.hash(state);
                b.hash(state);
            }
            SExp::Atom(_, a) => {
                a.hash(state);
            }
            SExp::QuotedString(_, _, a) => {
                a.hash(state);
            }
            SExp::Integer(_, i) => {
                u8_from_number(i.clone()).hash(state);
            }
        }
    }
}

fn make_cons(a: Rc<SExp>, b: Rc<SExp>) -> SExp {
    SExp::Cons(a.loc().ext(&b.loc()), Rc::clone(&a), Rc::clone(&b))
}

#[derive(Debug, PartialEq, Eq)]
enum SExpParseState {
    // The types of state that the Rust pre-forms can take
    Empty,
    CommentText,
    Bareword(Srcloc, Vec<u8>), //srcloc contains the file, line, column and length for the captured form
    QuotedText(Srcloc, u8, Vec<u8>),
    QuotedEscaped(Srcloc, u8, Vec<u8>),
    OpenList(Srcloc, bool),
    ParsingList(Srcloc, Rc<SExpParseState>, Vec<Rc<SExp>>, bool), // Rc<SExpParseState> is for the inner state of the list, bool is is_structured
    TermList(
        Srcloc,
        Option<Rc<SExp>>,   // this is the second value in the dot expression
        Rc<SExpParseState>, // used for inner parsing
        Vec<Rc<SExp>>,      // list content
    ),
    StartStructuredList(Srcloc),
}

#[derive(Debug, PartialEq, Eq)]
enum SExpParseResult {
    // the result of a call to parse an SExp
    Resume(SExpParseState),
    Emit(Rc<SExp>, SExpParseState),
    Error(Srcloc, String),
}

#[derive(Debug)]
enum Integral {
    Decimal,
    Hex,
    NotIntegralValue,
}

fn is_hex(s: &[u8]) -> bool {
    s.len() >= 2 && s[0] == b'0' && s[1] == b'x'
}

fn is_dec(s: &[u8]) -> bool {
    let mut first = true;
    let mut dec = true;

    for ch in s {
        if first && *ch == b'-' {
            // Nothing
        } else if !(*ch >= b'0' && *ch <= b'9') {
            dec = false;
            break;
        }
        first = false;
    }

    dec && *s != vec![b'-']
}

fn matches_integral(s: &[u8]) -> Integral {
    if is_hex(s) {
        Integral::Hex
    } else if is_dec(s) {
        Integral::Decimal
    } else {
        Integral::NotIntegralValue
    }
}

fn normalize_int(v: Vec<u8>, base: u32) -> Number {
    let s = Bytes::new(Some(BytesFromType::Raw(v))).decode();
    Number::from_str_radix(&s, base).unwrap()
}

// Hex values are _not_ numbers, they're byte strings expressed in hexadecimal
// while they correspond numerically, integral constants are byte-padded to the
// left to retain their sign and hex constants are considered unsigned so _not_
// padded.
fn from_hex(l: Srcloc, v: &[u8]) -> SExp {
    let mut result = vec![0; (v.len() - 1) / 2];
    let mut hex_const;
    // This assigns a new reference so the vec is copied only when we need
    // to pad it.
    let v_ref = if v.len() % 2 == 1 {
        hex_const = v.to_vec();
        hex_const.insert(2, b'0');
        &hex_const[2..]
    } else {
        &v[2..]
    };
    hex2bin(v_ref, &mut result).ok();
    SExp::QuotedString(l, b'x', result)
}

fn make_atom(l: Srcloc, v: Vec<u8>) -> SExp {
    let alen = v.len();
    if alen > 1 && v[0] == b'#' {
        // Search prims for appropriate primitive
        let want_name = v[1..].to_vec();
        for p in prims() {
            if want_name == p.0 {
                return p.1;
            }
        }

        // Fallback (probably)
        SExp::Atom(l, v[1..].to_vec())
    } else {
        match matches_integral(&v) {
            Integral::Hex => from_hex(l, &v),
            Integral::Decimal => {
                let intval = normalize_int(v, 10);
                if intval == bi_zero() {
                    SExp::Nil(l)
                } else {
                    SExp::Integer(l, intval)
                }
            }
            Integral::NotIntegralValue => SExp::Atom(l, v),
        }
    }
}

pub fn enlist(l: Srcloc, v: &[Rc<SExp>]) -> SExp {
    let mut result = SExp::Nil(l);
    for i_reverse in 0..v.len() {
        let i = v.len() - i_reverse - 1;
        result = make_cons(Rc::clone(&v[i]), Rc::new(result));
    }
    result
}

// this function takes a ParseState and returns an Emit ParseResult which contains the ParseState
fn emit(a: Rc<SExp>, current_state: SExpParseState) -> SExpParseResult {
    SExpParseResult::Emit(a, current_state)
}

fn error(l: Srcloc, t: &str) -> SExpParseResult {
    SExpParseResult::Error(l, t.to_string())
}

// this function takes a ParseState and returns a Resume ParseResult which contains the ParseState
fn resume(current_state: SExpParseState) -> SExpParseResult {
    SExpParseResult::Resume(current_state)
}

fn escape_quote(q: u8, s: &[u8]) -> String {
    let mut res: Vec<char> = Vec::new();
    let _: Vec<()> = s
        .iter()
        .map(|ch| {
            if *ch == q {
                res.push('\\');
            }
            res.push(*ch as char);
        })
        .collect();
    res.into_iter().collect()
}

fn list_no_parens(a: &SExp, b: &SExp) -> String {
    if b.nilp() {
        a.to_string()
    } else {
        match b {
            SExp::Cons(_, b, c) => a.to_string() + " " + &list_no_parens(b, c),
            _ => a.to_string() + " . " + &b.to_string(),
        }
    }
}

pub fn decode_string(v: &[u8]) -> String {
    String::from_utf8_lossy(v).as_ref().to_string()
}

pub fn printable(a: &[u8], quoted: bool) -> bool {
    !a.iter().any(|ch| {
        *ch < 32
            || *ch > 126
            || (!quoted && ((*ch as char).is_ascii_whitespace() || *ch == b'\''))
            || *ch == b'"'
            || *ch == b'\\'
    })
}

impl SExp {
    pub fn loc(&self) -> Srcloc {
        match self {
            SExp::Nil(l) => l.clone(),
            SExp::Cons(l, _, _) => l.clone(),
            SExp::Integer(l, _) => l.clone(),
            SExp::QuotedString(l, _, _) => l.clone(),
            SExp::Atom(l, _) => l.clone(),
        }
    }

    pub fn with_loc(&self, loc: Srcloc) -> SExp {
        match self {
            SExp::Nil(_) => SExp::Nil(loc),
            SExp::Cons(_, a, b) => SExp::Cons(loc, a.clone(), b.clone()),
            SExp::Integer(_, i) => SExp::Integer(loc, i.clone()),
            SExp::QuotedString(_, q, s) => SExp::QuotedString(loc, *q, s.clone()),
            SExp::Atom(_, a) => SExp::Atom(loc, a.clone()),
        }
    }

    pub fn atom_from_string(loc: Srcloc, s: &str) -> SExp {
        SExp::Atom(loc, s.as_bytes().to_vec())
    }

    pub fn atom_from_vec(loc: Srcloc, s: &[u8]) -> SExp {
        SExp::Atom(loc, s.to_vec())
    }

    pub fn quoted_from_string(loc: Srcloc, s: &str) -> SExp {
        SExp::QuotedString(loc, b'\"', s.as_bytes().to_vec())
    }

    pub fn nilp(&self) -> bool {
        let bizero: Number = zero();
        match self {
            SExp::Nil(_) => true,
            SExp::QuotedString(_, _, v) => v.is_empty(),
            SExp::Integer(_, i) => *i == bizero,
            SExp::Atom(_, a) => a.is_empty(),
            _ => false,
        }
    }

    pub fn listp(&self) -> bool {
        matches!(self, SExp::Nil(_) | SExp::Cons(_, _, _))
    }

    pub fn cons_fst(&self) -> Rc<SExp> {
        match self {
            SExp::Cons(_, a, _) => Rc::clone(a),
            _ => Rc::new(SExp::Nil(self.loc())),
        }
    }

    pub fn cons_snd(&self) -> Rc<SExp> {
        match self {
            SExp::Cons(_, _, b) => Rc::clone(b),
            _ => Rc::new(SExp::Nil(self.loc())),
        }
    }

    pub fn encode_mut(&self, v: &mut Vec<u8>) {
        match self {
            SExp::Nil(_) => v.push(0x80),
            SExp::Cons(_, a, b) => {
                v.push(0xff);
                a.encode_mut(v);
                b.encode_mut(v);
            }
            SExp::Integer(_, i) => {
                let mut bi_bytes = bigint_to_bytes_clvm(i).data().to_vec();

                v.append(&mut bi_bytes);
            }
            SExp::QuotedString(l, _, s) => {
                SExp::Integer(l.clone(), number_from_u8(s)).encode_mut(v);
            }
            SExp::Atom(l, a) => {
                SExp::Integer(l.clone(), number_from_u8(a)).encode_mut(v);
            }
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        self.encode_mut(&mut v);
        v
    }

    pub fn to_bigint(&self) -> Option<Number> {
        match self {
            SExp::Nil(_) => Some(zero()),
            SExp::Integer(_, v) => Some(v.clone()),
            SExp::QuotedString(_, _, v) => Some(bigint_from_bytes(
                &Bytes::new(Some(BytesFromType::Raw(v.to_vec()))),
                Some(TConvertOption { signed: true }),
            )),
            SExp::Atom(_, v) => Some(bigint_from_bytes(
                &Bytes::new(Some(BytesFromType::Raw(v.to_vec()))),
                Some(TConvertOption { signed: true }),
            )),
            _ => None,
        }
    }

    pub fn atomize(&self) -> SExp {
        match self {
            SExp::Integer(l, i) => SExp::Atom(l.clone(), u8_from_number(i.clone())),
            SExp::QuotedString(l, _, a) => SExp::Atom(l.clone(), a.clone()),
            _ => self.clone(),
        }
    }

    pub fn equal_to(&self, other: &SExp) -> bool {
        if self.nilp() && other.nilp() {
            true
        } else if self.nilp() || other.nilp() {
            false
        } else {
            match (self, other) {
                (SExp::Cons(_, r, s), SExp::Cons(_, t, u)) => r.equal_to(t) && s.equal_to(u),
                (SExp::Cons(_, _, _), _) => false,
                (_, SExp::Cons(_, _, _)) => false,
                (SExp::Integer(l, a), b) => SExp::Atom(l.clone(), u8_from_number(a.clone())) == *b,
                (SExp::QuotedString(l, _, a), b) => SExp::Atom(l.clone(), a.clone()) == *b,
                (SExp::Nil(l), b) => SExp::Atom(l.clone(), Vec::new()) == *b,
                (SExp::Atom(_, _), SExp::Integer(_, _)) => other == self,
                (SExp::Atom(_, _), SExp::QuotedString(_, _, _)) => other == self,
                (SExp::Atom(_, _), SExp::Nil(_)) => other == self,
                (SExp::Atom(_, a), SExp::Atom(_, b)) => a == b,
            }
        }
    }

    pub fn proper_list(&self) -> Option<Vec<SExp>> {
        let mut res = Vec::new();
        let mut track = Rc::new(self.clone());

        loop {
            if track.nilp() {
                return Some(res);
            } else {
                match track.borrow() {
                    SExp::Cons(_, l, r) => {
                        let borrowed: &SExp = l.borrow();
                        let cloned: SExp = borrowed.clone();
                        res.push(cloned);
                        track = r.clone();
                    }
                    _ => {
                        return None;
                    }
                }
            }
        }
    }

    pub fn get_number(&self) -> Result<Number, (Srcloc, String)> {
        match self {
            SExp::Integer(_, i) => Ok(i.clone()),
            SExp::Atom(_, v) => Ok(number_from_u8(v)),
            SExp::QuotedString(_, _, v) => Ok(number_from_u8(v)),
            SExp::Nil(_) => Ok(bi_zero()),
            _ => Err((self.loc(), format!("wanted atom got cons cell {self}"))),
        }
    }
}

fn restructure_list(mut this_list: Vec<Rc<SExp>>, srcloc: Srcloc) -> Rc<SExp> {
    // Check if the vector is empty
    if this_list.len() == 1 {
        return Rc::clone(&this_list[0]);
    }
    if this_list.is_empty() {
        return Rc::new(SExp::Nil(srcloc.clone()));
    }
    // Remove and get the middle element as the root
    let mid_index = this_list.len() / 2;
    let left_subtree = restructure_list(this_list.drain(..mid_index).collect(), srcloc.clone());
    let right_subtree = restructure_list(this_list, srcloc.clone());

    Rc::new(make_cons(left_subtree, right_subtree))
}

fn parse_sexp_step(loc: Srcloc, current_state: &SExpParseState, this_char: u8) -> SExpParseResult {
    // switch on our state
    match current_state {
        SExpParseState::Empty => match this_char as char {
            // we are not currently in a list
            '(' => resume(SExpParseState::OpenList(loc, false)), // move to OpenList state
            '\n' => resume(SExpParseState::Empty),               // new line, same state
            ';' => resume(SExpParseState::CommentText),
            ')' => error(loc, "Too many close parens"),
            '"' => resume(SExpParseState::QuotedText(loc, b'"', Vec::new())), // match on "
            '\'' => resume(SExpParseState::QuotedText(loc, b'\'', Vec::new())), // match on '
            '#' => resume(SExpParseState::StartStructuredList(loc)), // initiating a structured list
            ch => {
                if char::is_whitespace(ch) {
                    resume(SExpParseState::Empty)
                } else {
                    resume(SExpParseState::Bareword(loc, vec![this_char])) // start of a word - could be an atom or a keyword - the compiler will decide
                }
            }
        },
        // t is a Vec of the previous characters in this comment string
        SExpParseState::CommentText => match this_char as char {
            '\n' => resume(SExpParseState::Empty),
            _ => resume(SExpParseState::CommentText),
        },
        // we currently processing a new word
        SExpParseState::Bareword(srcloc, word_so_far) => {
            if char::is_whitespace(this_char as char) {
                // we've found a space, so it's the end of a word
                emit(
                    Rc::new(make_atom(srcloc.clone(), word_so_far.to_vec())),
                    SExpParseState::Empty,
                )
            } else {
                // otherwise add letter to word
                let mut word_copy = word_so_far.to_vec();
                word_copy.push(this_char);
                resume(SExpParseState::Bareword(srcloc.ext(&loc), word_copy))
            }
        }
        SExpParseState::QuotedText(srcloc, term, t) => {
            if this_char == b'\\' {
                // if we have a character escape then copy the character directly
                resume(SExpParseState::QuotedEscaped(
                    srcloc.clone(),
                    *term,
                    t.to_vec(),
                ))
            } else if this_char == *term {
                // otherwise check if it's the terminating character (either ' or ")
                emit(
                    Rc::new(SExp::QuotedString(srcloc.ext(&loc), *term, t.to_vec())), // add quoted string to parent list
                    SExpParseState::Empty,
                )
            } else {
                // otherwise copy the character
                let mut tcopy = t.to_vec();
                tcopy.push(this_char);
                resume(SExpParseState::QuotedText(srcloc.clone(), *term, tcopy))
            }
        }
        // copy the character the quoted text because we have put the escape character first
        SExpParseState::QuotedEscaped(srcloc, term, t) => {
            let mut tcopy = t.to_vec();
            tcopy.push(this_char);
            resume(SExpParseState::QuotedText(srcloc.clone(), *term, tcopy))
        }
        SExpParseState::OpenList(srcloc, is_structured) => match this_char as char {
            // we are beginning a new list
            ')' => emit(Rc::new(SExp::Nil(srcloc.ext(&loc))), SExpParseState::Empty), // create a Nil object
            '.' => error(loc, "Dot can't appear directly after begin paren"),
            _ => match parse_sexp_step(loc.clone(), &SExpParseState::Empty, this_char) {
                // fetch result of parsing as if we were in empty state
                SExpParseResult::Emit(o, current_state) => resume(SExpParseState::ParsingList(
                    // we found an object, resume processing
                    srcloc.ext(&loc),
                    Rc::new(current_state), // captured state from our pretend empty state
                    vec![o],
                    *is_structured,
                )),
                SExpParseResult::Resume(current_state) => resume(SExpParseState::ParsingList(
                    // we're still reading the object, resume processing
                    srcloc.ext(&loc),
                    Rc::new(current_state), // captured state from our pretend empty state
                    Vec::new(),
                    *is_structured,
                )),
                SExpParseResult::Error(l, e) => SExpParseResult::Error(l, e), // propagate error
            },
        },
        // We are in the middle of a list currently
        SExpParseState::ParsingList(srcloc, pp, list_content, is_structured) => {
            // pp is the captured inside-list state we received from OpenList
            match (this_char as char, pp.borrow(), is_structured) {
                ('.', SExpParseState::Empty, false) => resume(SExpParseState::TermList(
                    // dot notation showing cons cell
                    srcloc.ext(&loc),
                    None,
                    Rc::new(SExpParseState::Empty), // nested state is empty
                    list_content.to_vec(),
                )),
                ('.', SExpParseState::Empty, true) => {
                    error(loc, "Dot expressions disallowed in structured lists")
                }
                (')', SExpParseState::Empty, _) => {
                    if *is_structured {
                        emit(
                            // close list and emit it upwards as a complete entity
                            restructure_list(list_content.to_vec(), srcloc.clone()),
                            SExpParseState::Empty,
                        )
                    } else {
                        emit(
                            // close list and emit it upwards as a complete entity
                            Rc::new(enlist(srcloc.clone(), list_content)),
                            SExpParseState::Empty,
                        )
                    }
                }
                (')', SExpParseState::Bareword(l, t), _) => {
                    // you've reached the end of the word AND the end of the list, close list and emit upwards
                    // TODO: check bool and rearrange here
                    let parsed_atom = make_atom(l.clone(), t.to_vec());
                    let mut updated_list = list_content.to_vec();
                    updated_list.push(Rc::new(parsed_atom));
                    if *is_structured {
                        emit(
                            // close list and emit it upwards as a complete entity
                            restructure_list(updated_list, srcloc.clone()),
                            SExpParseState::Empty,
                        )
                    } else {
                        emit(
                            // close list and emit it upwards as a complete entity
                            Rc::new(enlist(srcloc.clone(), &updated_list)),
                            SExpParseState::Empty,
                        )
                    }
                }
                // analyze this character using the mock "inner state" stored in pp
                (_, _, _) => match parse_sexp_step(loc.clone(), pp.borrow(), this_char) {
                    //
                    SExpParseResult::Emit(o, current_state) => {
                        // add result of parse_sexp_step to our list
                        let mut list_copy = list_content.clone();
                        list_copy.push(o);
                        let result = SExpParseState::ParsingList(
                            srcloc.ext(&loc),
                            Rc::new(current_state),
                            list_copy,
                            *is_structured,
                        );
                        resume(result)
                    }
                    SExpParseResult::Resume(rp) => resume(SExpParseState::ParsingList(
                        // we aren't finished reading in our nested state
                        srcloc.ext(&loc),
                        Rc::new(rp), // store the returned state from parse_sexp_step in pp
                        list_content.to_vec(),
                        *is_structured,
                    )),
                    SExpParseResult::Error(l, e) => SExpParseResult::Error(l, e), // propagate error upwards
                },
            }
        }

        // if we're not in a comment and have already found a parsed second word for this dot expression
        SExpParseState::TermList(srcloc, Some(parsed), pp, list_content) => {
            match (this_char as char, pp.borrow()) {
                (')', SExpParseState::Empty) => {
                    // if we see a `)` then we're ready to close this list
                    let mut list_copy = list_content.to_vec();
                    match list_copy.pop() {
                        Some(v) => {
                            let new_tail = make_cons(v, Rc::clone(parsed));
                            if list_copy.is_empty() {
                                emit(Rc::new(new_tail), SExpParseState::Empty)
                            } else {
                                let mut result_list = new_tail;
                                for item in list_copy.iter().rev() {
                                    result_list = make_cons(Rc::clone(item), Rc::new(result_list));
                                }
                                emit(Rc::new(result_list), SExpParseState::Empty)
                                // emit the resultant list
                            }
                        }
                        None => error(loc, "Dot as first element of list?"),
                    }
                }
                _ => match parse_sexp_step(loc.clone(), pp.borrow(), this_char) {
                    // nothing should be emitted as we're a term list with an object found
                    SExpParseResult::Emit(_, _current_state) => {
                        error(loc, "found object during termlist")
                    }
                    // resume means it didn't finish parsing yet, so store inner state and keep going
                    SExpParseResult::Resume(current_state) => {
                        match current_state {
                            SExpParseState::Empty | SExpParseState::CommentText => {
                                resume(SExpParseState::TermList(
                                    srcloc.ext(&loc),
                                    Some(parsed.clone()),
                                    Rc::new(current_state), // store our partial inner parsestate in pp
                                    list_content.to_vec(),
                                ))
                            }
                            _ => error(loc, "Illegal state during term list."),
                        }
                    }
                    SExpParseResult::Error(l, e) => SExpParseResult::Error(l, e),
                },
            }
        }
        // we are passing a dot-expression (x . y) and not in a comment and don't have an object already discovered
        SExpParseState::TermList(srcloc, None, pp, list_content) => {
            // pp is the inner parsestate inside the dot-expressions
            match (this_char as char, pp.borrow()) {
                //match based on current character and inner state
                ('.', SExpParseState::Empty) => {
                    // if we aren't in a word and we see another dot that's illegal
                    error(loc, "Multiple dots in list notation are illegal")
                }
                (')', SExpParseState::Empty) => {
                    // attempt to close the list
                    if list_content.len() == 1 {
                        emit(Rc::clone(&list_content[0]), SExpParseState::Empty)
                    } else {
                        emit(
                            Rc::new(enlist(srcloc.ext(&loc), list_content)),
                            SExpParseState::Empty,
                        )
                    }
                }
                (')', SExpParseState::Bareword(l, t)) => {
                    let parsed_atom = make_atom(l.clone(), t.to_vec());
                    let mut list_copy = list_content.to_vec();
                    match list_copy.pop() {
                        Some(v) => {
                            let new_tail = make_cons(v, Rc::new(parsed_atom));
                            if list_copy.is_empty() {
                                emit(Rc::new(new_tail), SExpParseState::Empty)
                            } else {
                                let mut result_list = new_tail;
                                for item in list_copy.iter().rev() {
                                    result_list = make_cons(Rc::clone(item), Rc::new(result_list));
                                }
                                emit(Rc::new(result_list), SExpParseState::Empty)
                            }
                        }
                        None => error(loc, "Dot as first element of list?"),
                    }
                }
                // if we see anything other than ')' or '.' parse it as if we were in empty state
                (_, _) => match parse_sexp_step(loc.clone(), pp.borrow(), this_char) {
                    SExpParseResult::Emit(parsed_object, _current_state) => {
                        resume(SExpParseState::TermList(
                            loc,
                            Some(parsed_object), // assert parsed_object is not None and then store it in parsed_list
                            Rc::new(SExpParseState::Empty),
                            list_content.clone(),
                        ))
                    }
                    // resume means it didn't finish parsing yet, so store inner state and keep going
                    SExpParseResult::Resume(current_state) => resume(SExpParseState::TermList(
                        srcloc.ext(&loc),
                        None,
                        Rc::new(current_state), // store our partial inner parsestate in pp
                        list_content.to_vec(),
                    )),
                    SExpParseResult::Error(l, e) => SExpParseResult::Error(l, e),
                },
            }
        }
        SExpParseState::StartStructuredList(l) => {
            let new_srcloc = l.ext(&loc);
            match this_char as char {
                '(' => resume(SExpParseState::ParsingList(
                    // go into a ParsingList
                    new_srcloc,
                    Rc::new(SExpParseState::Empty), // we have no inner state
                    Vec::new(),
                    true, // note that this is a special StructuredList to be processed later
                )),
                _ => parse_sexp_step(
                    // if we don't see a '(' then process it as if the preceding '#' was part of a bareword
                    loc.clone(),
                    &SExpParseState::Bareword(loc, vec![b'#']),
                    this_char,
                ),
            }
        } // SExpParseState::StartStructuredList(_) => error(loc, "Missing srcloc"),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParsePartialResult {
    // we support compiling multiple things at once, keep these in a Vec
    // at the moment this will almost certainly only return 1 thing
    res: Vec<Rc<SExp>>,
    srcloc: Srcloc,
    parse_state: SExpParseState,
}

impl ParsePartialResult {
    pub fn new(srcloc: Srcloc) -> Self {
        ParsePartialResult {
            res: Default::default(),
            srcloc,
            parse_state: SExpParseState::Empty,
        }
    }
    pub fn push(&mut self, this_char: u8) -> Result<(), (Srcloc, String)> {
        let next_location = self.srcloc.clone().advance(this_char);

        // call parse_sexp_step for current character
        // it will return a ParseResult which contains the new ParseState
        match parse_sexp_step(self.srcloc.clone(), &self.parse_state, this_char) {
            // catch error and propagate itupwards
            SExpParseResult::Error(l, e) => {
                return Err((l, e));
            }
            // Keep parsing
            SExpParseResult::Resume(new_parse_state) => {
                self.srcloc = next_location;
                self.parse_state = new_parse_state;
            }
            // End of list (top level compile object), but not necessarily end of file
            SExpParseResult::Emit(o, new_parse_state) => {
                self.srcloc = next_location;
                self.parse_state = new_parse_state;
                self.res.push(o);
            }
        }

        Ok(())
    }

    pub fn finalize(self) -> Result<Vec<Rc<SExp>>, (Srcloc, String)> {
        // depending on the state when we finished return Ok or Err enums
        match self.parse_state {
            SExpParseState::Empty => Ok(self.res),
            SExpParseState::Bareword(l, t) => Ok(vec![Rc::new(make_atom(l, t))]),
            SExpParseState::CommentText => Ok(self.res),
            SExpParseState::QuotedText(l, _, _) => {
                Err((l, "unterminated quoted string".to_string()))
            }
            SExpParseState::QuotedEscaped(l, _, _) => {
                Err((l, "unterminated quoted string with escape".to_string()))
            }
            SExpParseState::OpenList(l, _) => Err((l, "Unterminated list (empty)".to_string())),
            SExpParseState::ParsingList(l, _, _, _) => {
                Err((l, "Unterminated mid list".to_string()))
            }
            SExpParseState::TermList(l, _, _, _) => Err((l, "Unterminated tail list".to_string())),
            SExpParseState::StartStructuredList(l) => {
                Err((l, "Unclosed structured list".to_string()))
            }
        }
    }
}

fn parse_sexp_inner<I>(start: Srcloc, s: I) -> Result<Vec<Rc<SExp>>, (Srcloc, String)>
where
    I: Iterator<Item = u8>,
{
    let mut partial_result = ParsePartialResult::new(start);

    // Loop through all the characters
    for this_char in s {
        partial_result.push(this_char)?;
    }

    partial_result.finalize()
}

///
/// Entrypoint for parsing chialisp input.
/// Called from compiler.rs
///
/// This produces Rc<SExp>, where SExp is described above.
///
pub fn parse_sexp<I>(start: Srcloc, input: I) -> Result<Vec<Rc<SExp>>, (Srcloc, String)>
where
    I: Iterator<Item = u8>,
{
    parse_sexp_inner(start, input)
}

#[cfg(test)]
fn check_parser_for_intermediate_result(
    parser: &mut ParsePartialResult,
    s: &str,
    desired: SExpParseState,
) {
    for this_char in s.bytes() {
        parser.push(this_char).unwrap();
    }
    assert_eq!(parser.parse_state, desired);
}

#[cfg(test)]
fn srcloc_range(name: &Rc<String>, start: usize, end: usize) -> Srcloc {
    Srcloc::new(name.clone(), 1, start).ext(&Srcloc::new(name.clone(), 1, end))
}

#[test]
fn test_tricky_parser_tail_01() {
    let testname = Rc::new("*test*".to_string());
    let loc = Srcloc::start(&testname);
    let mut parser = ParsePartialResult::new(loc.clone());
    check_parser_for_intermediate_result(
        &mut parser,
        "(1 . x",
        SExpParseState::TermList(
            srcloc_range(&testname, 1, 6),
            None,
            Rc::new(SExpParseState::Bareword(
                srcloc_range(&testname, 6, 6),
                vec![b'x'],
            )),
            vec![Rc::new(SExp::Integer(
                srcloc_range(&testname, 2, 2),
                bi_one(),
            ))],
        ),
    );

    parser.push(b')').expect("should complete");
    assert_eq!(
        parser.finalize(),
        Ok(vec![Rc::new(SExp::Cons(
            srcloc_range(&testname, 1, 7),
            Rc::new(SExp::Integer(srcloc_range(&testname, 2, 2), bi_one())),
            Rc::new(SExp::Atom(srcloc_range(&testname, 6, 7), b"x".to_vec()))
        ))])
    );
}

#[test]
fn test_tricky_parser_tail_02() {
    let testname = Rc::new("*test*".to_string());
    let loc = Srcloc::start(&testname);
    let mut parser = ParsePartialResult::new(loc.clone());
    check_parser_for_intermediate_result(
        &mut parser,
        "(1 . ()",
        SExpParseState::TermList(
            srcloc_range(&testname, 7, 7),
            Some(Rc::new(SExp::Nil(srcloc_range(&testname, 6, 7)))),
            Rc::new(SExpParseState::Empty),
            vec![Rc::new(SExp::Integer(
                srcloc_range(&testname, 2, 2),
                bi_one(),
            ))],
        ),
    );

    parser.push(b')').expect("should complete");
    assert_eq!(
        parser.finalize(),
        Ok(vec![Rc::new(SExp::Cons(
            srcloc_range(&testname, 1, 7),
            Rc::new(SExp::Integer(srcloc_range(&testname, 2, 2), bi_one())),
            Rc::new(SExp::Nil(srcloc_range(&testname, 6, 7)))
        ))])
    );
}

#[test]
fn test_tricky_parser_tail_03() {
    let testname = Rc::new("*test*".to_string());
    let loc = Srcloc::start(&testname);
    let mut parser = ParsePartialResult::new(loc.clone());
    check_parser_for_intermediate_result(
        &mut parser,
        "(1 . () ;; Test\n",
        SExpParseState::TermList(
            srcloc_range(&testname, 7, 16),
            Some(Rc::new(SExp::Nil(srcloc_range(&testname, 6, 7)))),
            Rc::new(SExpParseState::Empty),
            vec![Rc::new(SExp::Integer(
                srcloc_range(&testname, 2, 2),
                bi_one(),
            ))],
        ),
    );

    parser.push(b')').expect("should complete");
    assert_eq!(
        parser.finalize(),
        Ok(vec![Rc::new(SExp::Cons(
            srcloc_range(&testname, 1, 7),
            Rc::new(SExp::Integer(srcloc_range(&testname, 2, 2), bi_one())),
            Rc::new(SExp::Nil(srcloc_range(&testname, 6, 7)))
        ))])
    );
}

// This is a trait that generates a haskell-like ad-hoc type from the user's
// construction of NodeSel and ThisNode.
// the result is transformed into a NodeSel tree of NodePtr if it can be.
// The type of the result is an ad-hoc shape derived from the shape of the
// original request.
//
// This mirrors code in src/classic/clvm/sexp.rs
//
// It's a nicer way of modelling matches that will overtake bespoke code for a lot
// of things.
#[derive(Debug, Clone)]
pub enum NodeSel<T, U> {
    Cons(T, U),
}

#[derive(Debug, Clone)]
pub enum First<T> {
    Here(T),
}

#[derive(Debug, Clone)]
pub enum Rest<T> {
    Here(T),
}

#[derive(Debug, Clone)]
pub struct ThisNode;

pub enum Atom<T> {
    Here(T),
}

pub enum AtomValue<T> {
    Here(T),
}

pub trait SelectNode<T, E> {
    fn select_nodes(&self, s: Rc<SExp>) -> Result<T, E>;
}

impl<E> SelectNode<Rc<SExp>, E> for ThisNode {
    fn select_nodes(&self, s: Rc<SExp>) -> Result<Rc<SExp>, E> {
        Ok(s)
    }
}

impl SelectNode<(Srcloc, Vec<u8>), (Srcloc, String)> for Atom<()> {
    fn select_nodes(&self, s: Rc<SExp>) -> Result<(Srcloc, Vec<u8>), (Srcloc, String)> {
        if let SExp::Atom(loc, name) = s.borrow() {
            return Ok((loc.clone(), name.clone()));
        }

        Err((s.loc(), "Not an atom".to_string()))
    }
}

impl SelectNode<Srcloc, (Srcloc, String)> for Atom<&str> {
    fn select_nodes(&self, s: Rc<SExp>) -> Result<Srcloc, (Srcloc, String)> {
        let Atom::Here(name) = self;
        if let Ok((l, n)) = Atom::Here(()).select_nodes(s.clone()) {
            if n == name.as_bytes() {
                return Ok(l);
            }
        }

        Err((s.loc(), format!("Not an atom named {name}")))
    }
}

impl<const N: usize> SelectNode<Srcloc, (Srcloc, String)> for AtomValue<&[u8; N]> {
    fn select_nodes(&self, s: Rc<SExp>) -> Result<Srcloc, (Srcloc, String)> {
        let AtomValue::Here(name) = self;
        match s.borrow() {
            SExp::Nil(l) => {
                if name.is_empty() {
                    return Ok(l.clone());
                }
            }
            SExp::Atom(l, n) => {
                if n == name {
                    return Ok(l.clone());
                }
            }
            SExp::QuotedString(l, _, n) => {
                if n == name {
                    return Ok(l.clone());
                }
            }
            SExp::Integer(l, i) => {
                if &u8_from_number(i.clone()) == name {
                    return Ok(l.clone());
                }
            }
            _ => {}
        }

        Err((s.loc(), format!("Not an atom with content {name:?}")))
    }
}

impl SelectNode<(Srcloc, Vec<u8>), (Srcloc, String)> for AtomValue<()> {
    fn select_nodes(&self, s: Rc<SExp>) -> Result<(Srcloc, Vec<u8>), (Srcloc, String)> {
        let AtomValue::Here(name) = self;
        match s.borrow() {
            SExp::Nil(l) => {
                return Ok((l.clone(), vec![]));
            }
            SExp::Atom(l, n) => {
                return Ok((l.clone(), n.clone()));
            }
            SExp::QuotedString(l, _, n) => {
                return Ok((l.clone(), n.clone()));
            }
            SExp::Integer(l, i) => {
                let u8_vec = u8_from_number(i.clone());
                return Ok((l.clone(), u8_vec));
            }
            _ => {}
        }

        Err((s.loc(), format!("Not an atom with content {name:?}")))
    }
}

impl<E> SelectNode<(), E> for () {
    fn select_nodes(&self, _n: Rc<SExp>) -> Result<(), E> {
        Ok(())
    }
}

impl<R, T, E> SelectNode<First<T>, E> for First<R>
where
    R: SelectNode<T, E> + Clone,
    E: From<(Srcloc, String)>,
{
    fn select_nodes(&self, s: Rc<SExp>) -> Result<First<T>, E> {
        let First::Here(f) = &self;
        let NodeSel::Cons(first, ()) = NodeSel::Cons(f.clone(), ()).select_nodes(s)?;
        Ok(First::Here(first))
    }
}

impl<R, T, E> SelectNode<Rest<T>, E> for Rest<R>
where
    R: SelectNode<T, E> + Clone,
    E: From<(Srcloc, String)>,
{
    fn select_nodes(&self, s: Rc<SExp>) -> Result<Rest<T>, E> {
        let Rest::Here(f) = &self;
        let NodeSel::Cons((), rest) = NodeSel::Cons((), f.clone()).select_nodes(s)?;
        Ok(Rest::Here(rest))
    }
}

impl<R, S, T, U, E> SelectNode<NodeSel<T, U>, E> for NodeSel<R, S>
where
    R: SelectNode<T, E>,
    S: SelectNode<U, E>,
    E: From<(Srcloc, String)>,
{
    fn select_nodes(&self, s: Rc<SExp>) -> Result<NodeSel<T, U>, E> {
        let NodeSel::Cons(my_left, my_right) = &self;
        if let SExp::Cons(_, l, r) = s.borrow() {
            let first = my_left.select_nodes(l.clone())?;
            let rest = my_right.select_nodes(r.clone())?;
            Ok(NodeSel::Cons(first, rest))
        } else {
            Err(E::from((s.loc(), "not a cons".to_string())))
        }
    }
}

//
// Fuzzing support for SExp
//
#[cfg(any(test, feature = "fuzz"))]
fn find_in_structure_inner(
    parents: &mut Vec<Rc<SExp>>,
    structure: Rc<SExp>,
    target: &Rc<SExp>,
) -> bool {
    if let SExp::Cons(_, a, b) = structure.borrow() {
        parents.push(structure.clone());
        if find_in_structure_inner(parents, a.clone(), target) {
            return true;
        }
        if find_in_structure_inner(parents, b.clone(), target) {
            return true;
        }

        parents.pop();
    }

    structure == *target
}

#[cfg(any(test, feature = "fuzz"))]
pub fn extract_atom_replacement<Expr: Clone>(
    myself: &Expr,
    a: &[u8],
) -> Option<FuzzChoice<Expr, Vec<u8>>> {
    if a.starts_with(b"${") && a.ends_with(b"}") {
        if let Some(c_idx) = a.iter().position(|&c| c == b':') {
            return Some(FuzzChoice {
                tag: a[c_idx + 1..a.len() - 1].to_vec(),
                atom: myself.clone(),
            });
        }
    }

    None
}

#[cfg(any(test, feature = "fuzz"))]
impl ExprModifier for Rc<SExp> {
    type Expr = Self;
    type Tag = Vec<u8>;

    fn find_waiters(&self, waiters: &mut Vec<FuzzChoice<Self::Expr, Self::Tag>>) {
        match self.borrow() {
            SExp::Cons(_, a, b) => {
                a.find_waiters(waiters);
                b.find_waiters(waiters);
            }
            SExp::Atom(_, a) => {
                if let Some(r) = extract_atom_replacement(self, a) {
                    waiters.push(r);
                }
            }
            _ => {}
        }
    }

    fn replace_node(&self, to_replace: &Self::Expr, new_value: Self::Expr) -> Self::Expr {
        if let SExp::Cons(l, a, b) = self.borrow() {
            let new_a = a.replace_node(to_replace, new_value.clone());
            let new_b = b.replace_node(to_replace, new_value.clone());
            if Rc::as_ptr(&new_a) != Rc::as_ptr(a) || Rc::as_ptr(&new_b) != Rc::as_ptr(b) {
                return Rc::new(SExp::Cons(l.clone(), new_a, new_b));
            }
        }

        if self == to_replace {
            return new_value;
        }

        self.clone()
    }

    fn find_in_structure(&self, target: &Self::Expr) -> Option<Vec<Self::Expr>> {
        let mut parents = Vec::new();
        if find_in_structure_inner(&mut parents, self.clone(), target) {
            Some(parents)
        } else {
            None
        }
    }
}
