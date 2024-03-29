use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;

use num_bigint::ToBigInt;
use num_traits::ToPrimitive;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};

use crate::compiler::clvm::PrimOverride;
use crate::compiler::comptypes::{CompileErr, CompilerOpts};
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::{decode_string, printable, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::util::{number_from_u8, Number};

// If the bodyform represents a constant, only match a quoted string.
fn match_quoted_string(body: Rc<SExp>) -> Result<(Srcloc, Vec<u8>), CompileErr> {
    match body.borrow() {
        SExp::QuotedString(_, b'x', _) => {}
        SExp::QuotedString(al, _, an) => return Ok((al.clone(), an.clone())),
        _ => {}
    }

    Err(CompileErr(body.loc(), "string required".to_string()))
}

fn match_atom(body: Rc<SExp>) -> Result<(Srcloc, Vec<u8>), CompileErr> {
    if let SExp::Atom(al, an) = body.borrow() {
        return Ok((al.clone(), an.clone()));
    }
    Err(CompileErr(body.loc(), "atom required".to_string()))
}

enum MatchedNumber {
    MatchedInt(Srcloc, Number),
    MatchedHex(Srcloc, Vec<u8>),
}

fn match_number(body: Rc<SExp>) -> Result<MatchedNumber, CompileErr> {
    match body.borrow() {
        SExp::Integer(il, n) => {
            return Ok(MatchedNumber::MatchedInt(il.clone(), n.clone()));
        }
        SExp::QuotedString(ql, b'x', b) => {
            return Ok(MatchedNumber::MatchedHex(ql.clone(), b.clone()));
        }
        SExp::Atom(al, b) => {
            // An atom with unprintable characters is rendered as an integer.
            if !printable(b, false) {
                let to_integer = number_from_u8(b);
                return Ok(MatchedNumber::MatchedInt(al.clone(), to_integer));
            }
        }
        SExp::Nil(il) => {
            return Ok(MatchedNumber::MatchedInt(il.clone(), bi_zero()));
        }
        _ => {}
    }

    Err(CompileErr(body.loc(), "Not a number".to_string()))
}

fn numeric_value(body: Rc<SExp>) -> Result<Number, CompileErr> {
    match match_number(body.clone())? {
        MatchedNumber::MatchedInt(_, n) => Ok(n),
        MatchedNumber::MatchedHex(_, h) => Ok(number_from_u8(&h)),
    }
}

fn usize_value(body: Rc<SExp>) -> Result<usize, CompileErr> {
    let n = numeric_value(body.clone())?;
    if let Some(res) = n.to_usize() {
        Ok(res)
    } else {
        Err(CompileErr(body.loc(), "Value out of range".to_string()))
    }
}

/// A container for a function to evaluate in advanced preprocessor macros.
/// We use this trait (which is very similar to the extension trait in Evaluator)
/// as a definite handler for a specific named form, so optional returns aren't
/// needed.  These are held in a collection and looked up.  To be maximally
/// conservative with typing and lifetime, we hold these via Rc<dyn ...>.
pub trait ExtensionFunction {
    fn try_eval(&self, loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr>;
}

struct StringQ;

impl StringQ {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(StringQ)
    }
}

impl ExtensionFunction for StringQ {
    fn try_eval(&self, loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let res = match match_quoted_string(args[0].clone()) {
            Ok(_) => SExp::Integer(loc.clone(), bi_one()),
            _ => SExp::Nil(loc.clone()),
        };

        Ok(Rc::new(res))
    }
}

struct NumberQ;

impl NumberQ {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(NumberQ)
    }
}

impl ExtensionFunction for NumberQ {
    fn try_eval(&self, loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let res = match match_number(args[0].clone()) {
            Ok(_) => SExp::Integer(loc.clone(), bi_one()),
            _ => SExp::Nil(loc.clone()),
        };

        Ok(Rc::new(res))
    }
}

struct SymbolQ;

impl SymbolQ {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(SymbolQ)
    }
}

impl ExtensionFunction for SymbolQ {
    fn try_eval(&self, loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let res = match match_atom(args[0].clone()) {
            Ok(_) => SExp::Integer(loc.clone(), bi_one()),
            _ => SExp::Nil(loc.clone()),
        };

        Ok(Rc::new(res))
    }
}

struct SymbolToString;

impl SymbolToString {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(SymbolToString)
    }
}

impl ExtensionFunction for SymbolToString {
    fn try_eval(&self, _loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let (loc, value) = match_atom(args[0].clone())?;
        Ok(Rc::new(SExp::QuotedString(loc, b'\"', value)))
    }
}

struct StringToSymbol;

impl StringToSymbol {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(StringToSymbol)
    }
}

impl ExtensionFunction for StringToSymbol {
    fn try_eval(&self, _loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let (loc, value) = match_quoted_string(args[0].clone())?;
        Ok(Rc::new(SExp::Atom(loc, value)))
    }
}

struct StringAppend;

impl StringAppend {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(StringAppend)
    }
}

impl ExtensionFunction for StringAppend {
    fn try_eval(&self, loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let mut out_vec = Vec::new();
        let mut out_loc = None;
        for a in args.iter() {
            let (loc, mut value) = match_quoted_string(a.clone())?;
            if out_loc.is_none() {
                out_loc = Some(loc);
            }
            out_vec.append(&mut value);
        }
        Ok(Rc::new(SExp::QuotedString(
            out_loc.unwrap_or_else(|| loc.clone()),
            b'\"',
            out_vec,
        )))
    }
}

struct NumberToString;

impl NumberToString {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(NumberToString)
    }
}

impl ExtensionFunction for NumberToString {
    fn try_eval(&self, _loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let match_res = match_number(args[0].clone())?;
        let (use_loc, int_val) = match &match_res {
            MatchedNumber::MatchedInt(l, i) => (l.clone(), i.clone()),
            MatchedNumber::MatchedHex(l, h) => (l.clone(), number_from_u8(h)),
        };
        Ok(Rc::new(SExp::QuotedString(
            use_loc,
            b'\"',
            int_val.to_string().as_bytes().to_vec(),
        )))
    }
}

struct StringToNumber;

impl StringToNumber {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(StringToNumber)
    }
}

impl ExtensionFunction for StringToNumber {
    fn try_eval(&self, _loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let (loc, value) = match_quoted_string(args[0].clone())?;
        if let Ok(cvt_bi) = decode_string(&value).parse::<Number>() {
            Ok(Rc::new(SExp::Integer(loc, cvt_bi)))
        } else {
            Err(CompileErr(loc, "bad number".to_string()))
        }
    }
}

struct StringLength;

impl StringLength {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(StringLength)
    }
}

impl ExtensionFunction for StringLength {
    fn try_eval(&self, _loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let (loc, value) = match_quoted_string(args[0].clone())?;
        if let Some(len_bi) = value.len().to_bigint() {
            return Ok(Rc::new(SExp::Integer(loc, len_bi)));
        }
        Err(CompileErr(
            args[0].loc(),
            "Error getting string length".to_string(),
        ))
    }
}

struct Substring;

impl Substring {
    fn create() -> Rc<dyn ExtensionFunction> {
        Rc::new(Substring)
    }
}

impl ExtensionFunction for Substring {
    fn try_eval(&self, _loc: &Srcloc, args: &[Rc<SExp>]) -> Result<Rc<SExp>, CompileErr> {
        let start_element = usize_value(args[1].clone())?;
        let end_element = usize_value(args[2].clone())?;

        match args[0].borrow() {
            SExp::QuotedString(l, ch, s) => {
                if start_element > end_element || start_element > s.len() || end_element > s.len() {
                    return Err(CompileErr(
                        l.clone(),
                        "start greater than end in substring".to_string(),
                    ));
                }
                let result_value: Vec<u8> = s
                    .iter()
                    .take(end_element)
                    .skip(start_element)
                    .copied()
                    .collect();
                Ok(Rc::new(SExp::QuotedString(l.clone(), *ch, result_value)))
            }
            _ => Err(CompileErr(args[0].loc(), "Not a string".to_string())),
        }
    }
}

/// An evaluator extension for the preprocessor.
///
/// Implements scheme like conversion functions for handling chialisp programs and
/// bits of them.
///
/// These functions are provided:
///
/// Enhanced versions of builtin macros:
///
/// if   -- first class short circuiting, no round trip to CLVM space
/// list -- simple own implementation
/// c    -- cons preserving exact input values
/// f    -- first and rest exactly preserving part of cons.
/// r    --
///
/// Queries
///
/// string?
/// number?
/// symbol?
///
/// Basic conversion
///
/// string->symbol
/// symbol->string
/// string->number
/// number->string
///
/// Working with values
///
/// string-append s0 s1 ...
/// string-length
/// substring s start end
///
pub struct PreprocessorExtension {
    extfuns: HashMap<Vec<u8>, Rc<dyn ExtensionFunction>>,
}

impl PrimOverride for PreprocessorExtension {
    fn try_handle(
        &self,
        head: Rc<SExp>,
        _context: Rc<SExp>,
        tail: Rc<SExp>,
    ) -> Result<Option<Rc<SExp>>, RunFailure> {
        if let SExp::Atom(hl, head_atom) = head.borrow() {
            let have_args: Vec<Rc<SExp>> = if let Some(args_list) = tail.proper_list() {
                args_list.into_iter().map(Rc::new).collect()
            } else {
                return Ok(None);
            };

            if let Some(extension) = self.extfuns.get(head_atom) {
                let res = extension.try_eval(hl, &have_args)?;
                return Ok(Some(res));
            }
        }

        Ok(None)
    }
}

impl PreprocessorExtension {
    pub fn new() -> Self {
        let extfuns = [
            (b"string?".to_vec(), StringQ::create()),
            (b"number?".to_vec(), NumberQ::create()),
            (b"symbol?".to_vec(), SymbolQ::create()),
            (b"string->symbol".to_vec(), StringToSymbol::create()),
            (b"symbol->string".to_vec(), SymbolToString::create()),
            (b"string->number".to_vec(), StringToNumber::create()),
            (b"number->string".to_vec(), NumberToString::create()),
            (b"string-append".to_vec(), StringAppend::create()),
            (b"string-length".to_vec(), StringLength::create()),
            (b"substring".to_vec(), Substring::create()),
        ];
        PreprocessorExtension {
            extfuns: HashMap::from(extfuns),
        }
    }

    /// Introduce new primitive names for the operators we use to bootstrap macros.
    pub fn enrich_prims(&self, opts: Rc<dyn CompilerOpts>) -> Rc<dyn CompilerOpts> {
        let old_prim_map = opts.prim_map();
        let old_prim_map_borrowed: &HashMap<Vec<u8>, Rc<SExp>> = old_prim_map.borrow();
        let mut new_prim_map_cloned = old_prim_map_borrowed.clone();
        let srcloc = Srcloc::start("*defmac*");

        for (f, _) in self.extfuns.iter() {
            if !new_prim_map_cloned.contains_key(f) {
                new_prim_map_cloned
                    .insert(f.clone(), Rc::new(SExp::Atom(srcloc.clone(), f.clone())));
            }
        }

        opts.set_prim_map(Rc::new(new_prim_map_cloned))
    }
}
