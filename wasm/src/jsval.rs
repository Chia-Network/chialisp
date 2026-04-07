use js_sys::JSON::stringify;
use js_sys::{Array, BigInt, Object, Reflect, Uint8Array};
use wasm_bindgen::JsCast;

use num_bigint::ToBigInt;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;

use chialisp::classic::clvm::__type_compatibility__::{bi_zero, Bytes, BytesFromType};
use chialisp::compiler::sexp::SExp;
use chialisp::compiler::srcloc::{Srcloc, Until};
use chialisp::util::Number;

use wasm_bindgen::prelude::*;

use crate::api::t;
use crate::objects::{find_cached_sexp, js_cache_value_from_js};

pub fn array_to_value(v: Array) -> JsValue {
    let jref: &JsValue = v.as_ref();
    jref.clone()
}

pub fn object_to_value(v: &Object) -> JsValue {
    let jref: &JsValue = v.as_ref();
    jref.clone()
}

pub fn js_pair(a: JsValue, b: JsValue) -> JsValue {
    let pair_array = Array::new();
    pair_array.set(0, a);
    pair_array.set(1, b);
    array_to_value(pair_array)
}

pub fn js_object_from_sexp(v: Rc<SExp>) -> Result<JsValue, JsValue> {
    match v.borrow() {
        SExp::Nil(_) => Ok(JsValue::null()),
        SExp::Integer(_, i) => Ok(JsValue::bigint_from_str(&i.to_string())),
        SExp::QuotedString(_, _, q) => Ok(JsValue::from_str(
            &Bytes::new(Some(BytesFromType::Raw(q.clone()))).decode(),
        )),
        SExp::Atom(_, q) => Ok(JsValue::from_str(
            &Bytes::new(Some(BytesFromType::Raw(q.clone()))).decode(),
        )),
        SExp::Cons(_, a, b) => {
            if let Some(lst) = v.proper_list() {
                let array = Array::new();
                for (i, _) in lst.iter().enumerate() {
                    array.set(
                        i as u32,
                        js_object_from_sexp(Rc::new(lst[i].clone())).unwrap_or_else(|e| e),
                    );
                }
                Ok(array_to_value(array))
            } else {
                t(
                    &js_object_from_sexp(a.clone())?,
                    &js_object_from_sexp(b.clone())?,
                )
            }
        }
    }
}

pub fn get_property(o: &Object, name: &str) -> Option<JsValue> {
    Object::try_from(&Object::get_own_property_descriptor(
        o,
        &JsValue::from_str(name),
    ))
    .and_then(|desc_obj| match Object::try_from(desc_obj) {
        Some(o) => {
            for pj in Object::entries(o).values() {
                let pair = Array::from(&pj.unwrap());
                let propname = pair.get(0).as_string();
                if propname == Some("value".to_string()) {
                    return Some(pair.get(1));
                }
            }
            None
        }
        _ => None,
    })
}

fn location_lc_pair(o: &Object) -> Option<(usize, usize)> {
    get_property(o, "line").and_then(|l| l.as_f64()).map(|l| {
        (
            l as usize,
            get_property(o, "col")
                .and_then(|c| c.as_f64())
                .map(|c| c as usize)
                .unwrap_or_else(|| 0),
        )
    })
}

fn location(o: &Object) -> Option<Srcloc> {
    Object::try_from(o)
        .and_then(|lo| get_property(lo, "file").and_then(|f| f.as_string()))
        .and_then(|f| location_lc_pair(o).map(|(l, c)| (f, l, c)))
        .map(|(f, l, c)| Srcloc {
            file: Rc::new(f),
            line: l,
            col: c,
            until: get_property(o, "until")
                .and_then(|lo| Object::try_from(&lo).cloned())
                .and_then(|lo| location_lc_pair(&lo))
                .map(|(ll, lc)| Until { line: ll, col: lc }),
        })
}

pub fn detect_serializable(loc: &Srcloc, v: &JsValue) -> Option<Rc<SExp>> {
    let serialize_key = JsValue::from_str("serialize");
    js_sys::Reflect::get(v, &serialize_key)
        .ok()
        .and_then(|serialize| {
            Reflect::apply(serialize.unchecked_ref(), v, &js_sys::Array::new())
                .ok()
                .and_then(|array| {
                    let array = Array::from(&array);
                    let mut bytes_array: Vec<u8> = vec![];
                    for item in array.iter() {
                        if let Some(n) = item.as_f64() {
                            if !(0.0..=255.0).contains(&n) {
                                return None;
                            }
                            bytes_array.push(n as u8);
                        } else {
                            return None;
                        }
                    }

                    Some(Rc::new(SExp::QuotedString(loc.clone(), b'x', bytes_array)))
                })
        })
}

pub fn detect_convertible(v: &JsValue) -> Result<Rc<SExp>, JsValue> {
    let convert_key = JsValue::from_str("to_program");
    let to_program = js_sys::Reflect::get(v, &convert_key)?;
    let call_args = js_sys::Array::new();
    call_args.push(v);
    let call_result = Reflect::apply(to_program.unchecked_ref(), v, &call_args)?;
    let cacheval = js_cache_value_from_js(&call_result)?;
    let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
    Ok(cached.modern.clone())
}

pub fn sexp_from_js_object(sstart: Srcloc, v: &JsValue) -> Option<Rc<SExp>> {
    // Already converted value.
    if let Ok(res) = js_cache_value_from_js(v) {
        find_cached_sexp(res.entry, &res.content)
            .map(|result| result.modern.clone())
            .ok()
    } else if v.is_bigint() {
        BigInt::new(v)
            .ok()
            .and_then(|v| v.to_string(10).ok())
            .and_then(|v| v.as_string())
            .and_then(|v| v.parse::<Number>().ok())
            .map(|x| Rc::new(SExp::Integer(sstart.clone(), x)))
    } else if let Some(fval) = v.as_f64() {
        (fval as i64).to_bigint().map(|x| {
            if x == bi_zero() {
                Rc::new(SExp::Nil(sstart.clone()))
            } else {
                Rc::new(SExp::Integer(sstart.clone(), x))
            }
        })
    } else if let Some(g1_bytes) = detect_serializable(&sstart, v) {
        Some(g1_bytes)
    } else if let Ok(converted) = detect_convertible(v) {
        Some(converted)
    } else if Array::is_array(v) {
        let a = Array::from(v);
        let mut result_value = Rc::new(SExp::Nil(Srcloc::start("*js*")));
        for i_rev in 0..a.length() {
            let i = a.length() - i_rev - 1;
            if let Some(nv) = sexp_from_js_object(sstart.clone(), &a.get(i)) {
                result_value = Rc::new(SExp::Cons(nv.loc(), nv, result_value));
            }
        }
        Some(result_value)
    } else {
        Object::try_from(v)
            .map(|o| {
                let loc = get_property(o, "location")
                    .and_then(|o| Object::try_from(&o).cloned())
                    .and_then(|o| location(&o))
                    .unwrap_or_else(|| sstart.clone());
                if Uint8Array::instanceof(v) {
                    // Explicitly handle uint8array conversion.
                    let as_uint8array = Uint8Array::unchecked_from_js(v.clone());
                    return Some(Rc::new(SExp::Atom(loc, as_uint8array.to_vec())));
                }
                get_property(o, "pair")
                    .and_then(|p| {
                        let pa = Array::from(&p);
                        sexp_from_js_object(sstart.clone(), &pa.get(0)).map(|a| (pa, a))
                    })
                    .and_then(|(pa, a)| {
                        sexp_from_js_object(sstart.clone(), &pa.get(1))
                            .map(|b| Rc::new(SExp::Cons(loc, a, b)))
                    })
            })
            .unwrap_or_else(|| {
                v.as_string()
                    .map(|s| Some(Rc::new(SExp::Atom(sstart.clone(), s.as_bytes().to_vec()))))
                    .unwrap_or_else(|| {
                        stringify(v)
                            .ok()
                            .and_then(|v| v.as_string())
                            .and_then(|v| Number::from_str(&v).ok())
                            .map(|n| Rc::new(SExp::Integer(sstart.clone(), n)))
                    })
            })
    }
}

pub fn btreemap_to_object<'a>(iter: impl Iterator<Item = (&'a String, &'a String)>) -> JsValue {
    let entries = Array::new();
    for pv in iter {
        let pair = Array::new();
        pair.set(0, JsValue::from_str(pv.0));
        pair.set(1, JsValue::from_str(pv.1));
        let pair_ref: &JsValue = pair.as_ref();
        entries.set(entries.length(), pair_ref.clone());
    }
    object_to_value(Object::from_entries(&entries).as_ref().unwrap())
}

pub fn read_string_to_string_map(
    symbols: &js_sys::Object,
) -> Result<HashMap<String, String>, String> {
    let mut result = HashMap::new();
    for ent in js_sys::Object::keys(symbols).values() {
        let key = ent.unwrap().as_string().unwrap();
        match get_property(symbols, &key).unwrap().as_string() {
            Some(val) => {
                result.insert(key, val);
            }
            _ => {
                return Err(format!("key {key} wasn't string"));
            }
        }
    }
    Ok(result)
}
