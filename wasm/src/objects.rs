use js_sys;
use js_sys::{Array, JsString, Reflect};
use num_traits::cast::ToPrimitive;
use std::borrow::Borrow;
use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use chialisp::classic::clvm::__type_compatibility__::{
    bi_one, Bytes, Stream, UnvalidatedBytesFromType,
};
use chialisp::classic::clvm::serialize::{
    sexp_from_stream, sexp_to_stream, SimpleCreateCLVMObject,
};
use chialisp::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use chialisp::compiler::clvm::{convert_from_clvm_rs, convert_to_clvm_rs, sha256tree};
use chialisp::compiler::prims::{primapply, primcons, primquote};
use chialisp::compiler::sexp::SExp;
use chialisp::compiler::srcloc::Srcloc;
use clvmr::error::EvalErr;
use clvmr::Allocator;

use crate::api::{create_clvm_runner_err, get_next_id};
use crate::jsval::{js_object_from_sexp, sexp_from_js_object};

const DEFAULT_CACHE_ENTRIES: usize = 1024;

struct FunctionWrapperDesc {
    export_name: &'static str,
    member_name: &'static str,
    varargs: bool,
}

#[derive(Clone)]
pub struct ObjectCacheMember {
    pub modern: Rc<SExp>,
}

struct ObjectCache {
    cache_data: HashMap<i32, ObjectCacheMember>,
    cache_order: Vec<i32>,
    cache_length: usize,
}

pub struct JsCacheValue {
    pub entry: i32,
    pub content: String,
}

pub fn js_cache_value_from_js(jsval: &JsValue) -> Result<JsCacheValue, JsValue> {
    let entry = js_sys::Reflect::get(jsval, &JsString::from("id"))?
        .as_f64()
        .ok_or(JsString::from("id was not a number"))?;
    let content = js_sys::Reflect::get(jsval, &JsString::from("content"))?
        .as_string()
        .ok_or(JsString::from("content was not a string"))?;

    Ok(JsCacheValue {
        entry: entry as i32,
        content,
    })
}

impl Default for ObjectCache {
    fn default() -> Self {
        ObjectCache {
            cache_data: HashMap::default(),
            cache_order: Vec::default(),
            cache_length: DEFAULT_CACHE_ENTRIES,
        }
    }
}

impl ObjectCache {
    fn create_or_update_cache_entry(&mut self, id: i32, cache_member: ObjectCacheMember) {
        if id < 0 {
            // Special nil handling.
            return;
        }

        if self.cache_order.len() > self.cache_length {
            self.cache_data.remove(&self.cache_order[0]);
            self.cache_order.remove(0);
        }

        self.cache_data.insert(id, cache_member);
        self.cache_order.push(id);
    }

    fn create_entry_from_sexp(&mut self, id: i32, sexp: Rc<SExp>) -> Result<String, JsValue> {
        let mut allocator = Allocator::new();
        let node = convert_to_clvm_rs(&mut allocator, sexp.clone())
            .map_err(|_| js_sys::JsString::from("could not convert to clvm"))?;
        let mut stream = Stream::new(None);
        sexp_to_stream(&mut allocator, node, &mut stream);

        self.create_or_update_cache_entry(
            id,
            ObjectCacheMember {
                modern: sexp.clone(),
            },
        );

        Ok(stream.get_value().hex())
    }

    fn find_or_create_entry_from_hex(
        &mut self,
        entry: i32,
        content: &str,
    ) -> Result<ObjectCacheMember, JsValue> {
        if let Some(res) = self.cache_data.get(&entry) {
            return Ok(res.clone());
        }

        let mut allocator = Allocator::new();
        let bytes_from_hex =
            Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(content.to_string())))
                .map_err(|_| JsString::from("could not parse hex"))?;
        let mut stream = Stream::new(Some(bytes_from_hex));
        let parsed = sexp_from_stream(
            &mut allocator,
            &mut stream,
            Box::new(SimpleCreateCLVMObject {}),
        )
        .map(|x| x.1)
        .map_err(|_| JsString::from("could not parse sexp from hex"))?;
        let srcloc = Srcloc::start("*var*");
        let modern = convert_from_clvm_rs(&mut allocator, srcloc, parsed)
            .map_err(|_| JsString::from("could not realize parsed sexp"))?;

        let cache_entry = ObjectCacheMember { modern };
        self.create_or_update_cache_entry(entry, cache_entry.clone());

        Ok(cache_entry)
    }
}

static PROGRAM_FUNCTIONS: &[FunctionWrapperDesc] = &[
    FunctionWrapperDesc {
        export_name: "toString",
        member_name: "to_string_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "as_pair",
        member_name: "as_pair_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "listp",
        member_name: "listp_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "nullp",
        member_name: "nullp_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "as_int",
        member_name: "as_int_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "as_bigint",
        member_name: "as_bigint_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "as_bin",
        member_name: "as_bin_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "first",
        member_name: "first_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "rest",
        member_name: "rest_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "cons",
        member_name: "cons_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "run",
        member_name: "run_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "list_len",
        member_name: "list_len_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "equal_to",
        member_name: "equal_to_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "as_javascript",
        member_name: "as_javascript_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "curry",
        member_name: "curry_internal",
        varargs: true,
    },
    FunctionWrapperDesc {
        export_name: "sha256tree",
        member_name: "sha256tree_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "uncurry_error",
        member_name: "uncurry_error_internal",
        varargs: false,
    },
    FunctionWrapperDesc {
        export_name: "uncurry",
        member_name: "uncurry_internal",
        varargs: false,
    },
];

static TUPLE_FUNCTIONS: &[FunctionWrapperDesc] = &[FunctionWrapperDesc {
    export_name: "to_program",
    member_name: "tuple_to_program_internal",
    varargs: false,
}];

thread_local! {
    static OBJECT_CACHE: RefCell<ObjectCache> = {
        return RefCell::new(ObjectCache::default());
    };
    static PROGRAM_PROTOTYPE: RefCell<Option<JsValue>> = const { RefCell::new(None) };
    static TUPLE_PROTOTYPE: RefCell<Option<JsValue>> = const { RefCell::new(None) };
    static SRCLOC: Srcloc = Srcloc::start("*var*");
}

fn create_cached_sexp(id: i32, sexp: Rc<SExp>) -> Result<String, JsValue> {
    OBJECT_CACHE.with(|ocache| {
        let mut mut_object_cache_ref: RefMut<ObjectCache> = ocache.borrow_mut();
        mut_object_cache_ref.create_entry_from_sexp(id, sexp)
    })
}

fn get_srcloc() -> Srcloc {
    SRCLOC.with(|s| s.clone())
}

pub fn find_cached_sexp(entry: i32, content: &str) -> Result<ObjectCacheMember, JsValue> {
    if content == "80" {
        return Ok(ObjectCacheMember {
            modern: Rc::new(SExp::Nil(get_srcloc())),
        });
    }

    OBJECT_CACHE.with(|ocache| {
        let mut mut_object_cache_ref: RefMut<ObjectCache> = ocache.borrow_mut();
        mut_object_cache_ref.find_or_create_entry_from_hex(entry, content)
    })
}

#[wasm_bindgen(typescript_custom_section)]
const IProgram: &'static str = r#"
interface ITuple {
    to_program(): IProgram;
}

interface IProgram {
    toString(): string;
    as_pair(): ITuple;
    listp(): boolean;
    nullp(): boolean;
    as_int(): number;
    as_bigint(): bigint;
    as_bin(): Uint8Array;
    first(): IProgram;
    rest(): IProgram;
    cons(p: IProgram): IProgram;
    run(env: IProgram): [number, IProgram];
    list_len(): number;
    equal_to(other: IProgram): boolean;
    as_javascript(): any;
    curry(args: [IProgram]): IProgram;
    sha256tree(): Uint8Array;
    uncurry_error(): [IProgram, Array<IProgram>];
    uncurry(): [IProgram, Array<IProgram>|null];
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "IProgram")]
    pub type IProgram;
}

// Strategy for Program objects.
// We'll provide a Program object that allows users to have something that
// acts js-y but conserves compute time when possible.
//
// The result object we'll give back contain a string representation of the
// program they're associated with but also a cache hint.  If the cache object
// doesn't exist we'll re-parse the hex and re-cache it.
//
// The object is structured like this:
//
// {"cache-hint":33, "hex":"ff..."}
//
// Cache hints are never reused, and cleared when the counter transitions from
// low half to high half or vice versa.
//
// The object's prototype will include methods for Program, such as cons and call
// the real static methods of Program.
#[wasm_bindgen(inspectable)]
pub struct Program {}

// Build prototype
fn get_program_prototype() -> Result<JsValue, JsValue> {
    // We've already built the program prototype object.
    if let Some(pp) = PROGRAM_PROTOTYPE.with(|pp| pp.borrow().clone()) {
        return Ok(pp);
    }

    let prototype = js_sys::Object::new();

    let program_self = js_sys::eval("Program")?;
    for func_wrapper_desc in PROGRAM_FUNCTIONS.iter() {
        let pass_on_args = if func_wrapper_desc.varargs {
            "[args]"
        } else {
            "args"
        };
        let to_string_fun = js_sys::Function::new_with_args(
            "",
            &format!("const t = this; return function() {{ let args = Array.prototype.slice.call(arguments); let apply_args = {pass_on_args}; apply_args.unshift(this); return t.{}.apply(null, apply_args); }}", func_wrapper_desc.member_name)
        );

        let to_string_final = to_string_fun.call0(&program_self)?;
        js_sys::Reflect::set(
            &prototype,
            &js_sys::JsString::from(func_wrapper_desc.export_name),
            &to_string_final,
        )?;
    }

    PROGRAM_PROTOTYPE.with(|pp| {
        pp.replace(Some(prototype.clone().into()));
    });

    Ok(prototype.into())
}

fn get_tuple_prototype() -> Result<JsValue, JsValue> {
    if let Some(pp) = TUPLE_PROTOTYPE.with(|pp| pp.borrow().clone()) {
        return Ok(pp);
    }

    let prototype = js_sys::Object::new();

    let program_self = js_sys::eval("Program")?;
    for func_wrapper_desc in TUPLE_FUNCTIONS.iter() {
        let to_string_fun = js_sys::Function::new_with_args(
            "",
            &format!("const t = this; return function() {{ let args = Array.prototype.slice.call(arguments); args.unshift(this); return t.{}.apply(null, args); }}", func_wrapper_desc.member_name)
        );

        let to_string_final = to_string_fun.call0(&program_self)?;
        js_sys::Reflect::set(
            &prototype,
            &js_sys::JsString::from(func_wrapper_desc.export_name),
            &to_string_final,
        )?;
    }

    TUPLE_PROTOTYPE.with(|pp| {
        pp.replace(Some(prototype.clone().into()));
    });

    Ok(prototype.into())
}

pub fn finish_new_object(id: i32, encoded_hex: &str) -> Result<JsValue, JsValue> {
    let prototype = get_program_prototype()?;

    let new_object = js_sys::Object::new();
    js_sys::Reflect::set_prototype_of(&new_object, &prototype)?;

    js_sys::Reflect::set(
        &new_object,
        &js_sys::JsString::from("content"),
        &js_sys::JsString::from(encoded_hex),
    )?;
    js_sys::Reflect::set(
        &new_object,
        &js_sys::JsString::from("id"),
        &js_sys::Number::from(id),
    )?;

    Ok(new_object.into())
}

// Return a vector of arguments if the given SExp is the expected operator
// and has the required number of arguments.
fn match_op(
    opcode: u8,
    mut expected_args: usize,
    opname: &str,
    program: Rc<SExp>,
) -> Result<Vec<Rc<SExp>>, JsValue> {
    let plist = if expected_args == 0 {
        if let SExp::Cons(_, a, b) = program.borrow() {
            let a_borrowed: &SExp = a.borrow();
            let b_borrowed: &SExp = b.borrow();
            expected_args = 1;
            vec![a_borrowed.clone(), b_borrowed.clone()]
        } else {
            return Err(JsValue::from_str(&format!(
                "program was expected to be a cons, but wasn't: {program}"
            )));
        }
    } else if let Some(plist) = program.proper_list() {
        plist
    } else {
        // Not a list so can't be an apply.
        return Err(JsValue::from_str(&format!(
            "program wasn't a list representing an {opname} op: {program}"
        )));
    };

    // Not the right length
    if plist.len() != expected_args + 1 {
        return Err(JsValue::from_str(&format!(
            "program list wasn't a list of {} representing an {opname} op: {program}",
            expected_args + 1
        )));
    }

    // Not an apply
    if plist[0] != SExp::Atom(plist[0].loc(), vec![opcode]) {
        return Err(JsValue::from_str("program isn't an {opname} op: {program}"));
    }

    Ok(plist.into_iter().skip(1).map(Rc::new).collect())
}

fn cache_and_accumulate_arg(array: &Array, prog: Rc<SExp>) -> Result<(), JsValue> {
    let arg_id = get_next_id();
    let new_cached_arg = create_cached_sexp(arg_id, prog)?;
    let arg_js = finish_new_object(arg_id, &new_cached_arg)?;

    array.push(&arg_js);

    Ok(())
}

fn to_iprogram(v: JsValue) -> IProgram {
    v.unchecked_into::<IProgram>()
}

#[wasm_bindgen]
impl Program {
    pub fn to_internal(input: &JsValue) -> Result<JsValue, JsValue> {
        let loc = get_srcloc();
        let sexp = sexp_from_js_object(loc, input).map(Ok).unwrap_or_else(|| {
            Err(create_clvm_runner_err(
                "unable to convert to value".to_string(),
            ))
        })?;

        let new_id = get_next_id();

        let encoded = create_cached_sexp(new_id, sexp)?;

        // Build the object
        finish_new_object(new_id, &encoded)
    }

    #[wasm_bindgen]
    pub fn to(input: &JsValue) -> Result<IProgram, JsValue> {
        Program::to_internal(input).map(to_iprogram)
    }

    #[wasm_bindgen]
    pub fn from_hex(input: &str) -> Result<IProgram, JsValue> {
        let new_id = get_next_id();
        let _ = find_cached_sexp(new_id, input)?;
        finish_new_object(new_id, input).map(to_iprogram)
    }

    #[wasm_bindgen]
    pub fn null() -> Result<IProgram, JsValue> {
        let new_id = get_next_id();
        let encoded = create_cached_sexp(new_id, Rc::new(SExp::Nil(get_srcloc())))?;

        finish_new_object(new_id, &encoded).map(to_iprogram)
    }

    #[wasm_bindgen]
    pub fn sha256tree_internal(obj: &JsValue) -> Result<Vec<u8>, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        Ok(sha256tree(cached.modern.clone()))
    }

    #[wasm_bindgen]
    pub fn to_string_internal(obj: &JsValue) -> Result<JsValue, JsValue> {
        js_sys::Reflect::get(obj, &js_sys::JsString::from("content"))
    }

    #[wasm_bindgen]
    pub fn as_pair_internal(obj: &JsValue) -> Result<JsValue, JsValue> {
        let prototype = get_tuple_prototype()?;
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;

        if let SExp::Cons(_, a, b) = cached.modern.borrow() {
            let id_a = get_next_id();
            let new_cached_a = create_cached_sexp(id_a, a.clone())?;
            let object_a = finish_new_object(id_a, &new_cached_a)?;
            let id_b = get_next_id();
            let new_cached_b = create_cached_sexp(id_b, b.clone())?;
            let object_b = finish_new_object(id_b, &new_cached_b)?;

            let result_value = Array::new();
            result_value.set(0, object_a);
            result_value.set(1, object_b);
            // Support reading as a classic clvm input.
            Reflect::set(&result_value, &JsString::from("pair"), &result_value)?;
            Reflect::set_prototype_of(&result_value, &prototype)?;
            return Ok(result_value.into());
        }

        Ok(JsValue::null())
    }

    #[wasm_bindgen]
    pub fn listp_internal(obj: &JsValue) -> Result<bool, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        Ok(cacheval.content.starts_with("ff"))
    }

    #[wasm_bindgen]
    pub fn nullp_internal(obj: &JsValue) -> Result<bool, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        Ok(cacheval.content == "80")
    }

    #[wasm_bindgen]
    pub fn as_int_internal(obj: &JsValue) -> Result<i32, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        let number = cached
            .modern
            .get_number()
            .map_err(|_| JsString::from("not a number"))?;
        (number.to_i32()).ok_or(JsString::from("number out of range").into())
    }

    #[wasm_bindgen]
    pub fn as_bigint_internal(obj: &JsValue) -> Result<js_sys::BigInt, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        let number = cached
            .modern
            .get_number()
            .map_err(|_| JsString::from("not a number"))?;
        let num_string = number.to_string();
        let num_str: &str = &num_string;
        js_sys::BigInt::new(&JsString::from(num_str))
            .map_err(|_| JsString::from("couldn't construct bigint").into())
    }

    #[wasm_bindgen]
    pub fn first_internal(obj: &JsValue) -> Result<IProgram, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        if let SExp::Cons(_, a, _) = cached.modern.borrow() {
            let id_a = get_next_id();
            let new_cached_a = create_cached_sexp(id_a, a.clone())?;
            return finish_new_object(id_a, &new_cached_a).map(to_iprogram);
        }

        Err(JsString::from("not a cons").into())
    }

    #[wasm_bindgen]
    pub fn rest_internal(obj: &JsValue) -> Result<IProgram, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        if let SExp::Cons(_, _, a) = cached.modern.borrow() {
            let id_a = get_next_id();
            let new_cached_a = create_cached_sexp(id_a, a.clone())?;
            return finish_new_object(id_a, &new_cached_a).map(to_iprogram);
        }

        Err(JsString::from("not a cons").into())
    }

    #[wasm_bindgen]
    pub fn cons_internal(obj: &JsValue, other: &JsValue) -> Result<IProgram, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;

        let other_val = js_cache_value_from_js(other)?;
        let other_cache = find_cached_sexp(other_val.entry, &other_val.content)?;

        let new_id = get_next_id();
        let new_sexp = Rc::new(SExp::Cons(
            get_srcloc(),
            cached.modern.clone(),
            other_cache.modern.clone(),
        ));
        let new_cached = create_cached_sexp(new_id, new_sexp)?;
        finish_new_object(new_id, &new_cached).map(to_iprogram)
    }

    #[wasm_bindgen]
    pub fn run_internal(obj: &JsValue, args: &JsValue) -> Result<JsValue, JsValue> {
        let progval = js_cache_value_from_js(obj)?;
        let prog_cache = find_cached_sexp(progval.entry, &progval.content)?;

        let argval = js_cache_value_from_js(args)?;
        let arg_cache = find_cached_sexp(argval.entry, &argval.content)?;

        let mut allocator = Allocator::new();
        let prog_classic =
            convert_to_clvm_rs(&mut allocator, prog_cache.modern.clone()).map_err(|_| {
                let err: JsValue = JsString::from("error converting program").into();
                err
            })?;
        let arg_classic =
            convert_to_clvm_rs(&mut allocator, arg_cache.modern.clone()).map_err(|_| {
                let err: JsValue = JsString::from("error converting args").into();
                err
            })?;

        let runner = DefaultProgramRunner::default();
        let run_result = runner
            .run_program(&mut allocator, prog_classic, arg_classic, None)
            .map_err(|e| {
                let err_str = match e {
                    EvalErr::InternalError(_, e) => e.to_string(),
                    _ => e.to_string(),
                };
                let err: JsValue = JsString::from(err_str.as_str()).into();
                err
            })?;
        let modern_result = convert_from_clvm_rs(&mut allocator, get_srcloc(), run_result.1)
            .map_err(|_| {
                let err: JsValue = JsString::from("error converting result").into();
                err
            })?;
        let result_id = get_next_id();
        let new_cached_result = create_cached_sexp(result_id, modern_result)?;
        let result_object = finish_new_object(result_id, &new_cached_result)?;
        let cost_and_result_array = Array::new();
        cost_and_result_array.push(&JsValue::from_f64(run_result.0 as f64));
        cost_and_result_array.push(&result_object);
        Ok(cost_and_result_array.into())
    }

    #[wasm_bindgen]
    pub fn tuple_to_program_internal(obj: &JsValue) -> Result<IProgram, JsValue> {
        let a = js_sys::Reflect::get(obj, &JsString::from("0"))?;
        let b = js_sys::Reflect::get(obj, &JsString::from("1"))?;
        Program::cons_internal(&a, &b)
    }

    #[wasm_bindgen]
    pub fn as_bin_internal(obj: &JsValue) -> Result<Vec<u8>, JsValue> {
        let convert = Reflect::get(obj, &JsString::from("content"))?
            .as_string()
            .ok_or(JsString::from("content wasn't a hex string"))?;
        let bytes = Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(convert)))
            .map_err(|_| JsString::from("could not convert to binary data"))?;
        Ok(bytes.data().clone())
    }

    #[wasm_bindgen]
    pub fn list_len_internal(obj: &JsValue) -> Result<i32, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        let mut val_ref = cached.modern.clone();
        let mut count: i32 = 0;
        while let SExp::Cons(_, _, b) = val_ref.borrow() {
            val_ref = b.clone();
            count += 1;
        }
        Ok(count)
    }

    #[wasm_bindgen]
    pub fn equal_to_internal(a: &JsValue, b: &JsValue) -> Result<bool, JsValue> {
        let a_cacheval = js_cache_value_from_js(a)?;
        let a_cached = find_cached_sexp(a_cacheval.entry, &a_cacheval.content)?;
        let b_cacheval = js_cache_value_from_js(b)?;
        let b_cached = find_cached_sexp(b_cacheval.entry, &b_cacheval.content)?;
        // Short circuit address equality.
        if Rc::as_ptr(&a_cached.modern) == Rc::as_ptr(&b_cached.modern) {
            return Ok(true);
        }
        Ok(a_cached.modern == b_cached.modern)
    }

    #[wasm_bindgen]
    pub fn as_javascript_internal(obj: &JsValue) -> Result<JsValue, JsValue> {
        let cacheval = js_cache_value_from_js(obj)?;
        let cached = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        js_object_from_sexp(cached.modern.clone())
    }

    // Ported from chia.types.blockchain_format.program in chia-blockchain.
    //
    // original comment:
    //
    // Replicates the curry function from clvm_tools, taking advantage of *args
    // being a list.  We iterate through args in reverse building the code to
    // create a clvm list.
    //
    // Given arguments to a function addressable by the '1' reference in clvm
    //
    // fixed_args = 1
    //
    // Each arg is prepended as fixed_args = (c (q . arg) fixed_args)
    //
    // The resulting argument list is interpreted with apply (2)
    //
    // (2 (1 . self) rest)
    //
    // Resulting in a function which places its own arguments after those
    // curried in in the form of a proper list.
    #[wasm_bindgen]
    pub fn curry_internal(obj: &JsValue, args: Vec<JsValue>) -> Result<IProgram, JsValue> {
        let program_val = Program::to(obj)?;
        let cacheval = js_cache_value_from_js(&program_val)?;
        let program = find_cached_sexp(cacheval.entry, &cacheval.content)?;
        let mut fixed_args = Rc::new(SExp::Integer(get_srcloc(), bi_one()));

        for a in args.iter().rev() {
            let argval = Program::to(a)?;
            let a_cacheval = js_cache_value_from_js(&argval)?;
            let a_cached = find_cached_sexp(a_cacheval.entry, &a_cacheval.content)?;
            fixed_args = Rc::new(primcons(
                get_srcloc(),
                Rc::new(primquote(get_srcloc(), a_cached.modern.clone())),
                fixed_args,
            ));
        }

        let result = Rc::new(primapply(
            get_srcloc(),
            Rc::new(primquote(get_srcloc(), program.modern.clone())),
            fixed_args,
        ));

        let new_id = get_next_id();
        let new_cached = create_cached_sexp(new_id, result)?;
        finish_new_object(new_id, &new_cached).map(to_iprogram)
    }

    #[wasm_bindgen]
    pub fn uncurry_error_internal(obj: &JsValue) -> Result<Vec<IProgram>, JsValue> {
        let program_val = Program::to(obj)?;
        let cacheval = js_cache_value_from_js(&program_val)?;
        let program = find_cached_sexp(cacheval.entry, &cacheval.content)?;

        let apply_args = match_op(2, 2, "apply", program.modern.clone())?;

        // Not used in code, but detects a quoted program.
        let quoted_prog = match_op(1, 0, "quote", apply_args[0].clone())?;

        let retrieved_args = Array::new();
        let mut cons_expr = match_op(4, 2, "cons", apply_args[1].clone())?;

        let decons_and_unquote = |expr: Rc<SExp>| {
            if let Ok(unquoted_curry_argument) = match_op(1, 0, "quote", expr.clone()) {
                unquoted_curry_argument[0].clone()
            } else {
                expr
            }
        };

        cache_and_accumulate_arg(&retrieved_args, decons_and_unquote(cons_expr[0].clone()))?;
        let mut next_cons = cons_expr[1].clone();
        while matches!(next_cons.borrow(), SExp::Cons(_, _, _)) {
            cons_expr = match_op(4, 2, "cons", next_cons)?;

            // Convert to the external js form and insert into cache so we can
            // forego conversion if still cached.
            cache_and_accumulate_arg(&retrieved_args, decons_and_unquote(cons_expr[0].clone()))?;

            // Move on to the tail that's being built.
            next_cons = cons_expr[1].clone();
        }

        // Verify that we're at a 1 env ref.
        let borrowed_next: &SExp = next_cons.borrow();
        if borrowed_next != &SExp::Atom(next_cons.loc(), vec![1]) {
            return Err(JsValue::from_str("curry didn't end with 1 env ref"));
        }

        // Make a cache slot for the program.
        let mod_id = get_next_id();
        let new_cached_mod = create_cached_sexp(mod_id, quoted_prog[0].clone())?;
        let mod_js = finish_new_object(mod_id, &new_cached_mod)?;

        Ok(vec![
            mod_js.into(),
            retrieved_args.unchecked_into::<IProgram>(),
        ])
    }

    #[wasm_bindgen]
    pub fn uncurry_internal(obj: &JsValue) -> Result<Vec<IProgram>, JsValue> {
        if let Ok(res) = Program::uncurry_error_internal(obj) {
            Ok(res)
        } else {
            Ok(vec![
                obj.clone().unchecked_into::<IProgram>(),
                JsValue::null().unchecked_into::<IProgram>(),
            ])
        }
    }
}
