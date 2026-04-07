use std::collections::HashMap;
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::serialize::sexp_to_stream;
use crate::classic::clvm::sexp::{enlist, proper_list, rest, First, SelectNode, ThisNode};

use crate::classic::clvm_tools::binutils::disassemble;
use crate::classic::clvm_tools::sha256tree::sha256tree;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::comptypes::{CompileErr, CompilerOpts};
use crate::compiler::frontend::frontend;
use crate::compiler::sexp::parse_sexp;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::usecheck::check_parameters_used_compileform;

// export const PRELUDE = `<html>
// <head>
//   <link rel="stylesheet"
//       href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
//       integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T"
//       crossorigin="anonymous">
//   <script
//       src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
//       integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo"
//       crossorigin="anonymous"></script>
//   <script
//       src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
//       integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1"
//       crossorigin="anonymous"></script>
//   <script
//       src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"
//       integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM"
//       crossorigin="anonymous"></script>
// </head>
// <body>
// <div class="container">
// `;

// export const TRAILER = "</div></body></html>";

// export function dump_sexp(s: SExp, disassemble_f: typeof disassemble = disassemble){
//   return `<span id="${s.__repr__()}">${disassemble_f(s)}</span>`;
// }

/// Contains additional info beside the compiled form for chialisp functions/
/// These can be passed on and used by debuggers and such.
#[derive(Clone)]
pub struct FunctionExtraInfo {
    /// The form of the original arguments from the source code.
    pub args: NodePtr,
    /// Whether this function requires the constants and functions of the program
    /// as an additional hidden parameter.
    pub has_constants_tree: bool,
}

// // The function below is broken as of 2021/06/22.
// /*
// export function dump_invocation(
//   form: SExp,
//   rewrit_form: SExp,
//   env: SExp[],
//   result: SExp,
//   disassemble_f: typeof disassemble = disassemble,
// ){
//   print(`<hr><div class="invocation" id="${form}">`);
//   print(`<span class="form"><a name="id_${form}">${dump_sexp(form, disassemble)}</a></span>`);
//   print("<ul>")
//   if (form != rewrit_form){
//     print(
//       `<li>Rewritten as:<span class="form">`,
//       `<a name="id_${rewrit_form}">${dump_sexp(rewrit_form, disassemble)}</a></span></li>`,
//     );
//   }
//   env.forEach((e, i) => {
//     print(`<li>x${i}: <a href="#id_${e}">${dump_sexp(e, disassemble_f)}</a>`);
//   });
//   print("</ul>");
//   print(`<span class="form">${dump_sexp(result, disassemble_f)}</span>`);
//   if(form.listp() && form.list_len() > 1){
//     print(`<ul>`);
//     // @todo Implement here if original python code is fixed.
//   }
//   print(`</ul>`)
//   print("</div>")
// }
// export function trace_to_html(){
//     // @todo Implement here if original python code is fixed.
// }
// */
pub fn build_symbol_dump(
    allocator: &mut Allocator,
    constants_lookup: &HashMap<Vec<u8>, NodePtr>,
    extra_function_data: &HashMap<Vec<u8>, FunctionExtraInfo>,
    run_program: Rc<dyn TRunProgram>,
    extra_info: bool,
) -> Result<NodePtr, EvalErr> {
    let mut map_result: Vec<NodePtr> = Vec::new();

    for (k, v) in constants_lookup.iter() {
        let run_result = run_program.run_program(allocator, *v, NodePtr::NIL, None)?;

        let sha256 = sha256tree(allocator, run_result.1).hex();
        let sha_atom = allocator.new_atom(sha256.as_bytes())?;
        let name_atom = allocator.new_atom(&k.clone())?;

        map_result.push(allocator.new_pair(sha_atom, name_atom)?);

        if !extra_info {
            continue;
        }

        if let Some(extra) = extra_function_data.get(k) {
            let mut args_atom = Vec::new();
            let mut left_env_atom = Vec::new();
            args_atom.append(&mut sha256.as_bytes().to_vec());
            args_atom.append(&mut "_arguments".as_bytes().to_vec());

            left_env_atom.append(&mut sha256.as_bytes().to_vec());
            left_env_atom.append(&mut "_left_env".as_bytes().to_vec());

            let args_name_atom = allocator.new_atom(&args_atom)?;
            let left_env_name_atom = allocator.new_atom(&left_env_atom)?;

            let serialized_args = disassemble(allocator, extra.args, Some(0));
            let serialized_args_atom = allocator.new_atom(serialized_args.as_bytes())?;

            let left_env_value = allocator.new_atom(&[extra.has_constants_tree as u8])?;

            map_result.push(allocator.new_pair(args_name_atom, serialized_args_atom)?);
            map_result.push(allocator.new_pair(left_env_name_atom, left_env_value)?);
        }
    }

    enlist(allocator, &map_result)
}

fn text_trace(
    allocator: &mut Allocator,
    output: &mut Stream,
    disassemble_f: &dyn Fn(&mut Allocator, NodePtr) -> String,
    form: NodePtr,
    symbol: Option<String>,
    env_: NodePtr,
    result: &str,
) {
    let symbol_val;
    let mut env = env_;
    match symbol {
        Some(sym) => {
            env = rest(allocator, env).unwrap_or(NodePtr::NIL);
            let symbol_atom = allocator.new_atom(sym.as_bytes()).unwrap();
            let symbol_list = allocator.new_pair(symbol_atom, env).unwrap();
            symbol_val = disassemble_f(allocator, symbol_list);
        }
        _ => {
            symbol_val = format!(
                "{} [{}]",
                disassemble_f(allocator, form),
                disassemble_f(allocator, env)
            );
        }
    }

    output.write_str(&format!("{symbol_val} => {result}\n\n"));
}

fn table_trace(
    allocator: &mut Allocator,
    stdout: &mut Stream,
    disassemble_f: &dyn Fn(&mut Allocator, NodePtr) -> String,
    form: NodePtr,
    _symbol: Option<String>,
    env: NodePtr,
    result: &str,
) {
    let (sexp, args) = match allocator.sexp(form) {
        SExp::Pair(sexp, args) => (sexp, args),
        SExp::Atom => (form, NodePtr::NIL),
    };

    stdout.write_str(&format!("exp: {}\n", disassemble_f(allocator, sexp)));
    stdout.write_str(&format!("arg: {}\n", disassemble_f(allocator, args)));
    stdout.write_str(&format!("env: {}\n", disassemble_f(allocator, env)));
    stdout.write_str(&format!("val: {result}\n"));
    let mut sexp_stream = Stream::new(None);
    sexp_to_stream(allocator, sexp, &mut sexp_stream);
    let mut args_stream = Stream::new(None);
    sexp_to_stream(allocator, args, &mut args_stream);
    let mut benv_stream = Stream::new(None);
    sexp_to_stream(allocator, env, &mut benv_stream);
    stdout.write_str(&format!("bexp: {}\n", sexp_stream.get_value().pybytes()));
    stdout.write_str(&format!("barg: {}\n", args_stream.get_value().pybytes()));
    stdout.write_str(&format!("benv: {}\n", benv_stream.get_value().pybytes()));
    stdout.write_str("--\n");
}

type DisplayTraceFun = dyn Fn(
    &mut Allocator,
    &mut Stream,
    &dyn Fn(&mut Allocator, NodePtr) -> String,
    NodePtr,
    Option<String>,
    NodePtr,
    &str,
);

fn display_trace(
    allocator: &mut Allocator,
    stdout: &mut Stream,
    only_exn: bool,
    trace: &[NodePtr],
    disassemble_f: &dyn Fn(&mut Allocator, NodePtr) -> String,
    symbol_table: Option<HashMap<String, String>>,
    display_fun: &DisplayTraceFun,
) {
    for item in trace {
        let item_vec = proper_list(allocator, *item, true).unwrap();
        let form = item_vec[0];
        let env = item_vec[1];
        let not_exn = item_vec.len() > 2;
        let rv = if not_exn {
            disassemble_f(allocator, item_vec[2])
        } else {
            "(didn't finish)".to_string()
        };

        let h = sha256tree(allocator, form).hex();
        let symbol = symbol_table
            .as_ref()
            .and_then(|st| st.get(&h).map(|x| x.to_string()));

        let display = !only_exn || !not_exn;
        if display {
            display_fun(allocator, stdout, disassemble_f, form, symbol, env, &rv);
        }
    }
}

pub fn trace_to_text(
    allocator: &mut Allocator,
    stdout: &mut Stream,
    only_exn: bool,
    trace: &[NodePtr],
    symbol_table: Option<HashMap<String, String>>,
    disassemble_f: &dyn Fn(&mut Allocator, NodePtr) -> String,
) {
    display_trace(
        allocator,
        stdout,
        only_exn,
        trace,
        disassemble_f,
        symbol_table,
        &text_trace,
    );
}

pub fn trace_to_table(
    allocator: &mut Allocator,
    stdout: &mut Stream,
    only_exn: bool,
    trace: &[NodePtr],
    symbol_table: Option<HashMap<String, String>>,
    disassemble_f: &dyn Fn(&mut Allocator, NodePtr) -> String,
) {
    display_trace(
        allocator,
        stdout,
        only_exn,
        trace,
        disassemble_f,
        symbol_table,
        &table_trace,
    );
}

pub fn trace_pre_eval(
    allocator: &mut Allocator,
    append_log: &dyn Fn(&mut Allocator, NodePtr),
    symbol_table: Option<HashMap<String, String>>,
    sexp: NodePtr,
    args: NodePtr,
) -> Result<Option<NodePtr>, EvalErr> {
    let h = sha256tree(allocator, sexp);
    let recognized = symbol_table
        .as_ref()
        .and_then(|symbol_table| symbol_table.get(&h.hex()).map(|x| x.to_string()));

    if recognized.is_none() && symbol_table.is_some() {
        Ok(None)
    } else {
        m! {
            log_entry <- enlist(allocator, &[sexp, args]);
            let _ = append_log(allocator, log_entry);
            Ok(Some(log_entry))
        }
    }
}

pub fn check_unused(
    opts: Rc<dyn CompilerOpts>,
    input_program: &str,
) -> Result<(bool, String), CompileErr> {
    let mut output: Stream = Stream::new(None);
    let pre_forms = parse_sexp(Srcloc::start(&opts.filename()), input_program.bytes())?;
    let g = frontend(opts.clone(), &pre_forms)?;
    let unused = check_parameters_used_compileform(opts, Rc::new(g))?;

    if !unused.is_empty() {
        output.write_str("unused arguments detected at the mod level (lower case arguments are considered uncurried by convention)\n");
        for s in unused.iter() {
            output.write_str(&format!(
                " - {}\n",
                Bytes::new(Some(BytesFromType::Raw(s.clone()))).decode()
            ));
        }
        Ok((false, output.get_value().decode()))
    } else {
        Ok((true, output.get_value().decode()))
    }
}

pub fn program_hash_from_program_env_cons(
    allocator: &mut Allocator,
    prog_pair: NodePtr,
) -> Result<Bytes, EvalErr> {
    let First::Here(program) = First::Here(ThisNode::Here).select_nodes(allocator, prog_pair)?;
    Ok(sha256tree(allocator, program))
}

pub fn start_log_after(
    allocator: &mut Allocator,
    maybe_program_hash: Option<Bytes>,
    log: Vec<NodePtr>,
) -> Vec<NodePtr> {
    if let Some(hash) = maybe_program_hash {
        log.into_iter()
            .skip_while(|e| {
                if let Ok(program_hash) = program_hash_from_program_env_cons(allocator, *e) {
                    // Skip while we haven't found the hash we want.
                    program_hash.data() != hash.data()
                } else {
                    true
                }
            })
            .collect()
    } else {
        log
    }
}
