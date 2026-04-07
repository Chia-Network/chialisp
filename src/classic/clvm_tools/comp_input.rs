use std::collections::HashMap;
use std::rc::Rc;

use clvmr::{Allocator, NodePtr};

use crate::classic::clvm::__type_compatibility__::{Bytes, Stream, UnvalidatedBytesFromType};
use crate::classic::clvm::serialize::{sexp_from_stream, SimpleCreateCLVMObject};
use crate::classic::platform::argparse::ArgumentValue;

use crate::classic::clvm_tools::binutils::assemble_from_ir;
use crate::classic::clvm_tools::ir::reader::read_ir;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;

use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::{CompileErr, CompilerOpts};
use crate::compiler::debug::build_symbol_table_mut;
use crate::compiler::dialect::{detect_modern, AcceptedDialect};
use crate::compiler::optimize::maybe_finalize_program_via_classic_optimizer;
use crate::compiler::sexp;

pub fn get_disassembly_ver(p: &HashMap<String, ArgumentValue>) -> Option<usize> {
    if let Some(ArgumentValue::ArgInt(x)) = p.get("operators_version") {
        return Some(*x as usize);
    }

    None
}

fn get_string_and_filename_with_default(
    parsed_args: &HashMap<String, ArgumentValue>,
    argument: &str,
    default: Option<&str>,
) -> Option<(Option<String>, String)> {
    if let Some(ArgumentValue::ArgString(path, content)) = parsed_args.get(argument) {
        Some((path.clone(), content.to_string()))
    } else {
        default.map(|p| (None, p.to_string()))
    }
}

pub struct ParsedInputPathOrCode {
    pub path: Option<String>,
    pub content: String,
    pub parsed: NodePtr,
}

impl ParsedInputPathOrCode {
    pub fn use_filename(&self) -> String {
        self.path.clone().unwrap_or_else(|| "*command*".to_string())
    }
}

pub fn parse_tool_input_sexp(
    allocator: &mut Allocator,
    argument: &str,
    parsed_args: &HashMap<String, ArgumentValue>,
    default_hex: Option<&str>,
    default_sexp: Option<&str>,
) -> Result<ParsedInputPathOrCode, String> {
    match parsed_args.get("hex") {
        Some(_) => {
            let (path, sexp_text) = if let Some(r) =
                get_string_and_filename_with_default(parsed_args, argument, default_hex)
            {
                r
            } else {
                return Err("missing argument {argument}".to_string());
            };

            let use_sexp_text = if sexp_text.is_empty() {
                default_hex.unwrap_or_default()
            } else {
                &sexp_text
            }
            .to_string();

            let sexp_serialized =
                Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(use_sexp_text.clone())))
                    .map_err(|e| format!("{e:?}"))?;

            let mut prog_stream = Stream::new(Some(sexp_serialized));

            sexp_from_stream(
                allocator,
                &mut prog_stream,
                Box::new(SimpleCreateCLVMObject {}),
            )
            .map(|x| ParsedInputPathOrCode {
                path,
                content: use_sexp_text,
                parsed: x.1,
            })
            .map_err(|e| format!("{e:?}"))
        }
        _ => {
            let (path, use_sexp_text) = if let Some(r) =
                get_string_and_filename_with_default(parsed_args, argument, default_sexp)
            {
                r
            } else {
                return Err(format!("missing argument {argument}"));
            };

            read_ir(&use_sexp_text)
                .map_err(|e| format!("{e:?}"))
                .and_then(|v| {
                    Ok(ParsedInputPathOrCode {
                        path,
                        content: use_sexp_text,
                        parsed: assemble_from_ir(allocator, Rc::new(v))
                            .map_err(|e| format!("{e:?}"))?,
                    })
                })
        }
    }
}

pub struct RunAndCompileInputData {
    pub program: ParsedInputPathOrCode,
    pub args: ParsedInputPathOrCode,
    pub dialect: AcceptedDialect,
    pub opts: Rc<dyn CompilerOpts>,
    pub do_optimize: bool,
    pub search_paths: Vec<String>,
    pub symbol_table_output: String,
}

impl RunAndCompileInputData {
    pub fn new(
        allocator: &mut Allocator,
        parsed_args: &HashMap<String, ArgumentValue>,
    ) -> Result<RunAndCompileInputData, String> {
        let program = parse_tool_input_sexp(allocator, "path_or_code", parsed_args, None, None)?;
        let args = parse_tool_input_sexp(allocator, "env", parsed_args, Some("80"), Some("()"))?;
        let dialect = detect_modern(allocator, program.parsed);

        let do_optimize = parsed_args
            .get("optimize")
            .map(|x| matches!(x, ArgumentValue::ArgBool(true)))
            .unwrap_or_else(|| false);

        let search_paths = if let Some(ArgumentValue::ArgArray(v)) = parsed_args.get("include") {
            v.iter()
                .filter_map(|p| {
                    if let ArgumentValue::ArgString(_, s) = p {
                        Some(s.to_string())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut opts: Rc<dyn CompilerOpts> =
            Rc::new(DefaultCompilerOpts::new(&program.use_filename()))
                .set_dialect(dialect.clone())
                .set_search_paths(&search_paths)
                .set_optimize(do_optimize)
                .set_disassembly_ver(get_disassembly_ver(parsed_args));

        if let Some(stepping) = dialect.stepping {
            opts = opts
                .set_optimize(do_optimize || stepping > 22)
                .set_frontend_opt(stepping == 22);
        }

        let symbol_table_output = parsed_args
            .get("symbol_output_file")
            .and_then(|s| {
                if let ArgumentValue::ArgString(_, v) = s {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "main.sym".to_string());

        Ok(RunAndCompileInputData {
            program,
            args,
            dialect,
            do_optimize,
            search_paths,
            opts,
            symbol_table_output,
        })
    }

    pub fn use_filename(&self) -> String {
        self.program
            .path
            .clone()
            .unwrap_or_else(|| "*command*".to_string())
    }

    pub fn compile_modern(
        &self,
        allocator: &mut Allocator,
        symbol_table: &mut HashMap<String, String>,
    ) -> Result<Rc<sexp::SExp>, CompileErr> {
        let runner = Rc::new(DefaultProgramRunner::new());

        let unopt_res = compile_file(
            allocator,
            runner.clone(),
            self.opts.clone(),
            &self.program.content,
            symbol_table,
        );
        let res = unopt_res.and_then(|x| {
            maybe_finalize_program_via_classic_optimizer(
                allocator,
                runner,
                self.opts.clone(),
                self.do_optimize,
                &x,
            )
        })?;

        build_symbol_table_mut(symbol_table, &res);

        Ok(res)
    }
}
