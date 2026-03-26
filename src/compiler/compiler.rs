use num_bigint::ToBigInt;

use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io;
// use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::UNIX_EPOCH;

use clvm_rs::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm_tools::ir::r#type::NEW_BIT_CONSTANTS;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::classic::clvm::__type_compatibility__::Stream;
use crate::classic::clvm::sexp::sexp_as_bin;
use crate::compiler::cldb::hex_to_modern_sexp;
use crate::compiler::clvm::{convert_to_clvm_rs, run, sha256tree, NewStyleIntConversion};
use crate::compiler::codegen::{codegen, hoist_body_let_binding, process_helper_let_bindings};
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompileModuleComponent, CompileModuleOutput, CompilerOpts,
    CompilerOutput, ConstantKind, DefunData, Export, FrontendOutput, HelperForm, ImportLongName,
    IncludeDesc, IncludeProcessType, ModulePhase, PrimaryCodegen, StandalonePhaseInfo,
    SyntheticType,
};
use crate::compiler::dialect::{AcceptedDialect, KNOWN_DIALECTS};
use crate::compiler::diskcache::{set_cache_element, try_element_from_cache};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::depgraph::{DepgraphOptions, FunctionDependencyGraph};
use crate::compiler::optimize::get_optimizer;
use crate::compiler::prims;
use crate::compiler::resolve::{find_helper_target, resolve_namespaces};
use crate::compiler::sexp::{decode_string, enlist, parse_sexp_flags, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::{BasicCompileContext, CompileContextWrapper};
use crate::util::Number;

pub const SHA256TREE_PROGRAM_CLVM: &str = "(2 (1 2 (3 (7 5) (1 11 (1 . 2) (2 2 (4 2 (4 9 ()))) (2 2 (4 2 (4 13 ())))) (1 11 (1 . 1) 5)) 1) (4 (1 2 (3 (7 5) (1 11 (1 . 2) (2 2 (4 2 (4 9 ()))) (2 2 (4 2 (4 13 ())))) (1 11 (1 . 1) 5)) 1) 1))";

pub const FUZZ_TEST_PRE_CSE_MERGE_FIX_FLAG: usize = 1;

lazy_static! {
    pub static ref STANDARD_MACROS: String = {
        indoc! {"(
            (defmacro if (A B C) (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @)))
            (defmacro list ARGS
                            (defun compile-list
                                   (args)
                                   (if args
                                       (qq (c (unquote (f args))
                                             (unquote (compile-list (r args)))))
                                       ()))
                            (compile-list ARGS)
                    )
            (defun-inline / (A B) (f (divmod A B)))
            )
            "}
        .to_string()
    };
    pub static ref ADVANCED_MACROS: String = {
        indoc! {"(
            (defmac __chia__primitive__if (A B C)
              (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @))
              )

            (defun __chia__if (ARGS)
              (__chia__primitive__if (r (r (r ARGS)))
                (qq (a (i (unquote (f ARGS)) (com (unquote (f (r ARGS)))) (com (unquote (__chia__if (r (r ARGS)))))) @))
                (qq (a (i (unquote (f ARGS)) (com (unquote (f (r ARGS)))) (com (unquote (f (r (r ARGS)))))) @))
                )
              )

            (defmac if ARGS (__chia__if ARGS))

            (defun __chia__compile-list (args)
              (if args
                (c 4 (c (f args) (c (__chia__compile-list (r args)) ())))
                ()
                )
              )

            (defmac list ARGS (__chia__compile-list ARGS))

            (defun-inline / (A B) (f (divmod A B)))
            )
            "}
        .to_string()
    };
}

#[derive(Clone, Debug)]
pub struct DefaultCompilerOpts {
    pub include_dirs: Vec<String>,
    pub filename: String,
    pub code_generator: Option<PrimaryCodegen>,
    pub in_defun: bool,
    pub stdenv: bool,
    pub optimize: bool,
    pub frontend_opt: bool,
    pub frontend_check_live: bool,
    pub start_env: Option<Rc<SExp>>,
    pub disassembly_ver: Option<usize>,
    pub prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    pub diag_flags: Rc<HashSet<usize>>,
    pub dialect: AcceptedDialect,
    pub module_phase: Option<ModulePhase>,
}

pub fn create_prim_map() -> Rc<HashMap<Vec<u8>, Rc<SExp>>> {
    let mut prim_map: HashMap<Vec<u8>, Rc<SExp>> = HashMap::new();

    for p in prims::prims() {
        prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
    }

    Rc::new(prim_map)
}

pub fn do_desugar(
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
) -> Result<CompileForm, CompileErr> {
    // Transform let bindings, merging nested let scopes with the top namespace
    let hoisted_bindings = hoist_body_let_binding(
        opts.clone(),
        None,
        program.args.clone(),
        program.exp.clone(),
    )?;
    let mut new_helpers = hoisted_bindings.0;
    let expr = hoisted_bindings.1; // expr is the let-hoisted program

    // TODO: Distinguish the frontend_helpers and the hoisted_let helpers for later stages
    let mut combined_helpers = program.helpers.clone();
    combined_helpers.append(&mut new_helpers);
    let combined_helpers = process_helper_let_bindings(opts, &combined_helpers)?;

    Ok(CompileForm {
        helpers: combined_helpers,
        exp: expr,
        ..program.clone()
    })
}

pub fn finish_compilation(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    p2: CompileForm,
) -> Result<SExp, CompileErr> {
    let p3 = context.post_desugar_optimization(opts.clone(), p2)?;

    // generate code from AST, optionally with optimization
    let generated = codegen(context, opts.clone(), None, &p3)?;

    let g2 = context.post_codegen_output_optimize(opts, generated)?;

    Ok(g2)
}

pub fn compile_from_compileform(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    p0: CompileForm,
) -> Result<SExp, CompileErr> {
    let p1 = context.frontend_optimization(opts.clone(), p0)?;

    // Resolve includes, convert program source to lexemes
    let p2 = do_desugar(opts.clone(), &p1)?;

    finish_compilation(context, opts, p2)
}

fn create_hex_output_path(loc: Srcloc, file_path: &str, func: &str) -> Result<String, CompileErr> {
    let mut dir = PathBuf::from(file_path);
    let filename = PathBuf::from(file_path)
        .with_extension("")
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "program".to_string());
    dir.pop();
    let func_dot_hex_list = &[func.to_string(), "hex".to_string()];
    let func_dot_hex = func_dot_hex_list.join(".");
    let name_with_func_list = &[filename.to_string(), func_dot_hex];
    dir.push(name_with_func_list.join("_"));
    dir.into_os_string().into_string().map_err(|_| {
        CompileErr(
            loc,
            format!("could not make os file path for output {func}"),
        )
    })
}

pub fn find_exported_helper(
    opts: Rc<dyn CompilerOpts>,
    program: &CompileForm,
    fun_name: &[u8],
) -> Result<Option<HelperForm>, CompileErr> {
    let (_, parsed_name) = ImportLongName::parse(fun_name);
    Ok(
        find_helper_target(opts.clone(), &program.helpers, None, fun_name, &parsed_name)?
            .map(|(_, result)| result.clone()),
    )
}

fn modernize_constants(helpers: &mut [HelperForm], standalone_constants: &HashSet<Vec<u8>>) {
    for h in helpers.iter_mut() {
        match h {
            HelperForm::Defconstant(d) => {
                // Ensure that we upgrade the constant type.
                d.tabled = false;
                if standalone_constants.contains(&d.name) {
                    d.kind = ConstantKind::Module(false);
                }
            }
            HelperForm::Defnamespace(ns) => {
                modernize_constants(&mut ns.helpers, standalone_constants);
            }
            _ => {}
        }
    }
}

fn capture_standalone_constants(
    standalone_constants: &mut HashSet<Vec<u8>>,
    depgraph: &FunctionDependencyGraph,
    helpers: &[HelperForm],
    exports: &[Export],
) {
    // Find constants on which nothing depends (they're only output).
    for h in helpers.iter() {
        if let HelperForm::Defnamespace(ns) = h {
            capture_standalone_constants(standalone_constants, depgraph, &ns.helpers, exports)
        } else if matches!(h, HelperForm::Defconstant(_) | HelperForm::Defun(_, _)) {
            let match_exports = exports.iter().any(|e| match e {
                Export::MainProgram(_) => false,
                Export::Function(exdef) => &exdef.name.value == h.name(),
            });

            // It isn't exported so it isn't standalone.
            if !match_exports {
                continue;
            }

            let mut constant_is_depended = HashSet::new();
            depgraph.get_full_depended_on_by(&mut constant_is_depended, h.name());
            if constant_is_depended.is_empty() {
                standalone_constants.insert(h.name().to_vec());
            }
        }
    }
}

fn get_hex_name_of_export(
    opts: Rc<dyn CompilerOpts>,
    loc: &Srcloc,
    export: &Export,
) -> Result<String, CompileErr> {
    match export {
        Export::MainProgram(_) => {
            let mut output_path = PathBuf::from(&opts.filename());
            output_path.set_extension("hex");
            Ok(output_path.into_os_string().to_string_lossy().to_string())
        }
        Export::Function(desc) => {
            let use_name = decode_string(&desc.as_name.as_ref().unwrap_or(&desc.name).value);
            create_hex_output_path(loc.clone(), &opts.filename(), &use_name)
        }
    }
}

fn form_module_program_common_body(
    standalone_constants: &HashSet<Vec<u8>>,
    mut program: CompileForm,
    exports: &[Export],
) -> Result<CompileForm, CompileErr> {
    program
        .helpers
        .retain(|h| !standalone_constants.contains(h.name()));

    // The body should contain anything that is in the exports but not standalone
    // constants.
    let mut body = Rc::new(BodyForm::Value(SExp::Nil(program.loc())));
    let cons = Rc::new(BodyForm::Value(SExp::Integer(
        program.loc(),
        4_u32.to_bigint().unwrap(),
    )));

    // XXX Give exports locations.
    let hash_loc = program.loc.clone();
    let add_export = |body: &mut Rc<BodyForm>, target_name: &[u8], capture: &[u8]| {
        *body = Rc::new(BodyForm::Call(
            hash_loc.clone(),
            vec![
                cons.clone(),
                Rc::new(BodyForm::Call(
                    hash_loc.clone(),
                    vec![
                        cons.clone(),
                        Rc::new(BodyForm::Value(SExp::QuotedString(
                            hash_loc.clone(),
                            b'"',
                            target_name.to_vec(),
                        ))),
                        Rc::new(BodyForm::Value(SExp::Atom(
                            hash_loc.clone(),
                            capture.to_vec(),
                        ))),
                    ],
                    None,
                )),
                body.clone(),
            ],
            None,
        ));
    };

    for (target_name, capture) in exports.iter().filter_map(|e| {
        if let Export::Function(exdef) = e {
            let target_name = exdef.as_name.as_ref().unwrap_or(&exdef.name).value.clone();
            if !standalone_constants.contains(&exdef.name.value) {
                return Some((target_name, exdef.name.value.clone()));
            }
        }

        None
    }) {
        add_export(&mut body, &target_name, &capture);
    }

    program.exp = body;

    Ok(program)
}

fn populate_export_map(
    context: &mut BasicCompileContext,
    export_map: &mut BTreeMap<Vec<u8>, Rc<SExp>>,
    opts: Rc<dyn CompilerOpts>,
    code: Rc<SExp>,
) -> Result<(), CompileErr> {
    let runner = context.runner.clone();
    let mut result = run(
        context.allocator(),
        runner,
        opts.prim_map(),
        code.clone(),
        Rc::new(SExp::Nil(code.loc())),
        None,
        None,
    )?;

    while let SExp::Cons(_, first, rest) = result.borrow() {
        if let SExp::Cons(_, name, value) = first.borrow() {
            if let SExp::Atom(_, name) = name.atomize().borrow() {
                let mut hash_name: Vec<u8> = name.clone();
                hash_name.append(&mut b"_hash".to_vec());
                export_map.insert(
                    hash_name,
                    Rc::new(SExp::Atom(value.loc(), sha256tree(value.clone()))),
                );
                export_map.insert(name.clone(), value.clone());
            }
        }
        result = rest.clone();
    }

    Ok(())
}

/// Exports are returned main programs:
///
/// Single main
///
/// (export (X) (do-stuff X))
///
/// Multiple mains
///
/// (export foo)
/// (export bar)
pub fn compile_module(
    context: &mut BasicCompileContext,
    mut opts: Rc<dyn CompilerOpts>,
    standalone_constants: &HashSet<Vec<u8>>,
    mut program: CompileForm,
    exports: &[Export],
) -> Result<CompileModuleOutput, CompileErr> {
    let loc = program.loc();
    let mut dialect = opts.dialect();
    dialect.int_fix = true;
    dialect.stepping = dialect.stepping.map(|x| std::cmp::max(x, 25));

    opts = opts.set_optimize(true).set_dialect(dialect);

    if exports.is_empty() {
        return Err(CompileErr(
            loc.clone(),
            "A chialisp module should have at least one export".to_string(),
        ));
    }

    if exports.len() == 1 {
        if let Export::MainProgram(desc) = &exports[0] {
            // Single program.
            program.args = desc.args.clone();
            program.exp = desc.expr.clone();

            program = resolve_namespaces(opts.clone(), &program)?;
            modernize_constants(&mut program.helpers, standalone_constants);

            let output = Rc::new(compile_from_compileform(
                context,
                opts.clone(),
                program.clone(),
            )?);
            let converted = convert_to_clvm_rs(context.allocator(), output.clone())?;

            let mut output_path = PathBuf::from(&opts.filename());
            output_path.set_extension("hex");
            let output_path_str = output_path.into_os_string().to_string_lossy().to_string();
            let mut stream = Stream::new(None);
            stream.write(sexp_as_bin(context.allocator(), converted));
            let hex_data = stream.get_value().hex();
            set_cache_element(opts.clone(), &program, &output_path_str, &hex_data);
            opts.write_new_file(&output_path_str, hex_data.as_bytes())?;

            let (hash, summary) = compute_export_summary(
                loc.clone(),
                Rc::new(SExp::Nil(loc.clone())),
                b"program",
                output.clone(),
            );

            return Ok(CompileModuleOutput {
                summary,
                includes: program.include_forms.clone(),
                components: vec![CompileModuleComponent {
                    shortname: b"program".to_vec(),
                    filename: output_path_str,
                    content: output.clone(),
                    hash,
                }],
            });
        }
    }

    // Add hash functions for exports.
    let hash_loc = program.loc();
    for e in exports.iter() {
        if let Export::Function(exdef) = &e {
            if !standalone_constants.contains(&exdef.name.value) {
                add_inline_hash_for_constant(&mut program, &hash_loc, &exdef.name.value);
            }
        }
    }

    // First pass compilation: remove standalone constant helpers and produce
    // a body which contains all the non-standalone exports.

    let common_opts_assume_introspection =
        opts.set_module_phase(Some(ModulePhase::CommonPhase(true)));
    let common_program_assume_introspection = resolve_namespaces(
        common_opts_assume_introspection.clone(),
        &form_module_program_common_body(standalone_constants, program.clone(), exports)?,
    )?;

    let common_phase_functions = common_program_assume_introspection
        .helpers
        .iter()
        .any(|h| matches!(h, HelperForm::Defun(_, _)));
    let (common_opts, mut common_program) = if !common_phase_functions {
        let new_opts =
            opts.set_module_phase(Some(ModulePhase::CommonPhase(common_phase_functions)));
        (
            new_opts.clone(),
            resolve_namespaces(
                new_opts,
                &form_module_program_common_body(standalone_constants, program.clone(), exports)?,
            )?,
        )
    } else {
        (
            common_opts_assume_introspection.clone(),
            common_program_assume_introspection.clone(),
        )
    };
    modernize_constants(&mut common_program.helpers, standalone_constants);
    let common_output = compile_from_compileform(context, common_opts, common_program.clone())?;

    let mut captured_export_map: BTreeMap<Vec<u8>, Rc<SExp>> = BTreeMap::new();
    // Capture exports that are members of the common set.
    // We get a triple of output: (env_shape env code)
    let (env_shape, env, code) = (|| {
        if let Some(lst) = common_output.proper_list() {
            if lst.len() == 3 {
                return Ok((
                    Rc::new(lst[0].clone()),
                    Rc::new(lst[1].clone()),
                    Rc::new(lst[2].clone()),
                ));
            }
        }

        Err(CompileErr(
            common_program.loc(),
            format!(
                "Wrong environment shape result from common phase code generation: {common_output}"
            ),
        ))
    })()?;

    populate_export_map(context, &mut captured_export_map, opts.clone(), code)?;

    // Second pass compilation: for each export in standalone constants
    let cons = Rc::new(BodyForm::Value(SExp::Integer(
        program.loc(),
        4_u32.to_bigint().unwrap(),
    )));
    let second_stage_opts =
        opts.set_module_phase(Some(ModulePhase::StandalonePhase(StandalonePhaseInfo {
            env: env_shape,
            empty_common_phase: !common_phase_functions,
            left_env_value: env,
        })));
    for fun in exports.iter() {
        let (fun_name, export_name) = if let Export::Function(exdef) = fun {
            // Otherwise, capture it to produce to the output.
            (
                exdef.name.value.clone(),
                exdef
                    .as_name
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| exdef.name.clone())
                    .value
                    .to_vec(),
            )
        } else {
            return Err(CompileErr(
                loc.clone(),
                "got program, wanted fun".to_string(),
            ));
        };

        let second_stage_program = if let Some(h) =
            find_exported_helper(opts.clone(), &program, &fun_name)?
        {
            CompileForm {
                exp: Rc::new(BodyForm::Call(
                    h.loc(),
                    vec![
                        cons.clone(),
                        Rc::new(BodyForm::Call(
                            h.loc(),
                            vec![
                                cons.clone(),
                                Rc::new(BodyForm::Value(SExp::QuotedString(
                                    h.loc(),
                                    b'"',
                                    export_name.to_vec(),
                                ))),
                                Rc::new(BodyForm::Value(SExp::Atom(h.loc(), fun_name.to_vec()))),
                            ],
                            None,
                        )),
                        Rc::new(BodyForm::Value(SExp::Nil(h.loc()))),
                    ],
                    None,
                )),
                ..program.clone()
            }
        } else {
            return Err(CompileErr(
                program.loc(),
                format!(
                    "export helper {} not present while generating standalone constant code",
                    decode_string(&fun_name)
                ),
            ));
        };

        // remove_standalone_constant(&mut second_stage_program, &fun_name);
        let mut constant_culled_second_stage_program =
            resolve_namespaces(second_stage_opts.clone(), &second_stage_program)?;
        modernize_constants(
            &mut constant_culled_second_stage_program.helpers,
            standalone_constants,
        );

        let compiled_result = Rc::new(compile_from_compileform(
            context,
            second_stage_opts.clone(),
            constant_culled_second_stage_program,
        )?);

        populate_export_map(
            context,
            &mut captured_export_map,
            opts.clone(),
            compiled_result,
        )?;
    }

    // Components to use for the CompileModuleOutput, which downstream can be
    // collected for namespacing.
    let mut components = vec![];
    let mut prog_output = SExp::Nil(program.loc());

    for (export_name, export_value) in captured_export_map.iter() {
        let output_path =
            create_hex_output_path(loc.clone(), &opts.filename(), &decode_string(export_name))?;

        let m = CompileModuleComponent {
            shortname: export_name.to_vec(),
            filename: output_path.clone(),
            content: export_value.clone(),
            hash: sha256tree(export_value.clone()),
        };

        prog_output = SExp::Cons(
            loc.clone(),
            Rc::new(SExp::Cons(
                loc.clone(),
                Rc::new(SExp::Atom(loc.clone(), m.shortname.clone())),
                Rc::new(SExp::QuotedString(loc.clone(), b'x', m.hash.clone())),
            )),
            Rc::new(prog_output),
        );

        let mut stream = Stream::new(None);
        let converted_func = convert_to_clvm_rs(context.allocator(), m.content.clone())?;
        stream.write(sexp_as_bin(context.allocator(), converted_func));
        let hex_data = stream.get_value().hex();
        set_cache_element(opts.clone(), &program, &output_path, &hex_data);
        opts.write_new_file(&output_path, hex_data.as_bytes())?;

        components.push(m);
    }

    Ok(CompileModuleOutput {
        summary: Rc::new(prog_output),
        includes: program.include_forms.clone(),
        components,
    })
}

fn compute_export_summary(
    loc: Srcloc,
    list_tail: Rc<SExp>,
    shortname: &[u8],
    program_code: Rc<SExp>,
) -> (Vec<u8>, Rc<SExp>) {
    let hash = sha256tree(program_code);
    (
        hash.clone(),
        Rc::new(SExp::Cons(
            loc.clone(),
            Rc::new(SExp::Cons(
                loc.clone(),
                Rc::new(SExp::QuotedString(loc.clone(), b'"', shortname.to_vec())),
                Rc::new(SExp::QuotedString(loc, b'x', hash)),
            )),
            list_tail,
        )),
    )
}

pub fn try_from_cache(
    opts: Rc<dyn CompilerOpts>,
    cf: &CompileForm,
    exports: &[Export],
) -> Result<Option<CompilerOutput>, CompileErr> {
    let mut allocator = Allocator::new();
    let mut components = Vec::new();
    let mut summary = Rc::new(SExp::Nil(cf.loc.clone()));
    let mut data_to_write = Vec::new();

    for e in exports.iter() {
        let hex_file_name = { get_hex_name_of_export(opts.clone(), &cf.loc(), e) }?;
        let hex_data = if let Some(hd) = try_element_from_cache(opts.clone(), cf, &hex_file_name) {
            hd
        } else {
            return Ok(None);
        };

        let loaded_hex_data =
            // Don't fail on failure to load cache.  We can compile and regenerate.
            if let Ok(lh) = hex_to_modern_sexp(&mut allocator, &HashMap::new(), cf.loc.clone(), &hex_data) {
                lh
            } else {
                return Ok(None);
            };

        data_to_write.push((hex_file_name.clone(), hex_data.clone()));

        let shortname = if let Export::Function(desc) = e {
            desc.name.value.clone()
        } else {
            b"program".to_vec()
        };

        let (hash, new_summary) =
            compute_export_summary(cf.loc(), summary, &shortname, loaded_hex_data.clone());
        summary = new_summary;

        components.push(CompileModuleComponent {
            shortname,
            filename: hex_file_name,
            content: loaded_hex_data,
            hash,
        });
    }

    // if we got here, then we loaded all exports.
    // write (or rewrite) any hex files that were outputs of the elided build steps.
    let mut allocator = Allocator::new();
    let empty_symbols = HashMap::new();
    for (hex_file_name, hex_data) in data_to_write.into_iter() {
        opts.write_new_file(&hex_file_name, hex_data.as_bytes())?;
        // Write a hash file alongside.
        if !hex_file_name.ends_with(".hex") {
            continue;
        }

        let decoded_hex = hex_to_modern_sexp(&mut allocator, &empty_symbols, cf.loc(), &hex_data)?;
        let treehash = sha256tree(decoded_hex);
        let hash_file_name = format!("{}_hash.hex", &hex_file_name[0..hex_file_name.len() - 4]);
        let treehash_hex = hex::encode(&treehash);
        opts.write_new_file(&hash_file_name, treehash_hex.as_bytes())?;
    }

    Ok(Some(CompilerOutput::Module(CompileModuleOutput {
        summary,
        components,
        includes: cf.include_forms.clone(),
    })))
}

fn add_main_fingerprint(cf: &mut CompileForm, forms: &[Rc<SExp>]) {
    let form_list = Rc::new(enlist(cf.loc(), forms));
    cf.include_forms.push(IncludeDesc {
        kw: cf.loc(),
        nl: cf.loc(),
        name: b"main".to_vec(),
        kind: Some(IncludeProcessType::Compiled),
        fingerprint: sha256tree(form_list),
    });
}

fn form_hash_expression(inner_exp: Rc<BodyForm>) -> Rc<BodyForm> {
    let shloc = Srcloc::start("*sha256tree*");
    let parsed = parse_sexp_flags(shloc.clone(), SHA256TREE_PROGRAM_CLVM.bytes(), 0)
        .expect("should have parsed");
    let p0_borrowed: &SExp = parsed[0].borrow();

    Rc::new(BodyForm::Call(
        inner_exp.loc(),
        vec![
            Rc::new(BodyForm::Value(SExp::Integer(
                inner_exp.loc(),
                2_u32.to_bigint().unwrap(),
            ))),
            Rc::new(BodyForm::Quoted(p0_borrowed.clone())),
            Rc::new(BodyForm::Call(
                inner_exp.loc(),
                vec![
                    Rc::new(BodyForm::Value(SExp::Integer(
                        inner_exp.loc(),
                        4_u32.to_bigint().unwrap(),
                    ))),
                    inner_exp.clone(),
                    Rc::new(BodyForm::Quoted(SExp::Nil(inner_exp.loc()))),
                ],
                None,
            )),
        ],
        None,
    ))
}

fn add_inline_hash_for_constant(program: &mut CompileForm, loc: &Srcloc, fun_name: &[u8]) {
    let mut new_name = fun_name.to_vec();
    new_name.extend(b"_hash".to_vec());

    program.helpers.push(HelperForm::Defun(
        true,
        Box::new(DefunData {
            loc: loc.clone(),
            nl: loc.clone(),
            kw: None,
            name: new_name.clone(),
            args: Rc::new(SExp::Nil(loc.clone())),
            orig_args: Rc::new(SExp::Nil(loc.clone())),
            body: form_hash_expression(Rc::new(BodyForm::Value(SExp::Atom(
                loc.clone(),
                fun_name.to_vec(),
            )))),
            synthetic: Some(SyntheticType::WantInline),
        }),
    ));
}

pub fn compile_pre_forms(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    pre_forms: &[Rc<SExp>],
) -> Result<CompilerOutput, CompileErr> {
    let p0 = frontend(opts.clone(), pre_forms)?;

    match p0 {
        FrontendOutput::CompileForm(p0) => Ok(CompilerOutput::Program(
            p0.include_forms.clone(),
            compile_from_compileform(context, opts, p0)?,
        )),
        FrontendOutput::Module(mut cf, exports) => {
            add_main_fingerprint(&mut cf, pre_forms);
            // If we can read from cache, use that.  We'll use the actual form of the compileform
            // and opts (the dialect) to determine a cache hit.
            if let Some(result) = try_from_cache(opts.clone(), &cf, &exports)? {
                return Ok(result);
            }

            // cl23 always reflects optimization.
            let dialect = opts.dialect();
            let opts = if let Some(stepping) = dialect.stepping.as_ref() {
                opts.set_optimize(*stepping > 21)
            } else {
                opts
            };

            // We make a dependency graph of constants and functions.  There must
            // be a solveable hierarchy for constants, that is some must be top
            // level constants that are not depended on.  Thse will not be tabled
            // with the rest, but computed once.  Practially, these will be
            // constants that are computed only for export and no other constants
            // or functions depend on them.  Specifically, any constant that is
            // not used by a constant or function is forced to be inline.  We will
            // expand it when generating the constant output.
            let depgraph = FunctionDependencyGraph::new_with_options(
                &cf,
                DepgraphOptions {
                    with_constants: true,
                },
            );

            let mut standalone_constants = HashSet::new();

            capture_standalone_constants(
                &mut standalone_constants,
                &depgraph,
                &cf.helpers,
                &exports,
            );
            let result_form = CompilerOutput::Module(compile_module(
                context,
                opts.clone(),
                &standalone_constants,
                cf.clone(),
                &exports,
            )?);
            Ok(result_form)
        }
    }
}

pub fn compile_file(
    allocator: &mut Allocator,
    runner: Rc<dyn TRunProgram>,
    opts: Rc<dyn CompilerOpts>,
    content: &str,
    symbol_table: &mut HashMap<String, String>,
) -> Result<CompilerOutput, CompileErr> {
    let _int_conversion_bug = NewStyleIntConversion::new(opts.dialect().int_fix);
    let srcloc = Srcloc::start(&opts.filename());
    let flags = if opts.dialect().extra_numeric_constants {
        NEW_BIT_CONSTANTS
    } else {
        0
    };
    let pre_forms = parse_sexp_flags(srcloc.clone(), content.bytes(), flags)?;
    let mut context_wrapper = CompileContextWrapper::new(
        allocator,
        runner,
        symbol_table,
        get_optimizer(&srcloc, opts.clone())?,
    );

    compile_pre_forms(&mut context_wrapper.context, opts, &pre_forms)
}

impl CompilerOpts for DefaultCompilerOpts {
    fn filename(&self) -> String {
        self.filename.clone()
    }
    fn code_generator(&self) -> Option<PrimaryCodegen> {
        self.code_generator.clone()
    }
    fn dialect(&self) -> AcceptedDialect {
        self.dialect.clone()
    }
    fn in_defun(&self) -> bool {
        self.in_defun
    }
    fn stdenv(&self) -> bool {
        self.stdenv
    }
    fn optimize(&self) -> bool {
        self.optimize
    }
    fn frontend_opt(&self) -> bool {
        self.frontend_opt
    }
    fn module_phase(&self) -> Option<ModulePhase> {
        self.module_phase.clone()
    }
    fn frontend_check_live(&self) -> bool {
        self.frontend_check_live
    }
    fn start_env(&self) -> Option<Rc<SExp>> {
        self.start_env.clone()
    }
    fn prim_map(&self) -> Rc<HashMap<Vec<u8>, Rc<SExp>>> {
        self.prim_map.clone()
    }
    fn disassembly_ver(&self) -> Option<usize> {
        self.disassembly_ver
    }
    fn get_search_paths(&self) -> Vec<String> {
        self.include_dirs.clone()
    }
    fn diag_flags(&self) -> Rc<HashSet<usize>> {
        self.diag_flags.clone()
    }

    fn set_filename(&self, filename: &str) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.filename = filename.to_string();
        Rc::new(copy)
    }
    fn set_dialect(&self, dialect: AcceptedDialect) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.dialect = dialect;
        Rc::new(copy)
    }
    fn set_search_paths(&self, dirs: &[String]) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        dirs.clone_into(&mut copy.include_dirs);
        Rc::new(copy)
    }
    fn set_disassembly_ver(&self, ver: Option<usize>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.disassembly_ver = ver;
        Rc::new(copy)
    }
    fn set_in_defun(&self, new_in_defun: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.in_defun = new_in_defun;
        Rc::new(copy)
    }
    fn set_stdenv(&self, new_stdenv: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.stdenv = new_stdenv;
        Rc::new(copy)
    }
    fn set_optimize(&self, optimize: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.optimize = optimize;
        Rc::new(copy)
    }
    fn set_frontend_opt(&self, optimize: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.frontend_opt = optimize;
        Rc::new(copy)
    }
    fn set_frontend_check_live(&self, check: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.frontend_check_live = check;
        Rc::new(copy)
    }
    fn set_module_phase(&self, module_phase: Option<ModulePhase>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.module_phase = module_phase;
        Rc::new(copy)
    }
    fn set_code_generator(&self, new_code_generator: PrimaryCodegen) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.code_generator = Some(new_code_generator);
        Rc::new(copy)
    }
    fn set_start_env(&self, start_env: Option<Rc<SExp>>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.start_env = start_env;
        Rc::new(copy)
    }
    fn set_prim_map(&self, prims: Rc<HashMap<Vec<u8>, Rc<SExp>>>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.prim_map = prims;
        Rc::new(copy)
    }
    fn set_diag_flags(&self, flags: Rc<HashSet<usize>>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.diag_flags = flags;
        Rc::new(copy)
    }

    fn read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr> {
        if filename == "*macros*" {
            if self.dialect().strict {
                return Ok((filename, ADVANCED_MACROS.bytes().collect()));
            } else {
                return Ok((filename, STANDARD_MACROS.bytes().collect()));
            }
        } else if let Some(dialect) = KNOWN_DIALECTS.get(&filename) {
            return Ok((filename, dialect.content.bytes().collect()));
        }

        for dir in self.include_dirs.iter() {
            let mut p = PathBuf::from(dir);
            p.push(filename.clone());
            match fs::read(p.clone()) {
                Err(_e) => {
                    continue;
                }
                Ok(content) => {
                    return Ok((
                        p.to_str().map(|x| x.to_owned()).unwrap_or_else(|| filename),
                        content,
                    ));
                }
            }
        }
        Err(CompileErr(
            Srcloc::start(&inc_from),
            format!("could not find {filename} to include"),
        ))
    }

    fn get_file_mod_date(&self, loc: &Srcloc, filename: &str) -> Result<u64, CompileErr> {
        fs::metadata(filename)
            .map_err(|e| format!("could not get metadata for {filename}: {e:?}"))
            .and_then(|m| {
                m.modified()
                    .map_err(|e| format!("could not get modified time for {filename}: {e:?}"))
            })
            .and_then(|m| {
                m.duration_since(UNIX_EPOCH)
                    .map_err(|e| format!("Could not convert modified time of {filename} to seconds since unix epoch: {e:?}"))
            })
            .map(|m| m.as_secs())
            .map_err(|e| CompileErr(loc.clone(), e))
    }

    fn write_new_file(&self, target: &str, content: &[u8]) -> Result<(), CompileErr> {
        let path = Path::new(target);
        let parent_dir = path.parent();
        (|| {
            if let Some(p) = parent_dir {
                fs::create_dir_all(p)?;
            }
            fs::write(target, content)?;
            Ok(())
        })()
        .map_err(|e: io::Error| {
            CompileErr(
                Srcloc::start(&self.filename()),
                format!(
                    "could not write output file {} for {}, error {e:?}",
                    target,
                    self.filename()
                ),
            )
        })
    }

    fn compile_program(
        &self,
        context: &mut BasicCompileContext,
        sexp: Rc<SExp>,
    ) -> Result<CompilerOutput, CompileErr> {
        let _int_conversion_bug = NewStyleIntConversion::new(self.dialect.int_fix);
        let me = Rc::new(self.clone());
        let runner = context.runner.clone();
        let mut context_wrapper = CompileContextWrapper::new(
            &mut context.allocator,
            runner,
            &mut context.symbols,
            get_optimizer(&sexp.loc(), me.clone())?,
        );
        compile_pre_forms(&mut context_wrapper.context, me, &[sexp])
    }
}

impl DefaultCompilerOpts {
    pub fn new(filename: &str) -> DefaultCompilerOpts {
        DefaultCompilerOpts {
            include_dirs: vec![".".to_string()],
            filename: filename.to_string(),
            code_generator: None,
            in_defun: false,
            stdenv: true,
            optimize: false,
            frontend_opt: false,
            frontend_check_live: true,
            start_env: None,
            dialect: AcceptedDialect::default(),
            prim_map: create_prim_map(),
            disassembly_ver: None,
            module_phase: None,
            diag_flags: Rc::new(HashSet::default()),
        }
    }
}

fn path_to_function_inner(
    program: Rc<SExp>,
    hash: &[u8],
    path_mask: Number,
    current_path: Number,
) -> Option<Number> {
    let nextpath = path_mask.clone() * 2_i32.to_bigint().unwrap();
    match program.borrow() {
        SExp::Cons(_, a, b) => {
            path_to_function_inner(a.clone(), hash, nextpath.clone(), current_path.clone())
                .map(Some)
                .unwrap_or_else(|| {
                    path_to_function_inner(
                        b.clone(),
                        hash,
                        nextpath.clone(),
                        current_path.clone() + path_mask.clone(),
                    )
                    .map(Some)
                    .unwrap_or_else(|| {
                        let current_hash = sha256tree(program.clone());
                        if current_hash == hash {
                            Some(current_path + path_mask)
                        } else {
                            None
                        }
                    })
                })
        }
        _ => {
            let current_hash = sha256tree(program.clone());
            if current_hash == hash {
                Some(current_path + path_mask)
            } else {
                None
            }
        }
    }
}

pub fn path_to_function(program: Rc<SExp>, hash: &[u8]) -> Option<Number> {
    path_to_function_inner(program, hash, bi_one(), bi_zero())
}

fn op2(op: u32, code: Rc<SExp>, env: Rc<SExp>) -> Rc<SExp> {
    Rc::new(SExp::Cons(
        code.loc(),
        Rc::new(SExp::Integer(env.loc(), op.to_bigint().unwrap())),
        Rc::new(SExp::Cons(
            code.loc(),
            code.clone(),
            Rc::new(SExp::Cons(
                env.loc(),
                env.clone(),
                Rc::new(SExp::Nil(code.loc())),
            )),
        )),
    ))
}

fn quoted(env: Rc<SExp>) -> Rc<SExp> {
    Rc::new(SExp::Cons(
        env.loc(),
        Rc::new(SExp::Integer(env.loc(), bi_one())),
        env.clone(),
    ))
}

fn apply(code: Rc<SExp>, env: Rc<SExp>) -> Rc<SExp> {
    op2(2, code, env)
}

fn cons(f: Rc<SExp>, r: Rc<SExp>) -> Rc<SExp> {
    op2(4, f, r)
}

// compose (a (a path env) (c env 1))
pub fn rewrite_in_program(path: Number, env: Rc<SExp>) -> Rc<SExp> {
    apply(
        apply(
            // Env comes quoted, so divide by 2
            quoted(Rc::new(SExp::Integer(env.loc(), path / 2))),
            env.clone(),
        ),
        cons(env.clone(), Rc::new(SExp::Integer(env.loc(), bi_one()))),
    )
}

pub fn is_operator(op: u32, atom: &SExp) -> bool {
    match atom.to_bigint() {
        Some(n) => n == op.to_bigint().unwrap(),
        None => false,
    }
}

pub fn is_whole_env(atom: &SExp) -> bool {
    is_operator(1, atom)
}
pub fn is_apply(atom: &SExp) -> bool {
    is_operator(2, atom)
}
pub fn is_cons(atom: &SExp) -> bool {
    is_operator(4, atom)
}

// Extracts the environment from a clvm program that contains one.
// The usual form of a program to analyze is:
// (2 main (4 env 1))
pub fn extract_program_and_env(program: Rc<SExp>) -> Option<(Rc<SExp>, Rc<SExp>)> {
    // Most programs have apply as a toplevel form.  If we don't then it's
    // a form we don't understand.
    match program.proper_list() {
        Some(lst) => {
            if lst.len() != 3 {
                return None;
            }

            match (is_apply(&lst[0]), &lst[1], lst[2].proper_list()) {
                (true, real_program, Some(cexp)) => {
                    if cexp.len() != 3 || !is_cons(&cexp[0]) || !is_whole_env(&cexp[2]) {
                        None
                    } else {
                        Some((Rc::new(real_program.clone()), Rc::new(cexp[1].clone())))
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

pub fn is_at_capture(head: Rc<SExp>, rest: Rc<SExp>) -> Option<(Vec<u8>, Rc<SExp>)> {
    rest.proper_list().and_then(|l| {
        if l.len() != 2 {
            return None;
        }
        if let (SExp::Atom(_, a), SExp::Atom(_, cap)) = (head.borrow(), &l[0]) {
            if a == &vec![b'@'] {
                return Some((cap.clone(), Rc::new(l[1].clone())));
            }
        }

        None
    })
}
