mod macros;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use clvm_rs::error::EvalErr;
use clvmr::allocator::Allocator;
use clvmr::NodePtr;

use crate::classic::clvm_tools::binutils::assemble;
use crate::classic::clvm_tools::clvmc::compile_clvm_text_maybe_opt;
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};

use crate::compiler::cldb::hex_to_modern_sexp;
use crate::compiler::clvm;
use crate::compiler::clvm::{convert_from_clvm_rs, sha256tree, truthy, NewStyleIntConversion};
use crate::compiler::compiler::{compile_from_compileform, compile_pre_forms};
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompilerOpts, CompilerOutput, ConstantKind, DefconstData,
    HelperForm, ImportLongName, IncludeDesc, IncludeProcessType, LongNameTranslation,
    ModuleImportSpec, NamespaceData, NamespaceRefData, QualifiedModuleInfo,
};
use crate::compiler::dialect::{detect_modern, AcceptedDialect, KNOWN_DIALECTS};
use crate::compiler::frontend::{compile_helperform, compile_namespace, compile_nsref, frontend};
use crate::compiler::optimize::get_optimizer;
use crate::compiler::preprocessor::macros::PreprocessorExtension;
use crate::compiler::rename::rename_args_helperform;
use crate::compiler::resolve::{find_helper_target, resolve_namespaces};
use crate::compiler::runtypes::RunFailure;
use crate::compiler::sexp::{
    decode_string, enlist, parse_sexp, Atom, NodeSel, SExp, SelectNode, ThisNode,
};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::CompileContextWrapper;
use crate::util::ErrInto;

/// Determines how an included file is used.
///
/// Basic means that the file contains helper forms to include in the program.
/// Processed means that some kind of processing is done and the result is a named
/// constant.
#[derive(Clone, Debug)]
enum IncludeType {
    /// Normal include in chialisp.  The code in the target file will join the
    /// program being compiled.
    Basic(IncludeDesc),
    /// The data in the file will be processed in some way and the result will
    /// live in a named constant.
    Processed(Box<IncludeDesc>, IncludeProcessType, Vec<u8>),
}

#[derive(Debug)]
struct ImportNameMap {
    name: Option<ImportLongName>,
}

#[derive(Debug)]
pub enum StoredMacro {
    Waiting(HelperForm),
    Compiled(Rc<SExp>),
}

pub struct Preprocessor {
    subcompile_opts: Rc<dyn CompilerOpts>,
    opts: Rc<dyn CompilerOpts>,
    ppext: Rc<PreprocessorExtension>,
    runner: Rc<dyn TRunProgram>,
    strict: bool,
    prelude_import: Rc<SExp>,
    prototype_program: Vec<HelperForm>,
    stored_macros: HashMap<ImportLongName, StoredMacro>,

    namespace_stack: Vec<ImportNameMap>,
    imported_modules: HashSet<ImportLongName>,
}

fn compose_defconst(loc: Srcloc, name: &[u8], sexp: Rc<SExp>) -> Rc<SExp> {
    Rc::new(enlist(
        loc.clone(),
        &[
            Rc::new(SExp::Atom(loc.clone(), b"defconst".to_vec())),
            Rc::new(SExp::Atom(loc.clone(), name.to_vec())),
            Rc::new(SExp::Cons(
                loc.clone(),
                Rc::new(SExp::Atom(loc, vec![1])),
                sexp,
            )),
        ],
    ))
}

fn make_defmac_name(name: &[u8]) -> Vec<u8> {
    let mut res = b"__chia__defmac__".to_vec();
    res.append(&mut name.to_vec());
    res
}

fn nilize(v: Rc<SExp>) -> Rc<SExp> {
    if let SExp::Cons(l, a, b) = v.borrow() {
        let a_conv = nilize(a.clone());
        let b_conv = nilize(b.clone());
        if Rc::as_ptr(&a_conv) == Rc::as_ptr(a) && Rc::as_ptr(&b_conv) == Rc::as_ptr(b) {
            v.clone()
        } else {
            Rc::new(SExp::Cons(l.clone(), a_conv, b_conv))
        }
    } else if !truthy(v.clone()) {
        Rc::new(SExp::Nil(v.loc()))
    } else {
        v
    }
}

struct IterateIncludeUnit {
    parsed: Vec<Rc<SExp>>,
    item: usize,
}

impl Iterator for IterateIncludeUnit {
    type Item = Rc<SExp>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.item >= self.parsed.len() {
            return None;
        }

        let current_item = self.item;
        self.item += 1;

        Some(self.parsed[current_item].clone())
    }
}

/// Import modules can be enclosed, but needn't be:
///
/// An import file that contains a nil or a list beginning with a nonempty cons is
/// an enclosed clinc (classic style), otherwise it's a list of helper forms.
fn get_module_iterator(parsed: &[Rc<SExp>]) -> IterateIncludeUnit {
    // An empty form is a modern include file with 0 helpers.
    if parsed.is_empty() {
        return IterateIncludeUnit {
            parsed: vec![],
            item: 0,
        };
    }

    // A single form is an enclosed clinc if it is headed by a cons form.
    if parsed.len() == 1 {
        if let Some(lst) = parsed[0].proper_list() {
            if lst.is_empty() {
                return IterateIncludeUnit {
                    parsed: vec![],
                    item: 0,
                };
            }

            if !matches!(lst[0].borrow(), SExp::Cons(_, _, _)) {
                return IterateIncludeUnit {
                    parsed: parsed.to_vec(),
                    item: 0,
                };
            }

            return IterateIncludeUnit {
                parsed: lst.iter().cloned().map(Rc::new).collect(),
                item: 0,
            };
        }
    }

    // More than one form: conventional, not enclosed.
    IterateIncludeUnit {
        parsed: parsed.to_vec(),
        item: 0,
    }
}

fn import_name_to_module_name(
    loc: Srcloc,
    from_module: Option<&ImportLongName>,
    name: &[u8],
) -> Result<ImportLongName, CompileErr> {
    let (relative, parsed) = ImportLongName::parse(name);
    if relative {
        if let Some(m) = from_module {
            return Ok(m.combine(&parsed));
        }

        return Err(CompileErr(
            loc,
            "Relative module name requested from a module entry file".to_string(),
        ));
    }

    Ok(parsed)
}

fn make_namespace_container(
    loc: &Srcloc,
    nl: &Srcloc,
    target: &ImportLongName,
    helpers: Vec<Rc<SExp>>,
) -> Result<Rc<SExp>, CompileErr> {
    let mut result_vec = vec![
        Rc::new(SExp::Atom(loc.clone(), b"namespace".to_vec())),
        Rc::new(SExp::Atom(
            nl.clone(),
            target.as_u8_vec(LongNameTranslation::Namespace),
        )),
    ];
    result_vec.extend(helpers);
    Ok(Rc::new(enlist(loc.clone(), &result_vec)))
}

fn make_namespace_ref(
    loc: &Srcloc,
    kw: &Srcloc,
    nl: &Srcloc,
    target: &ImportLongName,
    spec: &ModuleImportSpec,
) -> HelperForm {
    HelperForm::Defnsref(Box::new(NamespaceRefData {
        loc: loc.clone(),
        kw: kw.clone(),
        nl: nl.clone(),
        rendered_name: target.as_u8_vec(LongNameTranslation::Namespace),
        longname: target.clone(),
        specification: spec.clone(),
    }))
}

pub fn detect_chialisp_module(
    loc: Srcloc,
    pre_forms: &[Rc<SExp>],
) -> Result<Option<AcceptedDialect>, CompileErr> {
    let dialect = if let Some(d) = KNOWN_DIALECTS.get("*standard-cl-25*") {
        d.accepted.clone()
    } else {
        return Err(CompileErr(
            loc.clone(),
            "internal error: default module dialect not found while compiling!".to_string(),
        ));
    };

    if pre_forms.is_empty() {
        return Ok(None);
    } else {
        // Multiple forms in a source file are always module style.
        // All other forms of chialisp use a single form in a file.
        if pre_forms.len() > 1 {
            for p in pre_forms.iter() {
                if let Ok(NodeSel::Cons(_kl, NodeSel::Cons((_nl, name), _))) = NodeSel::Cons(
                    Atom::Here("include"),
                    NodeSel::Cons(Atom::Here(()), Atom::Here("")),
                )
                .select_nodes(p.clone())
                {
                    if let Some(use_dialect) = KNOWN_DIALECTS.get(&decode_string(&name)) {
                        return Ok(Some(use_dialect.accepted.clone()));
                    }
                }
            }
            return Ok(Some(dialect));
        }

        if let Some(lst) = pre_forms[0].proper_list() {
            if matches!(lst[0].borrow(), SExp::Cons(_, _, _)) {
                return Ok(Some(dialect));
            }
        }
    }

    Ok(None)
}

#[test]
pub fn test_detect_chialisp_module_classic() {
    let filename = "resources/tests/module/programs/classic.clsp";
    let content = "(mod (X) (* X 13))";
    let loc = Srcloc::start(filename);
    let parsed = parse_sexp(loc.clone(), content.bytes()).expect("should parse");
    assert!(detect_chialisp_module(loc, &parsed).unwrap().is_none());
}

#[derive(Debug)]
pub struct ToplevelMod {
    pub forms: Vec<Rc<SExp>>,
    pub stripped_args: Rc<SExp>,
    pub loc: Srcloc,
}

pub enum ToplevelModParseResult {
    Mod(ToplevelMod),
    Simple(Srcloc, Vec<Rc<SExp>>),
}

pub fn parse_toplevel_mod(
    opts: Rc<dyn CompilerOpts>,
    top_loc: Option<Srcloc>,
    pre_forms: &[Rc<SExp>],
) -> Result<ToplevelModParseResult, CompileErr> {
    if pre_forms.is_empty() {
        return Err(CompileErr(
            Srcloc::start(&opts.filename()),
            "empty source file not allowed".to_string(),
        ));
    }

    let overall = top_loc.unwrap_or_else(|| {
        pre_forms
            .iter()
            .fold(pre_forms[0].loc(), |sum, val| sum.ext(&val.loc()))
    });
    if let Some(x) = pre_forms[0].proper_list() {
        if x.is_empty() {
            return Ok(ToplevelModParseResult::Simple(overall, pre_forms.to_vec()));
        }

        if let SExp::Atom(_, mod_atom) = &x[0] {
            if pre_forms.len() > 1 {
                return Err(CompileErr(
                    pre_forms[0].loc(),
                    "one toplevel mod form allowed".to_string(),
                ));
            }

            if *mod_atom == b"mod" {
                let args = Rc::new(x[1].atomize());
                let skip_idx = 2;

                if x.len() < 3 {
                    return Err(CompileErr(x[0].loc(), "incomplete mod form".to_string()));
                }

                return Ok(ToplevelModParseResult::Mod(ToplevelMod {
                    forms: x
                        .iter()
                        .skip(skip_idx)
                        .map(|s| Rc::new(s.clone()))
                        .collect(),
                    stripped_args: args,
                    loc: overall,
                }));
            }
        }
    }

    Ok(ToplevelModParseResult::Simple(overall, pre_forms.to_vec()))
}

#[derive(Debug)]
pub struct PreprocessResult {
    pub forms: Vec<Rc<SExp>>,
    pub modules: bool,
}

impl Preprocessor {
    pub fn new(opts: Rc<dyn CompilerOpts>) -> Self {
        let runner = Rc::new(DefaultProgramRunner::new());
        let ppext = Rc::new(PreprocessorExtension::new());
        let opts_prims = ppext.enrich_prims(opts.clone());
        let srcloc = Srcloc::start(&opts.filename());
        Preprocessor {
            subcompile_opts: opts.clone(),
            opts: opts_prims,
            ppext,
            runner,
            strict: opts.dialect().strict,
            prelude_import: Rc::new(SExp::Cons(
                srcloc.clone(),
                Rc::new(SExp::atom_from_string(srcloc.clone(), "include")),
                Rc::new(SExp::Cons(
                    srcloc.clone(),
                    Rc::new(SExp::quoted_from_string(srcloc.clone(), "*macros*")),
                    Rc::new(SExp::Nil(srcloc.clone())),
                )),
            )),
            stored_macros: HashMap::default(),
            imported_modules: HashSet::new(),
            namespace_stack: Vec::new(),
            prototype_program: Vec::new(),
        }
    }

    /// Get the current module name.
    pub fn current_module_name(&self) -> Option<ImportLongName> {
        if self.namespace_stack.is_empty() {
            None
        } else {
            self.namespace_stack[self.namespace_stack.len() - 1]
                .name
                .clone()
        }
    }

    fn make_namespace_helper(&self, loc: &Srcloc, name: &ImportLongName) -> HelperForm {
        let helpers = if self.opts.stdenv() {
            vec![make_namespace_ref(
                loc,
                loc,
                loc,
                &ImportLongName::parse(b"std.prelude").1,
                &ModuleImportSpec::Hiding(loc.clone(), vec![]),
            )]
        } else {
            vec![]
        };

        HelperForm::Defnamespace(Box::new(NamespaceData {
            loc: loc.clone(),
            nl: loc.clone(),
            kw: loc.clone(),
            rendered_name: name.as_u8_vec(LongNameTranslation::Namespace),
            longname: name.clone(),
            helpers,
        }))
    }

    /// Compute the expected filename from a module name.  It must be absolute
    /// if we're in the root file of a program, otherwise it can be relative.
    /// The namespace_stack determines that.
    pub fn import_name_to_module_name(
        &self,
        loc: Srcloc,
        name: &[u8],
    ) -> Result<ImportLongName, CompileErr> {
        let reference_name = self.current_module_name();
        import_name_to_module_name(loc, reference_name.as_ref(), name)
    }

    /// Given a specification of an include file, ltoad up the forms inside it and
    /// return them (or an error if the file couldn't be read or wasn't a list).
    pub fn process_include(
        &mut self,
        includes: &mut Vec<IncludeDesc>,
        include: &IncludeDesc,
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        let filename_and_content = self
            .opts
            .read_new_file(self.opts.filename(), decode_string(&include.name))?;
        let content = filename_and_content.1;
        let start_of_file = Srcloc::start(&decode_string(&include.name));

        // Because we're also subsequently returning CompileErr later in the pipe,
        // this needs an explicit err map.
        let parsed: Vec<Rc<SExp>> = parse_sexp(start_of_file.clone(), content.iter().copied())
            .err_into()
            .and_then(|x| match x[0].proper_list() {
                None => Err(CompileErr(
                    start_of_file,
                    "Includes should contain a list of forms".to_string(),
                )),
                Some(v) => Ok(v.iter().map(|x| Rc::new(x.clone())).collect()),
            })?;

        if self.strict {
            let mut result = Vec::new();
            for p in get_module_iterator(&parsed) {
                let mut new_forms = self.process_pp_form(includes, p.clone())?;
                result.append(&mut new_forms);
            }

            Ok(result)
        } else {
            Ok(parsed)
        }
    }

    fn import_program(
        &mut self,
        includes: &mut Vec<IncludeDesc>,
        _import_name: &ImportLongName,
        filename: &str,
        content: &[u8],
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        let srcloc = Srcloc::start(filename);
        let mut allocator = Allocator::new();
        let mut symbol_table = HashMap::new();
        let program_text = decode_string(content);
        let runner = Rc::new(DefaultProgramRunner::new());
        let pre_forms = parse_sexp(srcloc.clone(), content.iter().copied())?;
        let (have_module, dialect, classic_parse) =
            if let Some(dialect) = detect_chialisp_module(Srcloc::start(filename), &pre_forms)? {
                (true, dialect, NodePtr::NIL)
            } else {
                let classic_parse = assemble(&mut allocator, &program_text).map_err(|_| {
                    CompileErr(
                        srcloc.clone(),
                        format!("Could not parse {filename} to determine dialect"),
                    )
                })?;

                (
                    false,
                    detect_modern(&mut allocator, classic_parse),
                    classic_parse,
                )
            };

        let make_constant = |name: &[u8], s: SExp| {
            HelperForm::Defconstant(DefconstData {
                loc: srcloc.clone(),
                kind: ConstantKind::Complex,
                name: name.to_vec(),
                kw: None,
                nl: srcloc.clone(),
                tabled: true,
                body: Rc::new(BodyForm::Quoted(s)),
            })
            .to_sexp()
        };

        if !have_module {
            let dialect = detect_modern(&mut allocator, classic_parse);
            if dialect.stepping.is_none() {
                // Classic compile.
                let newly_compiled = compile_clvm_text_maybe_opt(
                    &mut allocator,
                    self.subcompile_opts.optimize(),
                    self.subcompile_opts.clone(),
                    &mut symbol_table,
                    &program_text,
                    filename,
                    true,
                )
                .map_err(|e| CompileErr(srcloc.clone(), format!("Subcompile failed: {:?}", e)))?;
                let converted =
                    convert_from_clvm_rs(&mut allocator, srcloc.clone(), newly_compiled)?;
                let converted_borrowed: &SExp = converted.borrow();
                return Ok(vec![
                    make_constant(b"program", converted_borrowed.clone()),
                    make_constant(
                        b"program_hash",
                        SExp::QuotedString(srcloc.clone(), b'x', sha256tree(converted)),
                    ),
                ]);
            }
        }

        let opts = self
            .subcompile_opts
            .set_dialect(dialect)
            .set_filename(filename);
        let mut context_wrapper = CompileContextWrapper::new(
            &mut allocator,
            runner,
            &mut symbol_table,
            get_optimizer(&srcloc, opts.clone())?,
        );

        match compile_pre_forms(&mut context_wrapper.context, opts.clone(), &pre_forms)? {
            CompilerOutput::Module(module_output) => {
                let mut output = Vec::new();
                for c in module_output.components.iter() {
                    let borrowed_content: &SExp = c.content.borrow();
                    output.push(make_constant(&c.shortname, borrowed_content.clone()));
                    let mut hash_name = c.shortname.clone();
                    hash_name.extend(b"_hash".to_vec());
                    output.push(make_constant(
                        &hash_name,
                        SExp::QuotedString(srcloc.clone(), b'x', c.hash.clone()),
                    ));
                }
                let mut module_output_includes = module_output.includes.clone();
                drop(context_wrapper);
                includes.append(&mut module_output_includes);
                Ok(output)
            }
            CompilerOutput::Program(_, compile_output) => Ok(vec![
                make_constant(b"program", compile_output.clone()),
                make_constant(
                    b"program_hash",
                    SExp::QuotedString(srcloc.clone(), b'x', sha256tree(Rc::new(compile_output))),
                ),
            ]),
        }
    }

    fn import_new_module_content(
        &mut self,
        loc: Srcloc,
        includes: &mut Vec<IncludeDesc>,
        import_name: &ImportLongName,
        parsed: &[SExp],
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        let mut out_forms = vec![];

        if self.opts.stdenv() {
            out_forms.push(
                make_namespace_ref(
                    &loc,
                    &loc,
                    &loc,
                    &ImportLongName::parse(b"std.prelude").1,
                    &ModuleImportSpec::Hiding(loc.clone(), vec![]),
                )
                .to_sexp(),
            );
        }

        self.namespace_stack.push(ImportNameMap {
            name: Some(import_name.clone()),
        });

        for p in parsed.iter() {
            for form in self.process_pp_form(includes, Rc::new(p.clone()))? {
                out_forms.push(form.clone());
            }
        }

        self.namespace_stack.pop();

        Ok(out_forms)
    }

    fn import_new_module(
        &mut self,
        loc: Srcloc,
        kw: &Srcloc,
        nl: &Srcloc,
        includes: &mut Vec<IncludeDesc>,
        import_name: &ImportLongName,
        kind: &ModuleImportSpec,
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        let filename_clsp = decode_string(
            &import_name.as_u8_vec(LongNameTranslation::Filename(".clsp".to_string())),
        );
        let filename_clinc = decode_string(
            &import_name.as_u8_vec(LongNameTranslation::Filename(".clinc".to_string())),
        );

        if let Ok((full_name, content)) =
            self.opts.read_new_file(self.opts.filename(), filename_clsp)
        {
            includes.push(IncludeDesc {
                kw: kw.clone(),
                nl: nl.clone(),
                name: full_name.as_bytes().to_vec(),
                kind: Some(IncludeProcessType::Module(Box::new(kind.clone()))),
            });
            return self.import_program(includes, import_name, &full_name, &content);
        }

        let (full_name, content) = self
            .opts
            .read_new_file(self.opts.filename(), filename_clinc)?;

        let parsed: Vec<SExp> = parse_sexp(Srcloc::start(&full_name), content.iter().copied())
            .map_err(|e| CompileErr(e.0, e.1))?
            .iter()
            .map(|r| {
                let re: &SExp = r.borrow();
                re.clone()
            })
            .collect();

        let res = self.import_new_module_content(loc, includes, import_name, &parsed)?;

        includes.push(IncludeDesc {
            kw: kw.clone(),
            nl: nl.clone(),
            name: full_name.as_bytes().to_vec(),
            kind: Some(IncludeProcessType::Module(Box::new(kind.clone()))),
        });

        Ok(res)
    }

    fn import_module(
        &mut self,
        loc: Srcloc,
        kw: Srcloc,
        nl: Srcloc,
        includes: &mut Vec<IncludeDesc>,
        spec: &ModuleImportSpec,
        import_name: &[u8],
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        // The name of a module needs more processing.
        let full_import_name = self.import_name_to_module_name(loc.clone(), import_name)?;
        let ns_helper = make_namespace_ref(&loc, &kw, &nl, &full_import_name, spec);

        if self.imported_modules.contains(&full_import_name) {
            // We've processed this already, generate namespace directives.
            self.add_helper(ns_helper.clone());
            return Ok(vec![ns_helper.to_sexp()]);
        }

        // Add an empty namespace to be added to while working.
        let empty_ns = self.make_namespace_helper(&loc, &full_import_name);
        self.prototype_program.push(empty_ns);

        // Process this module.
        let imported_content =
            self.import_new_module(loc.clone(), &kw, &nl, includes, &full_import_name, spec)?;
        let helper_forms: Vec<Rc<SExp>> = vec![
            make_namespace_container(&loc, &nl, &full_import_name, imported_content)?,
            ns_helper.to_sexp(),
        ];

        self.imported_modules.insert(full_import_name.clone());
        self.add_helper(ns_helper.clone());

        Ok(helper_forms)
    }

    fn process_embed(
        &mut self,
        loc: Srcloc,
        fname: &str,
        kind: &IncludeProcessType,
        constant_name: &[u8],
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        let mut allocator = Allocator::new();
        let run_to_compile_err = |e| match e {
            RunFailure::RunExn(l, x) => CompileErr(
                l,
                format!("failed to convert compiled clvm to expression: throw ({x})"),
            ),
            RunFailure::RunErr(l, e) => CompileErr(
                l,
                format!("failed to convert compiled clvm to expression: {e}"),
            ),
        };

        let (full_name, content) = self
            .opts
            .read_new_file(self.opts.filename(), fname.to_string())?;

        let content = if let IncludeProcessType::Bin = &kind {
            if self.opts.dialect().int_fix {
                Rc::new(SExp::QuotedString(loc.clone(), b'x', content))
            } else {
                Rc::new(SExp::Atom(loc.clone(), content))
            }
        } else if let IncludeProcessType::Hex = &kind {
            hex_to_modern_sexp(
                &mut allocator,
                &HashMap::new(),
                loc.clone(),
                &decode_string(&content),
            )
            .map_err(run_to_compile_err)?
        } else if let IncludeProcessType::Compiled = &kind {
            let decoded_content = decode_string(&content);
            let mut symtab = HashMap::new();
            let newly_compiled = compile_clvm_text_maybe_opt(
                &mut allocator,
                self.subcompile_opts.optimize(),
                self.subcompile_opts.clone(),
                &mut symtab,
                &decoded_content,
                &full_name,
                true,
            )
            .map_err(|e| CompileErr(loc.clone(), format!("Subcompile failed: {:?}", e)))?;

            convert_from_clvm_rs(&mut allocator, loc.clone(), newly_compiled)
                .map_err(run_to_compile_err)?
        } else {
            // IncludeProcessType::SExpression
            let parsed = parse_sexp(Srcloc::start(&full_name), content.iter().copied())
                .map_err(|e| CompileErr(e.0, e.1))?;
            if parsed.len() != 1 {
                return Err(CompileErr(loc, format!("More than one form in {fname}")));
            }

            parsed[0].clone()
        };

        Ok(vec![compose_defconst(loc, constant_name, content)])
    }

    // Support using the preprocessor to collect dependencies recursively.
    fn recurse_dependencies(
        &mut self,
        includes: &mut Vec<IncludeDesc>,
        kind: Option<IncludeProcessType>,
        desc: IncludeDesc,
    ) -> Result<(), CompileErr> {
        // Process an import
        let name_string = if let Some(IncludeProcessType::Module(_)) = kind {
            decode_string(
                &self
                    .import_name_to_module_name(desc.nl.clone(), &desc.name)?
                    .as_u8_vec(LongNameTranslation::Filename(".clinc".to_string())),
            )
        } else {
            decode_string(&desc.name)
        };

        if KNOWN_DIALECTS.contains_key(&name_string)
            && !matches!(desc.kind, Some(IncludeProcessType::Compiled))
        {
            return Ok(());
        }

        let (full_name, content) = self.opts.read_new_file(self.opts.filename(), name_string)?;
        includes.push(IncludeDesc {
            name: full_name.as_bytes().to_vec(),
            ..desc
        });

        // Only the None type (textural include) and Compiled type should gather includes.
        // Module includes are separate (and separately tracked) and the embedding types
        // should not be recursed (which had been an error at one point).
        if !matches!(kind, None | Some(IncludeProcessType::Compiled)) {
            return Ok(());
        }

        let parsed = parse_sexp(Srcloc::start(&full_name), content.iter().copied())
            .map_err(|e| CompileErr(e.0, e.1))?;
        if parsed.is_empty() {
            return Ok(());
        }

        let program_form = parsed[0].clone();
        if let Some(l) = program_form.proper_list() {
            for elt in l.iter() {
                self.process_pp_form(includes, Rc::new(elt.clone()))?;
            }
        }

        Ok(())
    }

    fn get_current_namespace(&self) -> Option<ImportLongName> {
        if self.namespace_stack.is_empty() {
            return None;
        }

        self.namespace_stack[self.namespace_stack.len() - 1]
            .name
            .clone()
    }

    fn add_helper(&mut self, h: HelperForm) {
        let namespace = self.get_current_namespace();
        if let Some(n) = &namespace {
            for existing_helper in self.prototype_program.iter_mut() {
                if let HelperForm::Defnamespace(ns) = existing_helper {
                    if ns.longname == *n {
                        ns.helpers.push(h);
                        break;
                    }
                }
            }
        } else {
            self.prototype_program.push(h);
        }
    }

    fn add_macro(&mut self, h: &HelperForm) {
        let current_module_name = self.current_module_name();
        let (_, helper_name) = ImportLongName::parse(h.name());
        let full_name = if let Some(module_name) = &current_module_name {
            module_name.with_child(h.name())
        } else {
            helper_name
        };
        self.stored_macros
            .insert(full_name, StoredMacro::Waiting(h.clone()));
        self.add_helper(h.clone());
    }

    fn find_macro(&mut self, loc: Srcloc, name: &[u8]) -> Result<Option<Rc<SExp>>, CompileErr> {
        let mut allocator = Allocator::new();
        let current_module_name = self.current_module_name();
        let (_, parsed_name) = ImportLongName::parse(name);
        let (parent_name, clean_last_name_component) = parsed_name.parent_and_name();

        let last_name_component = make_defmac_name(&clean_last_name_component);
        let updated_name = if let Some(parent) = &parent_name {
            parent.with_child(&last_name_component)
        } else {
            let (_, parsed_name) = ImportLongName::parse(&last_name_component);
            parsed_name
        };

        let current_module_name_ref = current_module_name.as_ref();
        let found_name = if let Some((tname, _helper)) = find_helper_target(
            self.opts.clone(),
            &self.prototype_program,
            current_module_name_ref,
            &clean_last_name_component,
            &updated_name,
        )? {
            if let Some(mac) = self.stored_macros.get_mut(&tname) {
                match mac {
                    StoredMacro::Compiled(use_macro) => {
                        return Ok(Some(use_macro.clone()));
                    }
                    StoredMacro::Waiting(_h) => tname,
                }
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        // as inline defuns because they're closest to that semantically.
        let optimizer = get_optimizer(&loc, self.opts.clone())?;
        let mut symbol_table = HashMap::new();
        let mut wrapper = CompileContextWrapper::new(
            &mut allocator,
            self.runner.clone(),
            &mut symbol_table,
            optimizer,
        );

        let mut main_helpers: Vec<HelperForm> = self.prototype_program.clone();

        if let Some(parent) = found_name.parent() {
            main_helpers.push(HelperForm::Defnsref(Box::new(NamespaceRefData {
                loc: loc.clone(),
                kw: loc.clone(),
                nl: loc.clone(),
                rendered_name: parent.as_u8_vec(LongNameTranslation::Namespace),
                longname: parent.clone(),
                specification: ModuleImportSpec::Qualified(Box::new(QualifiedModuleInfo {
                    loc: loc.clone(),
                    nl: loc.clone(),
                    kw: loc.clone(),
                    name: parent.clone(),
                    target: None,
                })),
            })));
        }

        let starting_program = CompileForm {
            loc: loc.clone(),
            args: Rc::new(SExp::Atom(loc.clone(), b"__chia__arg".to_vec())),
            include_forms: vec![],
            helpers: main_helpers,
            exp: Rc::new(BodyForm::Call(
                loc.clone(),
                vec![Rc::new(BodyForm::Value(SExp::Atom(
                    loc.clone(),
                    found_name.as_u8_vec(LongNameTranslation::Namespace),
                )))],
                Some(Rc::new(BodyForm::Value(SExp::Atom(
                    loc.clone(),
                    b"__chia__arg".to_vec(),
                )))),
            )),
        };

        let new_program = resolve_namespaces(self.opts.clone(), &starting_program)?;

        let compiled_program =
            compile_from_compileform(&mut wrapper.context, self.opts.clone(), new_program)?;
        self.stored_macros.insert(
            found_name.clone(),
            StoredMacro::Compiled(Rc::new(compiled_program.clone())),
        );
        Ok(Some(Rc::new(compiled_program)))
    }

    // Check for and apply preprocessor level macros.
    // This is maximally permissive.
    fn expand_macros(&mut self, body: Rc<SExp>) -> Result<Option<Rc<SExp>>, CompileErr> {
        if let SExp::Cons(_, _, _) = body.borrow() {
            // First expand inner macros.
            let mut rest_list = Vec::new();
            let mut rest = body.clone();
            while let SExp::Cons(l, head, next_rest) = rest.borrow() {
                rest_list.push((l.clone(), head.clone(), next_rest.clone()));
                rest = next_rest.clone();
            }

            let mut rest_expanded = self.expand_macros(rest.clone())?;
            let mut any_expanded = rest_expanded.is_some();
            for (l, head, tail) in rest_list.into_iter().rev() {
                let first_expanded = self.expand_macros(head.clone())?;
                any_expanded = any_expanded || first_expanded.is_some();
                if any_expanded {
                    let new_first = first_expanded.unwrap_or_else(|| head.clone());
                    let new_rest = rest_expanded.unwrap_or_else(|| tail.clone());
                    rest_expanded = Some(Rc::new(SExp::Cons(l.clone(), new_first, new_rest)));
                }
            }

            let new_self = rest_expanded;

            if let Ok(NodeSel::Cons((_, name), args)) = NodeSel::Cons(Atom::Here(()), ThisNode)
                .select_nodes(new_self.clone().unwrap_or_else(|| body.clone()))
            {
                // See if it's a form that calls one of our macros.
                if let Some(compiled_program) = self.find_macro(body.loc(), &name)? {
                    // Form argument env.
                    let mut allocator = Allocator::new();

                    let ppext: &PreprocessorExtension = self.ppext.borrow();
                    let res = clvm::run(
                        &mut allocator,
                        self.runner.clone(),
                        self.opts.prim_map(),
                        compiled_program,
                        args.clone(),
                        Some(ppext),
                        None,
                    )
                    .map(nilize)
                    .map_err(CompileErr::from)?;

                    if let Some(final_result) = self.expand_macros(res.clone())? {
                        return Ok(Some(final_result));
                    } else {
                        return Ok(Some(res));
                    }
                }
            }

            return Ok(new_self);
        }

        Ok(None)
    }

    // If it's a defmac (preprocessor level macro), add it to the evaulator.
    fn decode_macro(&mut self, definition: Rc<SExp>) -> Result<Option<()>, CompileErr> {
        if let Ok(NodeSel::Cons(
            (defmac_loc, kw),
            NodeSel::Cons((nl, name), NodeSel::Cons(args, body)),
        )) = NodeSel::Cons(
            Atom::Here(()),
            NodeSel::Cons(Atom::Here(()), NodeSel::Cons(ThisNode, ThisNode)),
        )
        .select_nodes(definition.clone())
        {
            let is_defmac = kw == b"defmac";
            if is_defmac
                || kw == b"defmacro"
                || kw == b"defun"
                || kw == b"defun-inline"
                || kw == b"defconst"
                || kw == b"defconstant"
                || kw == b"defalias"
            {
                if is_defmac {
                    let target_defun = Rc::new(SExp::Cons(
                        defmac_loc.clone(),
                        Rc::new(SExp::atom_from_string(defmac_loc, "defun")),
                        Rc::new(SExp::Cons(
                            nl.clone(),
                            Rc::new(SExp::Atom(nl, make_defmac_name(&name))),
                            Rc::new(SExp::Cons(args.loc(), args.clone(), body)),
                        )),
                    ));
                    if let Some(helpers) = compile_helperform(self.opts.clone(), target_defun)? {
                        for h in helpers.new_helpers.iter() {
                            let renamed = rename_args_helperform(h)?;
                            self.add_macro(&renamed);
                        }
                    } else {
                        return Err(CompileErr(
                            definition.loc(),
                            "defmac found but couldn't be converted to function".to_string(),
                        ));
                    }
                } else if let Some(helpers) = compile_helperform(self.opts.clone(), definition)? {
                    for h in helpers.new_helpers.iter() {
                        let renamed_helper = rename_args_helperform(h)?;
                        self.add_helper(renamed_helper);
                    }
                }
            }
        }

        Ok(None)
    }

    fn parse_import(
        &mut self,
        loc: Srcloc,
        form: &[SExp],
    ) -> Result<Option<IncludeType>, CompileErr> {
        if form.is_empty() {
            return Ok(None);
        }

        let name = if let SExp::Atom(_, name) = &form[0] {
            name
        } else {
            return Ok(None);
        };

        if name != b"import" {
            return Ok(None);
        }

        let import = if let HelperForm::Defnsref(import) = compile_nsref(loc.clone(), form)? {
            import
        } else {
            return Err(CompileErr(loc, "asked to parse import".to_string()));
        };

        let mod_kind = IncludeProcessType::Module(Box::new(import.specification.clone()));
        let fname = import.longname.as_u8_vec(LongNameTranslation::Namespace);

        Ok(Some(IncludeType::Processed(
            Box::new(IncludeDesc {
                kw: import.kw.clone(),
                nl: import.nl.clone(),
                name: fname.clone(),
                kind: Some(mod_kind.clone()),
            }),
            mod_kind,
            fname.clone(),
        )))
    }

    fn parse_namespace(
        &mut self,
        loc: Srcloc,
        includes: &mut Vec<IncludeDesc>,
        form: &[SExp],
    ) -> Result<Option<()>, CompileErr> {
        if form.is_empty() {
            return Ok(None);
        }

        let name = if let SExp::Atom(_, name) = &form[0] {
            name
        } else {
            return Ok(None);
        };

        if name != b"namespace" {
            return Ok(None);
        }

        let ns_helper = compile_namespace(self.opts.clone(), loc.clone(), form)?;
        if let HelperForm::Defnamespace(namespace) = &ns_helper {
            self.imported_modules.insert(namespace.longname.clone());
            self.import_new_module_content(
                ns_helper.loc(),
                includes,
                &namespace.longname,
                &form[2..],
            )?;
            return Ok(Some(()));
        }

        Err(CompileErr(loc, "asked to parse namespace".to_string()))
    }

    fn parse_include(&mut self, form: &[SExp]) -> Result<Option<IncludeType>, CompileErr> {
        if let [SExp::Atom(kw, include_kw), include_name] = form {
            if include_kw != b"include" {
                return Ok(None);
            }

            let (nl, fname) = if let SExp::Atom(nl, fname) = include_name {
                (nl.clone(), fname.clone())
            } else if let SExp::QuotedString(nl, _, fname) = include_name {
                (nl.clone(), fname.clone())
            } else {
                return Err(CompileErr(
                    include_name.loc(),
                    "Name must be an atom or string".to_string(),
                ));
            };

            return Ok(Some(IncludeType::Basic(IncludeDesc {
                kw: kw.clone(),
                nl: nl.clone(),
                name: fname.clone(),
                kind: None,
            })));
        }

        Ok(None)
    }

    fn parse_embed(
        &mut self,
        body: Rc<SExp>,
        form: &[SExp],
    ) -> Result<Option<IncludeType>, CompileErr> {
        if let [SExp::Atom(kl, embed_file), SExp::Atom(_, name), SExp::Atom(kind_loc, kind), fname_atom] =
            form
        {
            if embed_file != b"embed-file" {
                return Ok(None);
            }

            let (nl, fname) = if let SExp::Atom(nl, fname) = fname_atom {
                (nl.clone(), fname.clone())
            } else if let SExp::QuotedString(nl, _, fname) = fname_atom {
                (nl.clone(), fname.clone())
            } else {
                return Err(CompileErr(
                    fname_atom.loc(),
                    "Name must be an atom or string".to_string(),
                ));
            };

            if kind == b"hex" {
                return Ok(Some(IncludeType::Processed(
                    Box::new(IncludeDesc {
                        kw: kl.clone(),
                        nl: nl.clone(),
                        kind: Some(IncludeProcessType::Hex),
                        name: fname.clone(),
                    }),
                    IncludeProcessType::Hex,
                    name.clone(),
                )));
            } else if kind == b"bin" {
                return Ok(Some(IncludeType::Processed(
                    Box::new(IncludeDesc {
                        kw: kl.clone(),
                        nl: nl.clone(),
                        kind: Some(IncludeProcessType::Bin),
                        name: fname.clone(),
                    }),
                    IncludeProcessType::Bin,
                    name.clone(),
                )));
            } else if kind == b"sexp" {
                return Ok(Some(IncludeType::Processed(
                    Box::new(IncludeDesc {
                        kw: kl.clone(),
                        nl: nl.clone(),
                        kind: Some(IncludeProcessType::SExpression),
                        name: fname.clone(),
                    }),
                    IncludeProcessType::SExpression,
                    name.clone(),
                )));
            } else {
                return Err(CompileErr(
                    kind_loc.clone(),
                    format!("bad include kind in embed-file {body}"),
                ));
            }
        } else if let [SExp::Atom(kl, compile_file), SExp::Atom(_, name), fname_atom] = form {
            if compile_file != b"compile-file" {
                return Ok(None);
            }

            let (nl, fname) = if let SExp::Atom(nl, fname) = fname_atom {
                (nl.clone(), fname.clone())
            } else if let SExp::QuotedString(nl, _, fname) = fname_atom {
                (nl.clone(), fname.clone())
            } else {
                return Err(CompileErr(
                    fname_atom.loc(),
                    "Name must be an atom or string".to_string(),
                ));
            };

            return Ok(Some(IncludeType::Processed(
                Box::new(IncludeDesc {
                    kw: kl.clone(),
                    nl: nl.clone(),
                    kind: Some(IncludeProcessType::Compiled),
                    name: fname.clone(),
                }),
                IncludeProcessType::Compiled,
                name.clone(),
            )));
        }

        Ok(None)
    }

    /* Expand include inline in forms */
    fn process_pp_form(
        &mut self,
        includes: &mut Vec<IncludeDesc>,
        unexpanded_body: Rc<SExp>,
    ) -> Result<Vec<Rc<SExp>>, CompileErr> {
        let mut body = unexpanded_body.clone();

        loop {
            let new_body = self.expand_macros(body.clone())?;
            // Keep expanding until we run out of expansions.
            if let Some(new_body) = new_body {
                body = new_body;
            } else {
                break;
            }
        }

        // Support using the preprocessor to collect dependencies recursively.
        let as_list: Option<Vec<SExp>> = body
            .proper_list()
            .map(|x| x.iter().map(|elt| elt.atomize()).collect());
        let included: Option<IncludeType> = if let Some(x) = as_list.as_ref() {
            if let Some(res) = self.parse_import(unexpanded_body.loc(), x)? {
                Some(res)
            } else if self
                .parse_namespace(unexpanded_body.loc(), includes, x)?
                .is_some()
            {
                None
            } else if let Some(res) = self.parse_include(x)? {
                Some(res)
            } else if let Some(res) = self.parse_embed(body.clone(), x)? {
                Some(res)
            } else if !x.is_empty() {
                // Include is only allowed as a proper form.  It's a keyword in
                // this language.
                if let SExp::Atom(_, inc) = &x[0] {
                    if "include".as_bytes().to_vec() == *inc {
                        return Err(CompileErr(
                            body.loc(),
                            format!("bad tail in include {body}"),
                        ));
                    }
                }

                None
            } else {
                None
            }
        } else {
            None
        };

        if let Some(IncludeType::Basic(i)) = &included {
            self.recurse_dependencies(includes, None, i.clone())?;
            self.process_include(includes, i)
        } else if let Some(IncludeType::Processed(f, IncludeProcessType::Module(spec), name)) =
            &included
        {
            if self.namespace_stack.is_empty() {
                self.namespace_stack.push(ImportNameMap { name: None });
            }
            self.import_module(body.loc(), f.kw.clone(), f.nl.clone(), includes, spec, name)
        } else if let Some(IncludeType::Processed(f, kind, name)) = &included {
            self.recurse_dependencies(includes, Some(kind.clone()), *f.clone())?;
            self.process_embed(body.loc(), &decode_string(&f.name), kind, name)
        } else if let Some(()) = self.decode_macro(body.clone())? {
            Ok(vec![])
        } else {
            Ok(vec![body])
        }
    }

    pub fn run(
        &mut self,
        includes: &mut Vec<IncludeDesc>,
        cmod: &[Rc<SExp>],
    ) -> Result<PreprocessResult, CompileErr> {
        let mut result = Vec::new();

        if self.opts.stdenv() {
            let mut lst = self.process_pp_form(includes, self.prelude_import.clone())?;
            result.append(&mut lst);
        }
        for f in cmod.iter() {
            let mut lst = self.process_pp_form(includes, f.clone())?;
            result.append(&mut lst);
        }

        Ok(PreprocessResult {
            forms: result,
            modules: false,
        })
    }

    pub fn run_modules(
        &mut self,
        includes: &mut Vec<IncludeDesc>,
        cmod: &[Rc<SExp>],
    ) -> Result<PreprocessResult, CompileErr> {
        // Always active since this launched afterthe fix.
        let _int_conversion_bug = NewStyleIntConversion::new(true);
        if !cmod.is_empty() {
            self.prelude_import = Rc::new(SExp::Cons(
                cmod[0].loc(),
                Rc::new(SExp::atom_from_string(cmod[0].loc(), "import")),
                Rc::new(SExp::Cons(
                    cmod[0].loc(),
                    Rc::new(SExp::atom_from_string(cmod[0].loc(), "std.prelude")),
                    Rc::new(SExp::Nil(cmod[0].loc())),
                )),
            ));
        }
        let m = self.run(includes, cmod)?;
        Ok(PreprocessResult { modules: true, ..m })
    }
}

/// Run the preprocessor over this code, which at present just finds (include ...)
/// forms in the source and includes the content of in a combined list.  If a file
/// can't be found via the directory list in CompilerOrs.
pub fn preprocess(
    opts: Rc<dyn CompilerOpts>,
    includes: &mut Vec<IncludeDesc>,
    cmod: &[Rc<SExp>],
) -> Result<PreprocessResult, CompileErr> {
    let mut p = Preprocessor::new(opts);
    p.run(includes, cmod)
}

/// Visit all files used during compilation.
/// This reports a list of all files used while compiling the input file, via any
/// form that causes compilation to include another file.  The file names are path
/// expanded based on the include path they were found in (from opts).
pub fn gather_dependencies(
    mut opts: Rc<dyn CompilerOpts>,
    real_input_path: &str,
    file_content: &str,
) -> Result<Vec<IncludeDesc>, CompileErr> {
    let mut allocator = Allocator::new();

    let assembled_input = assemble(&mut allocator, file_content).map_err(|e| {
        CompileErr(
            Srcloc::start(real_input_path),
            match e {
                EvalErr::InternalError(_, e) => e.to_string(),
                _ => e.to_string(),
            },
        )
    })?;
    let dialect = detect_modern(&mut allocator, assembled_input);
    opts = opts.set_stdenv(dialect.strict).set_dialect(dialect.clone());
    if let Some(stepping) = dialect.stepping {
        opts = opts
            .set_optimize(stepping > 22)
            .set_frontend_opt(stepping > 21);
    }

    let parsed = parse_sexp(Srcloc::start(real_input_path), file_content.bytes())?;
    let program = frontend(opts, &parsed)?;

    let filtered_results: Vec<IncludeDesc> = program
        .compileform()
        .clone()
        .include_forms
        .into_iter()
        .filter(|f| !f.name.starts_with(b"*"))
        .collect();
    Ok(filtered_results)
}
