use rand::Rng;
use std::borrow::Borrow;
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm_tools::binutils::assemble;
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use crate::compiler::clvm::convert_from_clvm_rs;
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{CompilerOpts, CompilerOutput};
use crate::compiler::fuzz::{FuzzGenerator, FuzzTypeParams, Rule};
use crate::compiler::sexp::{decode_string, enlist, parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::tests::compiler::fuzz::{compose_sexp, simple_run, simple_seeded_rng};
use crate::tests::compiler::modules::{
    hex_to_clvm, perform_compile_of_file, TestModuleCompilerOpts,
};

const VARIATIONS_PER_PROGRAM: u32 = 1000;
const MAX_EXPANSIONS_BEFORE_TERMINATING: usize = 18;
const MAX_EXPANSIONS_TOTAL: usize = 120;

const ASSIGN_PROGRAM: &str = include_str!("../../../resources/tests/fuzz_test_assign_bug_1.clsp");
const ASSIGN_ARGS_PROGRAM: &str =
    include_str!("../../../resources/tests/fuzz_test_assign_bug_1_args.clsp");
const ASSIGN_CLASSIC_PROGRAM: &str =
    include_str!("../../../resources/tests/fuzz_test_assign_bug_1_classic.clsp");
const RECURSE_PROGRAM: &str = include_str!("../../../resources/tests/fuzz_test_recurse_bug_0.clsp");
const RECURSE_CLASSIC_PROGRAM: &str =
    include_str!("../../../resources/tests/fuzz_test_recurse_bug_0_classic.clsp");

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExprKind {
    Scalar,
    List,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Template {
    kind: ExprKind,
    original: Rc<SExp>,
    depth: usize,
    allow_literal: bool,
}

#[derive(Clone, Debug)]
struct ResourceProgramState {
    loc: Srcloc,
    templates: Vec<Template>,
}

impl ResourceProgramState {
    fn add_template(
        &mut self,
        kind: ExprKind,
        original: Rc<SExp>,
        depth: usize,
        allow_literal: bool,
    ) -> usize {
        let template_id = self.templates.len();
        self.templates.push(Template {
            kind,
            original,
            depth,
            allow_literal,
        });
        template_id
    }
}

struct ResourceProgramFuzz;

impl FuzzTypeParams for ResourceProgramFuzz {
    type Tag = Vec<u8>;
    type Expr = Rc<SExp>;
    type Error = String;
    type State = ResourceProgramState;
}

#[derive(Clone)]
struct TargetSpec {
    source: &'static str,
    kind: ExprKind,
    depth: usize,
    allow_literal: bool,
}

#[derive(Clone)]
struct LoadedProgram {
    topnode: Rc<SExp>,
    initial_templates: Vec<Template>,
}

fn atom(loc: &Srcloc, name: &str) -> Rc<SExp> {
    Rc::new(SExp::Atom(loc.clone(), name.as_bytes().to_vec()))
}

fn integer(loc: &Srcloc, value: i64) -> Rc<SExp> {
    Rc::new(SExp::Integer(loc.clone(), value.into()))
}

fn nil(loc: &Srcloc) -> Rc<SExp> {
    Rc::new(SExp::Nil(loc.clone()))
}

fn list(loc: &Srcloc, items: &[Rc<SExp>]) -> Rc<SExp> {
    Rc::new(enlist(loc.clone(), items))
}

fn placeholder(loc: &Srcloc, idx: usize, template_id: usize) -> Rc<SExp> {
    atom(loc, &format!("${{{idx}:template|{template_id}}}"))
}

fn decode_template_id(tag: &[u8]) -> Option<usize> {
    let tag = std::str::from_utf8(tag).ok()?;
    let template_id = tag.strip_prefix("template|")?;
    template_id.parse().ok()
}

fn as_atom(expr: &Rc<SExp>) -> Option<Vec<u8>> {
    match expr.borrow() {
        SExp::Atom(_, name) => Some(name.clone()),
        _ => None,
    }
}

fn list_elements(expr: &Rc<SExp>) -> Option<Vec<Rc<SExp>>> {
    let mut result = Vec::new();
    let mut cursor = expr.clone();
    loop {
        match cursor.borrow() {
            SExp::Nil(_) => return Some(result),
            SExp::Cons(_, left, right) => {
                result.push(left.clone());
                cursor = right.clone();
            }
            _ => return None,
        }
    }
}

fn template_placeholder(
    state: &mut ResourceProgramState,
    idx: &mut usize,
    kind: ExprKind,
    original: Rc<SExp>,
    depth: usize,
    allow_literal: bool,
) -> Rc<SExp> {
    let template_id = state.add_template(kind, original, depth, allow_literal);
    let placeholder = placeholder(&state.loc, *idx, template_id);
    *idx += 1;
    placeholder
}

fn child_template(
    state: &mut ResourceProgramState,
    idx: &mut usize,
    parent: &Template,
    kind: ExprKind,
    original: Rc<SExp>,
) -> Rc<SExp> {
    template_placeholder(
        state,
        idx,
        kind,
        original,
        parent.depth.saturating_sub(1),
        parent.allow_literal,
    )
}

fn skeletonize_original(
    state: &mut ResourceProgramState,
    idx: usize,
    template: &Template,
) -> Rc<SExp> {
    let mut next_idx = idx;
    skeletonize_with_kind(
        state,
        &mut next_idx,
        template,
        template.kind.clone(),
        template.original.clone(),
    )
}

fn skeletonize_with_kind(
    state: &mut ResourceProgramState,
    idx: &mut usize,
    parent: &Template,
    kind: ExprKind,
    expr: Rc<SExp>,
) -> Rc<SExp> {
    if parent.depth == 0 {
        return expr;
    }

    let Some(items) = list_elements(&expr) else {
        return expr;
    };
    let Some(op) = items.first().and_then(as_atom) else {
        return expr;
    };
    let op = decode_string(&op);
    let loc = &state.loc.clone();

    match op.as_str() {
        "if" | "i" if items.len() == 4 => list(
            loc,
            &[
                items[0].clone(),
                child_template(state, idx, parent, ExprKind::Scalar, items[1].clone()),
                child_template(state, idx, parent, kind.clone(), items[2].clone()),
                child_template(state, idx, parent, kind, items[3].clone()),
            ],
        ),
        "f" if items.len() == 2 => list(
            loc,
            &[
                items[0].clone(),
                child_template(state, idx, parent, ExprKind::List, items[1].clone()),
            ],
        ),
        "r" if items.len() == 2 => list(
            loc,
            &[
                items[0].clone(),
                child_template(state, idx, parent, ExprKind::List, items[1].clone()),
            ],
        ),
        "c" if items.len() == 3 => list(
            loc,
            &[
                items[0].clone(),
                child_template(state, idx, parent, ExprKind::Scalar, items[1].clone()),
                child_template(state, idx, parent, ExprKind::List, items[2].clone()),
            ],
        ),
        "nth" | "walk" if items.len() == 3 => list(
            loc,
            &[
                items[0].clone(),
                child_template(state, idx, parent, ExprKind::List, items[1].clone()),
                child_template(state, idx, parent, ExprKind::Scalar, items[2].clone()),
            ],
        ),
        "strlen" if items.len() == 2 => list(
            loc,
            &[
                items[0].clone(),
                child_template(state, idx, parent, ExprKind::Scalar, items[1].clone()),
            ],
        ),
        "+" | "-" | "*" | "=" | ">" | "<" | "all" | "any" => {
            let mut replacement = Vec::with_capacity(items.len());
            replacement.push(items[0].clone());
            for item in items.iter().skip(1) {
                replacement.push(child_template(
                    state,
                    idx,
                    parent,
                    ExprKind::Scalar,
                    item.clone(),
                ));
            }
            list(loc, &replacement)
        }
        _ => expr,
    }
}

struct OriginalTemplateRule;

impl Rule<ResourceProgramFuzz> for OriginalTemplateRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?
            .clone();

        if terminate || template.depth == 0 {
            return Ok(Some(template.original));
        }

        Ok(Some(skeletonize_original(state, idx, &template)))
    }
}

struct ScalarLiteralRule;

impl Rule<ResourceProgramFuzz> for ScalarLiteralRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?;
        if terminate || template.kind != ExprKind::Scalar || !template.allow_literal {
            return Ok(None);
        }

        Ok(Some(integer(&state.loc, (idx % 7) as i64)))
    }
}

struct ScalarPreservingWrapperRule;

impl Rule<ResourceProgramFuzz> for ScalarPreservingWrapperRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?;
        if terminate || template.kind != ExprKind::Scalar {
            return Ok(None);
        }

        Ok(Some(list(
            &state.loc,
            &[
                atom(&state.loc, "f"),
                list(
                    &state.loc,
                    &[
                        atom(&state.loc, "c"),
                        template.original.clone(),
                        nil(&state.loc),
                    ],
                ),
            ],
        )))
    }
}

struct ScalarAddZeroRule;

impl Rule<ResourceProgramFuzz> for ScalarAddZeroRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?
            .clone();
        if terminate || template.kind != ExprKind::Scalar || template.depth == 0 {
            return Ok(None);
        }

        let mut next_idx = idx + 1;
        let child = template_placeholder(
            state,
            &mut next_idx,
            ExprKind::Scalar,
            template.original,
            template.depth - 1,
            template.allow_literal,
        );
        Ok(Some(list(
            &state.loc,
            &[atom(&state.loc, "+"), child, integer(&state.loc, 0)],
        )))
    }
}

struct IfWrapperRule;

impl Rule<ResourceProgramFuzz> for IfWrapperRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?;
        if terminate {
            return Ok(None);
        }

        Ok(Some(list(
            &state.loc,
            &[
                atom(&state.loc, "if"),
                integer(&state.loc, 1),
                template.original.clone(),
                template.original.clone(),
            ],
        )))
    }
}

struct ListRestConsRule;

impl Rule<ResourceProgramFuzz> for ListRestConsRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?
            .clone();
        if terminate || template.kind != ExprKind::List || template.depth == 0 {
            return Ok(None);
        }

        let mut next_idx = idx + 1;
        let child = template_placeholder(
            state,
            &mut next_idx,
            ExprKind::List,
            template.original,
            template.depth - 1,
            template.allow_literal,
        );
        Ok(Some(list(
            &state.loc,
            &[
                atom(&state.loc, "r"),
                list(
                    &state.loc,
                    &[atom(&state.loc, "c"), integer(&state.loc, 0), child],
                ),
            ],
        )))
    }
}

struct ListRebuildRule;

impl Rule<ResourceProgramFuzz> for ListRebuildRule {
    fn check(
        &self,
        state: &mut ResourceProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        _parents: &[Rc<SExp>],
    ) -> Result<Option<Rc<SExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?;
        if terminate || template.kind != ExprKind::List {
            return Ok(None);
        }

        Ok(Some(list(
            &state.loc,
            &[
                atom(&state.loc, "c"),
                list(
                    &state.loc,
                    &[atom(&state.loc, "f"), template.original.clone()],
                ),
                list(
                    &state.loc,
                    &[atom(&state.loc, "r"), template.original.clone()],
                ),
            ],
        )))
    }
}

fn fuzz_rules() -> Vec<Rc<dyn Rule<ResourceProgramFuzz>>> {
    vec![
        Rc::new(OriginalTemplateRule),
        Rc::new(ScalarLiteralRule),
        Rc::new(ScalarPreservingWrapperRule),
        Rc::new(ScalarAddZeroRule),
        Rc::new(IfWrapperRule),
        Rc::new(ListRestConsRule),
        Rc::new(ListRebuildRule),
    ]
}

fn replace_targets(
    state: &mut ResourceProgramState,
    idx: &mut usize,
    expr: Rc<SExp>,
    specs: &[TargetSpec],
) -> Rc<SExp> {
    for spec in specs {
        if expr.to_string() == spec.source {
            return template_placeholder(
                state,
                idx,
                spec.kind.clone(),
                expr,
                spec.depth,
                spec.allow_literal,
            );
        }
    }

    match expr.borrow() {
        SExp::Cons(loc, left, right) => Rc::new(SExp::Cons(
            loc.clone(),
            replace_targets(state, idx, left.clone(), specs),
            replace_targets(state, idx, right.clone(), specs),
        )),
        _ => expr.clone(),
    }
}

fn load_program(source_name: &str, source: &str, specs: &[TargetSpec]) -> LoadedProgram {
    let loc = Srcloc::start(source_name);
    let parsed = parse_sexp(loc.clone(), source.bytes()).expect("resource program should parse");
    let mut state = ResourceProgramState {
        loc: loc.clone(),
        templates: Vec::new(),
    };
    let mut idx = 0;
    let varied_forms: Vec<Rc<SExp>> = parsed
        .into_iter()
        .map(|form| replace_targets(&mut state, &mut idx, form, specs))
        .collect();
    assert!(
        !state.templates.is_empty(),
        "fuzz target list for {source_name} must match at least one source form"
    );

    LoadedProgram {
        topnode: Rc::new(enlist(loc, &varied_forms)),
        initial_templates: state.templates,
    }
}

fn generated_variation<R: Rng + Sized>(
    rng: &mut R,
    loaded: &LoadedProgram,
    source_name: &str,
) -> String {
    let mut state = ResourceProgramState {
        loc: Srcloc::start(source_name),
        templates: loaded.initial_templates.clone(),
    };
    let rules = fuzz_rules();
    let mut fuzzer = FuzzGenerator::new(loaded.topnode.clone(), &rules);
    let mut expansions = 0;
    while fuzzer
        .expand(
            &mut state,
            expansions > MAX_EXPANSIONS_BEFORE_TERMINATING,
            rng,
        )
        .expect("resource program fuzzer should keep expanding")
    {
        expansions += 1;
        assert!(
            expansions < MAX_EXPANSIONS_TOTAL,
            "resource program fuzzing should terminate for {source_name}"
        );
    }

    list_elements(fuzzer.result())
        .expect("top node is the list of source forms")
        .into_iter()
        .map(|form| form.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn module_opts(filename: &str) -> TestModuleCompilerOpts {
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(filename))
        .set_optimize(true)
        .set_frontend_opt(false)
        .set_search_paths(&["resources/tests/module".to_string()]);
    TestModuleCompilerOpts::new(opts)
}

fn compile_module_component(filename: &str, source: &str) -> Vec<u8> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let source_opts = module_opts(filename);
    let compiled = perform_compile_of_file(
        &mut allocator,
        runner,
        source_opts.clone(),
        filename,
        source,
    )
    .unwrap_or_else(|err| panic!("variation from {filename} should compile: {err:?}"));
    let CompilerOutput::Module(module) = compiled.compiled else {
        panic!("variation from {filename} should compile to a module");
    };
    assert_eq!(
        module.components.len(),
        1,
        "resource fuzz modules should export one runnable program"
    );
    let component_filename = &module.components[0].filename;
    compiled
        .source_opts
        .get_written_file(component_filename)
        .unwrap_or_else(|| panic!("compiled component {component_filename} should be written"))
}

fn compile_program(filename: &str, source: &str) -> Rc<SExp> {
    let mut allocator = Allocator::new();
    let runner = Rc::new(DefaultProgramRunner::new());
    let source_opts = module_opts(filename);
    let compiled = perform_compile_of_file(&mut allocator, runner, source_opts, filename, source)
        .unwrap_or_else(|err| panic!("{filename} should compile: {err:?}"));
    let CompilerOutput::Program(_, program) = compiled.compiled else {
        panic!("{filename} should compile to a program");
    };
    Rc::new(program)
}

fn generate_assign_args() -> Rc<SExp> {
    let program = compile_program(
        "resources/tests/fuzz_test_assign_bug_1_args.clsp",
        ASSIGN_ARGS_PROGRAM,
    );
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new("*fuzz-args-run*"));
    simple_run(opts, program, nil(&Srcloc::start("*fuzz-args*")))
        .expect("assign argument generator should run")
}

fn run_module_component(filename: &str, component_hex: &[u8], args: &str) {
    let _ = run_module_component_result(filename, component_hex, args);
}

fn run_module_component_result(filename: &str, component_hex: &[u8], args: &str) -> Rc<SExp> {
    let mut allocator = Allocator::new();
    let runner = DefaultProgramRunner::new();
    let program = hex_to_clvm(&mut allocator, component_hex);
    let env = assemble(&mut allocator, args).expect("test arguments should assemble");
    let result = runner
        .run_program(&mut allocator, program, env, None)
        .unwrap_or_else(|err| panic!("compiled variation from {filename} should run: {err:?}"))
        .1;
    convert_from_clvm_rs(&mut allocator, Srcloc::start(filename), result)
        .expect("reference program result should convert to SExp")
}

fn run_reference_program(filename: &str, source: &str, args: &str) -> Rc<SExp> {
    let component_hex = compile_module_component(filename, source);
    run_module_component_result(filename, &component_hex, args)
}

fn run_classic_program(filename: &str, source: &str, args: &str) -> Rc<SExp> {
    assert!(source.trim_start().starts_with("(mod "));
    assert!(!source.contains("(include"));
    assert!(!source.contains("(assign"));

    let program = compile_program(filename, source);
    let opts: Rc<dyn CompilerOpts> = Rc::new(DefaultCompilerOpts::new(filename));
    let env = compose_sexp(Srcloc::start(filename), args);
    simple_run(opts, program, env)
        .unwrap_or_else(|err| panic!("classic analogue {filename} should run: {err:?}"))
}

fn assign_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec {
            source: "(f previous_state)",
            kind: ExprKind::List,
            depth: 3,
            allow_literal: false,
        },
        TargetSpec {
            source: "(f game_state)",
            kind: ExprKind::List,
            depth: 3,
            allow_literal: false,
        },
        TargetSpec {
            source: "(+ white_off black_off mover_parity turn_number (strlen commit_hash) (strlen mover_commit) (strlen opponent_seed))",
            kind: ExprKind::Scalar,
            depth: 3,
            allow_literal: true,
        },
        TargetSpec {
            source: "(nth board 0)",
            kind: ExprKind::Scalar,
            depth: 3,
            allow_literal: true,
        },
    ]
}

fn recurse_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec {
            source: "(> (f lst) 0)",
            kind: ExprKind::Scalar,
            depth: 3,
            allow_literal: true,
        },
        TargetSpec {
            source: "(r lst)",
            kind: ExprKind::List,
            depth: 3,
            allow_literal: false,
        },
        TargetSpec {
            source: "(+ i 1)",
            kind: ExprKind::Scalar,
            depth: 3,
            allow_literal: false,
        },
        TargetSpec {
            source: "(walk board 0)",
            kind: ExprKind::Scalar,
            depth: 3,
            allow_literal: true,
        },
    ]
}

fn run_variations(source_name: &str, source: &str, targets: &[TargetSpec], args: &str, seed: u32) {
    let loaded = load_program(source_name, source, targets);
    for variation in 0..VARIATIONS_PER_PROGRAM {
        let mut rng = simple_seeded_rng(seed | variation);
        let program_text = generated_variation(&mut rng, &loaded, source_name);
        let component_hex = compile_module_component(source_name, &program_text);
        run_module_component(source_name, &component_hex, args);
    }
}

#[test]
fn fuzz_resource_program_variations_run() {
    let assign_args = generate_assign_args();
    run_variations(
        "resources/tests/fuzz_test_assign_bug_1.clsp",
        ASSIGN_PROGRAM,
        &assign_targets(),
        &assign_args.to_string(),
        0xA551_0000,
    );
    run_variations(
        "resources/tests/fuzz_test_recurse_bug_0.clsp",
        RECURSE_PROGRAM,
        &recurse_targets(),
        "((1 2 3 4 5 6 7 8 9 10 11 12 13))",
        0x5EC0_0000,
    );
}

#[test]
fn fuzz_test_assign_bug_1_classic_matches_reference() {
    let args = generate_assign_args().to_string();
    let reference = run_reference_program(
        "resources/tests/fuzz_test_assign_bug_1.clsp",
        ASSIGN_PROGRAM,
        &args,
    );
    let classic = run_classic_program(
        "resources/tests/fuzz_test_assign_bug_1_classic.clsp",
        ASSIGN_CLASSIC_PROGRAM,
        &args,
    );
    eprintln!("reference {reference}");
    eprintln!("classic   {classic}");
    assert_eq!(reference, classic);
}

#[test]
fn fuzz_test_recurse_bug_0_classic_matches_reference() {
    let args = "((1 2 3 4 5 6 7 8 9 10 11 12 13))";
    let reference = run_reference_program(
        "resources/tests/fuzz_test_recurse_bug_0.clsp",
        RECURSE_PROGRAM,
        args,
    );
    let classic = run_classic_program(
        "resources/tests/fuzz_test_recurse_bug_0_classic.clsp",
        RECURSE_CLASSIC_PROGRAM,
        args,
    );
    assert_eq!(reference, classic);
}
