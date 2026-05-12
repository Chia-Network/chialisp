use rand::Rng;
use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use clvmr::Allocator;

use crate::classic::clvm_tools::binutils::assemble;
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use crate::compiler::clvm::convert_from_clvm_rs;
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{CompilerOpts, CompilerOutput};
use crate::compiler::fuzz::{ExprModifier, FuzzChoice, FuzzGenerator, FuzzTypeParams, Rule};
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
const ASSIGN_SUBEXPRESSION_MAP: &str = include_str!(
    "../../../resources/tests/fuzz_test_assign_bug_1_modern_classic_subexpression_map.json"
);
const RECURSE_PROGRAM: &str = include_str!("../../../resources/tests/fuzz_test_recurse_bug_0.clsp");
const RECURSE_CLASSIC_PROGRAM: &str =
    include_str!("../../../resources/tests/fuzz_test_recurse_bug_0_classic.clsp");
const RECURSE_SUBEXPRESSION_MAP: &str = include_str!(
    "../../../resources/tests/fuzz_test_recurse_bug_0_modern_classic_subexpression_map.json"
);

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

#[derive(Clone, Debug, Eq, PartialEq)]
struct PairTemplate {
    kind: ExprKind,
    modern_original: Rc<SExp>,
    classic_original: Rc<SExp>,
    depth: usize,
    allow_literal: bool,
}

#[derive(Clone, Debug)]
struct PartialPairTemplate {
    kind: ExprKind,
    modern_original: Rc<SExp>,
    classic_original: Option<Rc<SExp>>,
    depth: usize,
    allow_literal: bool,
    placeholder_idx: usize,
}

#[derive(Clone, Debug)]
struct PairedProgramState {
    loc: Srcloc,
    templates: Vec<PairTemplate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PairedSExp {
    modern: Rc<SExp>,
    classic: Rc<SExp>,
}

struct PairedResourceProgramFuzz;

impl FuzzTypeParams for PairedResourceProgramFuzz {
    type Tag = Vec<u8>;
    type Expr = Rc<PairedSExp>;
    type Error = String;
    type State = PairedProgramState;
}

#[derive(Clone)]
struct PairTargetSpec {
    modern_source: String,
    classic_source: String,
    kind: ExprKind,
    depth: usize,
    allow_literal: bool,
}

#[derive(Clone)]
struct LoadedPairedProgram {
    topnode: Rc<PairedSExp>,
    initial_templates: Vec<PairTemplate>,
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

fn paired(modern: Rc<SExp>, classic: Rc<SExp>) -> Rc<PairedSExp> {
    Rc::new(PairedSExp { modern, classic })
}

fn paired_atom(loc: &Srcloc, name: &str) -> Rc<PairedSExp> {
    paired(atom(loc, name), atom(loc, name))
}

fn paired_integer(loc: &Srcloc, value: i64) -> Rc<PairedSExp> {
    paired(integer(loc, value), integer(loc, value))
}

fn paired_nil(loc: &Srcloc) -> Rc<PairedSExp> {
    paired(nil(loc), nil(loc))
}

fn paired_list(loc: &Srcloc, items: &[Rc<PairedSExp>]) -> Rc<PairedSExp> {
    let modern_items: Vec<Rc<SExp>> = items.iter().map(|item| item.modern.clone()).collect();
    let classic_items: Vec<Rc<SExp>> = items.iter().map(|item| item.classic.clone()).collect();
    paired(list(loc, &modern_items), list(loc, &classic_items))
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

fn paired_list_elements(expr: &Rc<PairedSExp>) -> Option<Vec<Rc<PairedSExp>>> {
    let modern_items = list_elements(&expr.modern)?;
    let classic_items = list_elements(&expr.classic)?;
    if modern_items.len() != classic_items.len() {
        return None;
    }

    Some(
        modern_items
            .into_iter()
            .zip(classic_items)
            .map(|(modern, classic)| paired(modern, classic))
            .collect(),
    )
}

fn find_atom(expr: &Rc<SExp>, atom_name: &[u8]) -> Option<Rc<SExp>> {
    match expr.borrow() {
        SExp::Atom(_, name) if name == atom_name => Some(expr.clone()),
        SExp::Cons(_, left, right) => {
            find_atom(left, atom_name).or_else(|| find_atom(right, atom_name))
        }
        _ => None,
    }
}

impl ExprModifier for Rc<PairedSExp> {
    type Expr = Self;
    type Tag = Vec<u8>;

    fn find_waiters(&self, waiters: &mut Vec<FuzzChoice<Self::Expr, Self::Tag>>) {
        let mut modern_waiters = Vec::new();
        self.modern.find_waiters(&mut modern_waiters);
        for waiter in modern_waiters {
            let placeholder_name =
                as_atom(&waiter.atom).expect("resource fuzzer waiters should be placeholder atoms");
            let classic_atom = find_atom(&self.classic, &placeholder_name)
                .expect("classic paired program should contain the same placeholder");
            waiters.push(FuzzChoice {
                tag: waiter.tag,
                atom: paired(waiter.atom, classic_atom),
            });
        }
    }

    fn replace_node(&self, to_replace: &Self::Expr, new_value: Self::Expr) -> Self::Expr {
        paired(
            self.modern
                .replace_node(&to_replace.modern, new_value.modern.clone()),
            self.classic
                .replace_node(&to_replace.classic, new_value.classic.clone()),
        )
    }

    fn find_in_structure(&self, target: &Self::Expr) -> Option<Vec<Self::Expr>> {
        if self.modern.find_in_structure(&target.modern).is_some()
            && self.classic.find_in_structure(&target.classic).is_some()
        {
            Some(vec![self.clone()])
        } else {
            None
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

fn paired_template_placeholder(
    state: &mut PairedProgramState,
    idx: &mut usize,
    kind: ExprKind,
    original: Rc<PairedSExp>,
    depth: usize,
    allow_literal: bool,
) -> Rc<PairedSExp> {
    let template_id = state.templates.len();
    state.templates.push(PairTemplate {
        kind,
        modern_original: original.modern,
        classic_original: original.classic,
        depth,
        allow_literal,
    });
    let placeholder = placeholder(&state.loc, *idx, template_id);
    *idx += 1;
    paired(placeholder.clone(), placeholder)
}

fn paired_child_template(
    state: &mut PairedProgramState,
    idx: &mut usize,
    parent: &PairTemplate,
    kind: ExprKind,
    original: Rc<PairedSExp>,
) -> Rc<PairedSExp> {
    paired_template_placeholder(
        state,
        idx,
        kind,
        original,
        parent.depth.saturating_sub(1),
        parent.allow_literal,
    )
}

fn skeletonize_paired_original(
    state: &mut PairedProgramState,
    idx: usize,
    template: &PairTemplate,
) -> Rc<PairedSExp> {
    let mut next_idx = idx;
    skeletonize_paired_with_kind(
        state,
        &mut next_idx,
        template,
        template.kind.clone(),
        paired(
            template.modern_original.clone(),
            template.classic_original.clone(),
        ),
    )
}

fn skeletonize_paired_with_kind(
    state: &mut PairedProgramState,
    idx: &mut usize,
    parent: &PairTemplate,
    kind: ExprKind,
    expr: Rc<PairedSExp>,
) -> Rc<PairedSExp> {
    if parent.depth == 0 {
        return expr;
    }

    let Some(items) = paired_list_elements(&expr) else {
        return expr;
    };
    let Some(op) = items.first().and_then(|item| as_atom(&item.modern)) else {
        return expr;
    };
    let op = decode_string(&op);
    let loc = &state.loc.clone();

    match op.as_str() {
        "if" | "i" if items.len() == 4 => paired_list(
            loc,
            &[
                items[0].clone(),
                paired_child_template(state, idx, parent, ExprKind::Scalar, items[1].clone()),
                paired_child_template(state, idx, parent, kind.clone(), items[2].clone()),
                paired_child_template(state, idx, parent, kind, items[3].clone()),
            ],
        ),
        "f" if items.len() == 2 => paired_list(
            loc,
            &[
                items[0].clone(),
                paired_child_template(state, idx, parent, ExprKind::List, items[1].clone()),
            ],
        ),
        "r" if items.len() == 2 => paired_list(
            loc,
            &[
                items[0].clone(),
                paired_child_template(state, idx, parent, ExprKind::List, items[1].clone()),
            ],
        ),
        "c" if items.len() == 3 => paired_list(
            loc,
            &[
                items[0].clone(),
                paired_child_template(state, idx, parent, ExprKind::Scalar, items[1].clone()),
                paired_child_template(state, idx, parent, ExprKind::List, items[2].clone()),
            ],
        ),
        "nth" | "walk" if items.len() == 3 => paired_list(
            loc,
            &[
                items[0].clone(),
                paired_child_template(state, idx, parent, ExprKind::List, items[1].clone()),
                paired_child_template(state, idx, parent, ExprKind::Scalar, items[2].clone()),
            ],
        ),
        "strlen" if items.len() == 2 => paired_list(
            loc,
            &[
                items[0].clone(),
                paired_child_template(state, idx, parent, ExprKind::Scalar, items[1].clone()),
            ],
        ),
        "+" | "-" | "*" | "=" | ">" | "<" | "all" | "any" => {
            let mut replacement = Vec::with_capacity(items.len());
            replacement.push(items[0].clone());
            for item in items.iter().skip(1) {
                replacement.push(paired_child_template(
                    state,
                    idx,
                    parent,
                    ExprKind::Scalar,
                    item.clone(),
                ));
            }
            paired_list(loc, &replacement)
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

impl Rule<PairedResourceProgramFuzz> for OriginalTemplateRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
        let Some(template_id) = decode_template_id(tag) else {
            return Ok(None);
        };
        let template = state
            .templates
            .get(template_id)
            .ok_or("template tag must reference a known template")?
            .clone();

        if terminate || template.depth == 0 {
            return Ok(Some(paired(
                template.modern_original,
                template.classic_original,
            )));
        }

        Ok(Some(skeletonize_paired_original(state, idx, &template)))
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

impl Rule<PairedResourceProgramFuzz> for ScalarLiteralRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
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

        Ok(Some(paired_integer(&state.loc, (idx % 7) as i64)))
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

impl Rule<PairedResourceProgramFuzz> for ScalarPreservingWrapperRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
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

        Ok(Some(paired_list(
            &state.loc,
            &[
                paired_atom(&state.loc, "f"),
                paired_list(
                    &state.loc,
                    &[
                        paired_atom(&state.loc, "c"),
                        paired(
                            template.modern_original.clone(),
                            template.classic_original.clone(),
                        ),
                        paired_nil(&state.loc),
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

impl Rule<PairedResourceProgramFuzz> for ScalarAddZeroRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
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
        let child = paired_template_placeholder(
            state,
            &mut next_idx,
            ExprKind::Scalar,
            paired(template.modern_original, template.classic_original),
            template.depth - 1,
            template.allow_literal,
        );
        Ok(Some(paired_list(
            &state.loc,
            &[
                paired_atom(&state.loc, "+"),
                child,
                paired_integer(&state.loc, 0),
            ],
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

impl Rule<PairedResourceProgramFuzz> for IfWrapperRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
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

        let original = paired(
            template.modern_original.clone(),
            template.classic_original.clone(),
        );
        Ok(Some(paired_list(
            &state.loc,
            &[
                paired_atom(&state.loc, "if"),
                paired_integer(&state.loc, 1),
                original.clone(),
                original,
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

impl Rule<PairedResourceProgramFuzz> for ListRestConsRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
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
        let child = paired_template_placeholder(
            state,
            &mut next_idx,
            ExprKind::List,
            paired(template.modern_original, template.classic_original),
            template.depth - 1,
            template.allow_literal,
        );
        Ok(Some(paired_list(
            &state.loc,
            &[
                paired_atom(&state.loc, "r"),
                paired_list(
                    &state.loc,
                    &[
                        paired_atom(&state.loc, "c"),
                        paired_integer(&state.loc, 0),
                        child,
                    ],
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

impl Rule<PairedResourceProgramFuzz> for ListRebuildRule {
    fn check(
        &self,
        state: &mut PairedProgramState,
        tag: &Vec<u8>,
        _idx: usize,
        terminate: bool,
        _parents: &[Rc<PairedSExp>],
    ) -> Result<Option<Rc<PairedSExp>>, String> {
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

        let original = paired(
            template.modern_original.clone(),
            template.classic_original.clone(),
        );
        Ok(Some(paired_list(
            &state.loc,
            &[
                paired_atom(&state.loc, "c"),
                paired_list(
                    &state.loc,
                    &[paired_atom(&state.loc, "f"), original.clone()],
                ),
                paired_list(&state.loc, &[paired_atom(&state.loc, "r"), original]),
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

fn paired_fuzz_rules() -> Vec<Rc<dyn Rule<PairedResourceProgramFuzz>>> {
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

fn canonical_form(source_name: &str, source: &str) -> String {
    let loc = Srcloc::start(source_name);
    let parsed =
        parse_sexp(loc, source.bytes()).unwrap_or_else(|err| panic!("{source_name}: {err:?}"));
    assert_eq!(
        parsed.len(),
        1,
        "{source_name} should contain exactly one Chialisp form"
    );
    parsed[0].to_string()
}

fn load_subexpression_map(map_name: &str, map_json: &str) -> HashMap<String, String> {
    let raw: HashMap<String, String> =
        serde_json::from_str(map_json).unwrap_or_else(|err| panic!("{map_name}: {err:?}"));
    raw.into_iter()
        .map(|(modern, classic)| {
            (
                canonical_form(&format!("{map_name}:modern"), &modern),
                canonical_form(&format!("{map_name}:classic"), &classic),
            )
        })
        .collect()
}

fn paired_targets(targets: &[TargetSpec], map_name: &str, map_json: &str) -> Vec<PairTargetSpec> {
    let subexpression_map = load_subexpression_map(map_name, map_json);
    targets
        .iter()
        .map(|target| {
            let modern_source = canonical_form("modern fuzz target", target.source);
            let classic_source = subexpression_map
                .get(&modern_source)
                .unwrap_or_else(|| panic!("{map_name} should map fuzz target {modern_source}"))
                .clone();
            PairTargetSpec {
                modern_source,
                classic_source,
                kind: target.kind.clone(),
                depth: target.depth,
                allow_literal: target.allow_literal,
            }
        })
        .collect()
}

fn count_form_matches(expr: &Rc<SExp>, target: &str) -> usize {
    let self_match = usize::from(expr.to_string() == target);
    match expr.borrow() {
        SExp::Cons(_, left, right) => {
            self_match + count_form_matches(left, target) + count_form_matches(right, target)
        }
        _ => self_match,
    }
}

fn assert_map_forms_exist(
    map_name: &str,
    map_json: &str,
    modern_name: &str,
    modern_source: &str,
    classic_name: &str,
    classic_source: &str,
) {
    let subexpression_map = load_subexpression_map(map_name, map_json);
    let modern_forms = parse_sexp(Srcloc::start(modern_name), modern_source.bytes())
        .expect("modern resource program should parse");
    let classic_forms = parse_sexp(Srcloc::start(classic_name), classic_source.bytes())
        .expect("classic resource program should parse");
    let modern_top = Rc::new(enlist(Srcloc::start(modern_name), &modern_forms));
    let classic_top = Rc::new(enlist(Srcloc::start(classic_name), &classic_forms));

    for (modern, classic) in subexpression_map {
        assert!(
            count_form_matches(&modern_top, &modern) > 0,
            "{map_name} modern form should exist in {modern_name}: {modern}"
        );
        assert!(
            count_form_matches(&classic_top, &classic) > 0,
            "{map_name} classic form should exist in {classic_name}: {classic}"
        );
    }
}

fn replace_modern_pair_targets(
    loc: &Srcloc,
    templates: &mut Vec<PartialPairTemplate>,
    classic_queues: &mut HashMap<String, VecDeque<usize>>,
    idx: &mut usize,
    expr: Rc<SExp>,
    specs: &[PairTargetSpec],
) -> Rc<SExp> {
    let expr_string = expr.to_string();
    for spec in specs {
        if expr_string == spec.modern_source {
            let template_id = templates.len();
            templates.push(PartialPairTemplate {
                kind: spec.kind.clone(),
                modern_original: expr,
                classic_original: None,
                depth: spec.depth,
                allow_literal: spec.allow_literal,
                placeholder_idx: *idx,
            });
            classic_queues
                .entry(spec.classic_source.clone())
                .or_default()
                .push_back(template_id);
            let placeholder = placeholder(loc, *idx, template_id);
            *idx += 1;
            return placeholder;
        }
    }

    match expr.borrow() {
        SExp::Cons(expr_loc, left, right) => Rc::new(SExp::Cons(
            expr_loc.clone(),
            replace_modern_pair_targets(loc, templates, classic_queues, idx, left.clone(), specs),
            replace_modern_pair_targets(loc, templates, classic_queues, idx, right.clone(), specs),
        )),
        _ => expr.clone(),
    }
}

fn replace_classic_pair_targets(
    loc: &Srcloc,
    templates: &mut [PartialPairTemplate],
    classic_queues: &mut HashMap<String, VecDeque<usize>>,
    expr: Rc<SExp>,
    specs: &[PairTargetSpec],
) -> Rc<SExp> {
    let expr_string = expr.to_string();
    for spec in specs {
        if expr_string == spec.classic_source {
            if let Some(template_id) = classic_queues
                .get_mut(&spec.classic_source)
                .and_then(VecDeque::pop_front)
            {
                templates[template_id].classic_original = Some(expr);
                return placeholder(loc, templates[template_id].placeholder_idx, template_id);
            }
        }
    }

    match expr.borrow() {
        SExp::Cons(expr_loc, left, right) => Rc::new(SExp::Cons(
            expr_loc.clone(),
            replace_classic_pair_targets(loc, templates, classic_queues, left.clone(), specs),
            replace_classic_pair_targets(loc, templates, classic_queues, right.clone(), specs),
        )),
        _ => expr.clone(),
    }
}

fn load_paired_program(
    source_name: &str,
    source: &str,
    classic_name: &str,
    classic_source: &str,
    specs: &[PairTargetSpec],
) -> LoadedPairedProgram {
    let loc = Srcloc::start(source_name);
    let modern_parsed =
        parse_sexp(loc.clone(), source.bytes()).expect("modern resource program should parse");
    let classic_loc = Srcloc::start(classic_name);
    let classic_parsed = parse_sexp(classic_loc, classic_source.bytes())
        .expect("classic resource program should parse");
    let mut templates = Vec::new();
    let mut classic_queues = HashMap::new();
    let mut idx = 0;
    let modern_forms: Vec<Rc<SExp>> = modern_parsed
        .into_iter()
        .map(|form| {
            replace_modern_pair_targets(
                &loc,
                &mut templates,
                &mut classic_queues,
                &mut idx,
                form,
                specs,
            )
        })
        .collect();
    assert!(
        !templates.is_empty(),
        "fuzz target list for {source_name} must match at least one source form"
    );

    let classic_forms: Vec<Rc<SExp>> = classic_parsed
        .into_iter()
        .map(|form| {
            replace_classic_pair_targets(&loc, &mut templates, &mut classic_queues, form, specs)
        })
        .collect();
    for (classic_source, remaining) in classic_queues {
        assert!(
            remaining.is_empty(),
            "{classic_name} should contain a mapped classic form for every modern occurrence: {classic_source}"
        );
    }

    let initial_templates: Vec<PairTemplate> = templates
        .into_iter()
        .enumerate()
        .map(|(idx, template)| PairTemplate {
            kind: template.kind,
            modern_original: template.modern_original,
            classic_original: template.classic_original.unwrap_or_else(|| {
                panic!("{classic_name} should fill classic original for template {idx}")
            }),
            depth: template.depth,
            allow_literal: template.allow_literal,
        })
        .collect();

    LoadedPairedProgram {
        topnode: paired(
            Rc::new(enlist(loc.clone(), &modern_forms)),
            Rc::new(enlist(loc, &classic_forms)),
        ),
        initial_templates,
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

fn generated_paired_variation<R: Rng + Sized>(
    rng: &mut R,
    loaded: &LoadedPairedProgram,
    source_name: &str,
) -> (String, String) {
    let mut state = PairedProgramState {
        loc: Srcloc::start(source_name),
        templates: loaded.initial_templates.clone(),
    };
    let rules = paired_fuzz_rules();
    let mut fuzzer = FuzzGenerator::new(loaded.topnode.clone(), &rules);
    let mut expansions = 0;
    while fuzzer
        .expand(
            &mut state,
            expansions > MAX_EXPANSIONS_BEFORE_TERMINATING,
            rng,
        )
        .expect("paired resource program fuzzer should keep expanding")
    {
        expansions += 1;
        assert!(
            expansions < MAX_EXPANSIONS_TOTAL,
            "paired resource program fuzzing should terminate for {source_name}"
        );
    }

    let result = fuzzer.result();
    let modern = list_elements(&result.modern)
        .expect("modern top node is the list of source forms")
        .into_iter()
        .map(|form| form.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let classic = list_elements(&result.classic)
        .expect("classic top node is the list of source forms")
        .into_iter()
        .map(|form| form.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    (modern, classic)
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

fn run_paired_variations(
    source_name: &str,
    source: &str,
    classic_name: &str,
    classic_source: &str,
    map_name: &str,
    map_json: &str,
    targets: &[TargetSpec],
    args: &str,
    seed: u32,
) {
    let paired_targets = paired_targets(targets, map_name, map_json);
    let loaded = load_paired_program(
        source_name,
        source,
        classic_name,
        classic_source,
        &paired_targets,
    );
    for variation in 0..VARIATIONS_PER_PROGRAM {
        let mut rng = simple_seeded_rng(seed | variation);
        let (modern_text, classic_text) =
            generated_paired_variation(&mut rng, &loaded, source_name);
        let reference = run_reference_program(source_name, &modern_text, args);
        let classic = run_classic_program(classic_name, &classic_text, args);
        assert_eq!(
            reference, classic,
            "paired variation {variation} from {source_name} should match {classic_name}"
        );
    }
}

#[test]
fn fuzz_resource_subexpression_maps_match_sources() {
    assert_map_forms_exist(
        "resources/tests/fuzz_test_assign_bug_1_modern_classic_subexpression_map.json",
        ASSIGN_SUBEXPRESSION_MAP,
        "resources/tests/fuzz_test_assign_bug_1.clsp",
        ASSIGN_PROGRAM,
        "resources/tests/fuzz_test_assign_bug_1_classic.clsp",
        ASSIGN_CLASSIC_PROGRAM,
    );
    assert_map_forms_exist(
        "resources/tests/fuzz_test_recurse_bug_0_modern_classic_subexpression_map.json",
        RECURSE_SUBEXPRESSION_MAP,
        "resources/tests/fuzz_test_recurse_bug_0.clsp",
        RECURSE_PROGRAM,
        "resources/tests/fuzz_test_recurse_bug_0_classic.clsp",
        RECURSE_CLASSIC_PROGRAM,
    );
}

#[test]
fn fuzz_resource_program_variations_run() {
    let assign_args = generate_assign_args();
    run_paired_variations(
        "resources/tests/fuzz_test_assign_bug_1.clsp",
        ASSIGN_PROGRAM,
        "resources/tests/fuzz_test_assign_bug_1_classic.clsp",
        ASSIGN_CLASSIC_PROGRAM,
        "resources/tests/fuzz_test_assign_bug_1_modern_classic_subexpression_map.json",
        ASSIGN_SUBEXPRESSION_MAP,
        &assign_targets(),
        &assign_args.to_string(),
        0xA551_0000,
    );
    run_paired_variations(
        "resources/tests/fuzz_test_recurse_bug_0.clsp",
        RECURSE_PROGRAM,
        "resources/tests/fuzz_test_recurse_bug_0_classic.clsp",
        RECURSE_CLASSIC_PROGRAM,
        "resources/tests/fuzz_test_recurse_bug_0_modern_classic_subexpression_map.json",
        RECURSE_SUBEXPRESSION_MAP,
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
