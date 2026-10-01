//! Versioned, structural debug metadata for serialized CLVM programs.
//!
//! The persisted form is CLVM serialization, not a hash-indexed side table:
//!
//! ```text
//! ("CHIALISP_DEBUG" 1 (1 8) PROGRAM_SHA256
//!   ((PATH FULL_UTF8_SOURCE) ...)
//!   (STRING ...)
//!   ((FILE START_LINE START_COLUMN END_LINE END_COLUMN) ...)
//!   ((FUNCTION_NAME LEFT_ENV PARAMETER_TREE) ...)
//!   SHADOW_TREE)
//! ```
//!
//! Coordinates are one-based display columns. Tabs advance to the next
//! one-based 8-column tab stop, matching [`Srcloc::advance`]. Span ends are
//! exclusive. Parameter trees use `(0)` for nil, `(1 NAME PATH CONSTRAINT)` for
//! names, and `(2 LEFT RIGHT)` for destructuring. The shadow tree uses
//! `(0 SPAN STRING FUNCTION ATOM)` for atoms and
//! `(1 SPAN STRING FUNCTION LEFT RIGHT)` for pairs. Table references are
//! encoded as `index + 1`; zero means absent.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr};
use clvm_rs::serde::node_to_bytes;
use num_bigint::ToBigInt;
use sha2::{Digest, Sha256};

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::keyword_from_atom;
use crate::classic::clvm::serialize::{sexp_from_stream, sexp_to_stream, SimpleCreateCLVMObject};
use crate::classic::clvm_tools::binutils::disassemble_with_kw;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::compiler::compile_file;
use crate::compiler::comptypes::{
    CompileErr, CompilerOpts, CompilerOutput, HasCompilerOptsDelegation,
};
use crate::compiler::dialect::KNOWN_DIALECTS;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::{src_location_max, Srcloc};
use crate::util::u8_from_number;

const MAGIC: &[u8] = b"CHIALISP_DEBUG";
pub const DEBUG_METADATA_VERSION: usize = 1;
pub const DEBUG_METADATA_TAB_WIDTH: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugSourceFile {
    pub path: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugSourceSpan {
    pub file: usize,
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DebugNode {
    Atom {
        span: Option<usize>,
        label: Option<usize>,
        function: Option<usize>,
        value: Vec<u8>,
    },
    Pair {
        span: Option<usize>,
        label: Option<usize>,
        function: Option<usize>,
        left: Box<DebugNode>,
        right: Box<DebugNode>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DebugParameter {
    Nil,
    Name {
        name: usize,
        path: Vec<u8>,
        constraint: ParameterConstraint,
    },
    Pair(Box<DebugParameter>, Box<DebugParameter>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugFunction {
    pub name: usize,
    pub left_env: bool,
    pub parameters: DebugParameter,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugMetadata {
    pub program_sha256: [u8; 32],
    pub files: Vec<DebugSourceFile>,
    pub strings: Vec<String>,
    pub spans: Vec<DebugSourceSpan>,
    pub functions: Vec<DebugFunction>,
    pub tree: DebugNode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugCompileArtifact {
    /// Export name for module components; `None` for an ordinary program or
    /// module summary.
    pub export_name: Option<String>,
    pub program: Vec<u8>,
    pub metadata: Vec<u8>,
    /// The legacy hash symbol map remains available and unchanged in shape.
    pub symbols: HashMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameMatch {
    Exact,
    Curried,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolizedFrame {
    pub matched: FrameMatch,
    pub function: Option<String>,
    pub function_index: Option<usize>,
    pub source_span: Option<usize>,
    /// CLVM tree hash of the executable frame, retained even when unknown.
    pub program_hash: [u8; 32],
    /// Canonical CLVM serialization of values quoted into curry wrappers.
    pub bound_arguments: Vec<Vec<u8>>,
    /// Canonical CLVM serialization of arguments supplied by the caller.
    pub runtime_arguments: Vec<Vec<u8>>,
    /// Fully named and constrained arguments derived from the sidecar.
    pub arguments: Vec<StackFrameArgument>,
}

/// A raw evaluator frame whose program and environment are canonical CLVM
/// serializations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerializedFrame {
    pub program: Vec<u8>,
    pub environment: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolizedSerializedFrame {
    /// Index of the matching sidecar in its [`DebugMetadataCollection`].
    /// Unknown frames do not have one.
    pub metadata_index: Option<usize>,
    pub frame: SymbolizedFrame,
}

/// Decoded sidecars indexed by the structural identities of their programs
/// and nested executable forms.
#[derive(Clone, Debug, Default)]
pub struct DebugMetadataCollection {
    entries: Vec<DebugMetadata>,
    program_identities: HashMap<[u8; 32], usize>,
    structural_identities: HashMap<[u8; 32], Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParameterConstraint {
    Integer,
    Atom,
    Bytes(usize),
    Pair,
    ProperList,
    Unknown,
    Union(Vec<ParameterConstraint>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentBinding {
    Bound,
    Runtime,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackFrameArgument {
    pub name: String,
    pub binding: ArgumentBinding,
    pub constraint: ParameterConstraint,
    pub value: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackFrameStyle {
    Lisp,
    Python,
}

#[derive(Default)]
struct InternState {
    files: Vec<DebugSourceFile>,
    file_indices: HashMap<String, usize>,
    source_indices: Vec<SourceIndex>,
    strings: Vec<String>,
    string_indices: HashMap<String, usize>,
    spans: Vec<DebugSourceSpan>,
    span_indices: HashMap<(usize, usize, usize, usize, usize), usize>,
}

#[derive(Default)]
struct SourceIndex {
    line_starts: Vec<usize>,
}

impl SourceIndex {
    fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(
            source
                .bytes()
                .enumerate()
                .filter_map(|(index, byte)| (byte == b'\n').then_some(index + 1)),
        );
        Self { line_starts }
    }

    fn line<'a>(&self, source: &'a str, line: usize) -> Option<&'a str> {
        let index = line.checked_sub(1)?;
        let start = *self.line_starts.get(index)?;
        if start == source.len() && (source.is_empty() || source.ends_with('\n')) {
            return None;
        }
        let mut end = self
            .line_starts
            .get(index + 1)
            .map(|next| next - 1)
            .unwrap_or(source.len());
        if end > start
            && self.line_starts.get(index + 1).is_some()
            && source.as_bytes()[end - 1] == b'\r'
        {
            end -= 1;
        }
        Some(&source[start..end])
    }
}

fn atom_bytes(value: &SExp) -> Vec<u8> {
    match value {
        SExp::Nil(_) => Vec::new(),
        SExp::Atom(_, value) | SExp::QuotedString(_, _, value) => value.clone(),
        SExp::Integer(_, value) => u8_from_number(value.clone()),
        SExp::Cons(_, _, _) => unreachable!("pair passed to atom_bytes"),
    }
}

fn serialize_program(program: &SExp) -> Result<Vec<u8>, String> {
    let mut allocator = Allocator::new();
    let node = convert_to_clvm_rs(&mut allocator, Rc::new(program.clone()))
        .map_err(|error| format!("cannot serialize debug program: {error:?}"))?;
    let mut stream = Stream::new(None);
    sexp_to_stream(&mut allocator, node, &mut stream);
    Ok(stream.get_value().data().clone())
}

struct IndexedProgramNode<'a> {
    program: &'a SExp,
    hash: [u8; 32],
    children: Option<(Box<IndexedProgramNode<'a>>, Box<IndexedProgramNode<'a>>)>,
}

struct ProgramIndex<'a> {
    root: IndexedProgramNode<'a>,
    first_by_hash: HashMap<[u8; 32], &'a SExp>,
    #[cfg(test)]
    hash_visits: usize,
    #[cfg(test)]
    subtree_lookups: Cell<usize>,
}

impl<'a> ProgramIndex<'a> {
    fn new(program: &'a SExp) -> Self {
        let mut hash_visits = 0;
        let root = Self::build_node(program, &mut hash_visits);
        let mut first_by_hash = HashMap::new();
        Self::index_first_occurrences(&root, &mut first_by_hash);
        Self {
            root,
            first_by_hash,
            #[cfg(test)]
            hash_visits,
            #[cfg(test)]
            subtree_lookups: Cell::new(0),
        }
    }

    fn build_node(program: &'a SExp, hash_visits: &mut usize) -> IndexedProgramNode<'a> {
        *hash_visits += 1;
        match program {
            SExp::Cons(_, left, right) => {
                let left = Box::new(Self::build_node(left, hash_visits));
                let right = Box::new(Self::build_node(right, hash_visits));
                let mut hasher = Sha256::new();
                hasher.update([2]);
                hasher.update(left.hash);
                hasher.update(right.hash);
                IndexedProgramNode {
                    program,
                    hash: hasher.finalize().into(),
                    children: Some((left, right)),
                }
            }
            _ => {
                IndexedProgramNode {
                    program,
                    // The legacy helper preserves the compiler's active integer
                    // conversion convention. This clones only the leaf atom,
                    // never a subtree.
                    hash: crate::compiler::clvm::sha256tree(Rc::new(program.clone()))
                        .try_into()
                        .expect("tree hashes are 32 bytes"),
                    children: None,
                }
            }
        }
    }

    fn index_first_occurrences(
        node: &IndexedProgramNode<'a>,
        first_by_hash: &mut HashMap<[u8; 32], &'a SExp>,
    ) {
        first_by_hash.entry(node.hash).or_insert(node.program);
        if let Some((left, right)) = &node.children {
            Self::index_first_occurrences(left, first_by_hash);
            Self::index_first_occurrences(right, first_by_hash);
        }
    }

    fn find_subtree_by_hash(&self, hash: &str) -> Option<&'a SExp> {
        #[cfg(test)]
        self.subtree_lookups.set(self.subtree_lookups.get() + 1);
        let decoded: [u8; 32] = hex::decode(hash).ok()?.try_into().ok()?;
        (hash == hex::encode(decoded))
            .then(|| self.first_by_hash.get(&decoded).copied())
            .flatten()
    }

    fn build_symbol_table(&self, symbols: &mut HashMap<String, String>) {
        Self::build_symbol_table_node(&self.root, symbols);
    }

    fn build_symbol_table_node(
        node: &IndexedProgramNode<'_>,
        symbols: &mut HashMap<String, String>,
    ) {
        if let Some((left, right)) = &node.children {
            Self::build_symbol_table_node(left, symbols);
            Self::build_symbol_table_node(right, symbols);
            symbols
                .entry(hex::encode(node.hash))
                .or_insert_with(|| node.program.loc().to_string());
        } else {
            symbols.insert(hex::encode(node.hash), node.program.loc().to_string());
        }
    }
}

#[cfg(test)]
fn tree_hash_hex(program: &SExp) -> String {
    hex::encode(ProgramIndex::new(program).root.hash)
}

fn source_for_file(path: &str, supplied: &HashMap<String, String>) -> Result<String, String> {
    supplied
        .get(path)
        .cloned()
        .or_else(|| {
            KNOWN_DIALECTS
                .get(path)
                .map(|dialect| dialect.content.to_string())
        })
        // Compiler-generated locations name virtual files. They have no
        // external source text; their complete source is the empty string.
        .or_else(|| (path.starts_with('*') && path.ends_with('*')).then(String::new))
        .ok_or_else(|| format!("debug metadata has no UTF-8 source for {path}"))
}

pub(crate) type CapturedSources = Rc<RefCell<HashMap<String, String>>>;

#[derive(Clone)]
struct SourceCapturingCompilerOpts {
    opts: Rc<dyn CompilerOpts>,
    sources: CapturedSources,
}

impl HasCompilerOptsDelegation for SourceCapturingCompilerOpts {
    fn compiler_opts(&self) -> Rc<dyn CompilerOpts> {
        self.opts.clone()
    }

    fn update_compiler_opts<F: FnOnce(Rc<dyn CompilerOpts>) -> Rc<dyn CompilerOpts>>(
        &self,
        f: F,
    ) -> Rc<dyn CompilerOpts> {
        Rc::new(Self {
            opts: f(self.opts.clone()),
            sources: self.sources.clone(),
        })
    }

    fn override_read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr> {
        let requested = filename.clone();
        let (resolved, content) = self.opts.read_new_file(inc_from, filename)?;
        if let Ok(source) = String::from_utf8(content.clone()) {
            let mut sources = self.sources.borrow_mut();
            sources.insert(requested, source.clone());
            sources.insert(resolved.clone(), source);
        }
        Ok((resolved, content))
    }
}

pub(crate) fn capture_compiler_sources(
    opts: Rc<dyn CompilerOpts>,
    main_path: String,
    main_source: String,
) -> (Rc<dyn CompilerOpts>, CapturedSources) {
    let sources = Rc::new(RefCell::new(HashMap::from([(main_path, main_source)])));
    (
        Rc::new(SourceCapturingCompilerOpts {
            opts,
            sources: sources.clone(),
        }),
        sources,
    )
}

impl InternState {
    fn intern_file(
        &mut self,
        path: &str,
        supplied: &HashMap<String, String>,
    ) -> Result<usize, String> {
        if let Some(index) = self.file_indices.get(path) {
            return Ok(*index);
        }
        let index = self.files.len();
        let source = source_for_file(path, supplied)?;
        self.source_indices.push(SourceIndex::new(&source));
        self.files.push(DebugSourceFile {
            path: path.to_string(),
            source,
        });
        self.file_indices.insert(path.to_string(), index);
        Ok(index)
    }

    fn intern_string(&mut self, value: &str) -> usize {
        if let Some(index) = self.string_indices.get(value) {
            return *index;
        }
        let index = self.strings.len();
        self.strings.push(value.to_string());
        self.string_indices.insert(value.to_string(), index);
        index
    }

    fn intern_span(
        &mut self,
        loc: &Srcloc,
        supplied: &HashMap<String, String>,
    ) -> Result<usize, String> {
        let file = self.intern_file(loc.file.as_str(), supplied)?;
        let (end_line, end_column) = src_location_max(loc);
        let source = &self.files[file].source;
        let source_index = &self.source_indices[file];
        let start_column = source_display_column(source, source_index, loc.line, loc.col);
        let end_column = source_display_column(source, source_index, end_line, end_column);
        let key = (file, loc.line, start_column, end_line, end_column);
        if let Some(index) = self.span_indices.get(&key) {
            return Ok(*index);
        }
        let index = self.spans.len();
        self.spans.push(DebugSourceSpan {
            file,
            start_line: loc.line,
            start_column,
            end_line,
            end_column,
        });
        self.span_indices.insert(key, index);
        Ok(index)
    }
}

fn source_display_column(
    source: &str,
    source_index: &SourceIndex,
    line: usize,
    compiler_column: usize,
) -> usize {
    let Some(source_line) = source_index.line(source, line) else {
        return compiler_column;
    };
    let mut byte_column = 1usize;
    let mut display_column = 1usize;
    for character in source_line.chars() {
        if byte_column >= compiler_column {
            break;
        }
        if character == '\t' {
            byte_column =
                (byte_column + DEBUG_METADATA_TAB_WIDTH) & !(DEBUG_METADATA_TAB_WIDTH - 1);
            display_column =
                (display_column + DEBUG_METADATA_TAB_WIDTH) & !(DEBUG_METADATA_TAB_WIDTH - 1);
        } else {
            byte_column += character.len_utf8();
            display_column += 1;
        }
    }
    display_column + compiler_column.saturating_sub(byte_column)
}

fn build_tree(
    program: &IndexedProgramNode<'_>,
    symbols: &HashMap<String, String>,
    functions_by_hash: &HashMap<String, usize>,
    sources: &HashMap<String, String>,
    state: &mut InternState,
) -> Result<DebugNode, String> {
    let span = Some(state.intern_span(&program.program.loc(), sources)?);
    let hash = hex::encode(program.hash);
    let label = symbols
        .get(&hash)
        .filter(|name| !name.contains('('))
        .map(|name| state.intern_string(name));
    let function = functions_by_hash.get(&hash).copied();
    match &program.children {
        Some((left, right)) => Ok(DebugNode::Pair {
            span,
            label,
            function,
            left: Box::new(build_tree(
                left,
                symbols,
                functions_by_hash,
                sources,
                state,
            )?),
            right: Box::new(build_tree(
                right,
                symbols,
                functions_by_hash,
                sources,
                state,
            )?),
        }),
        None => Ok(DebugNode::Atom {
            span,
            label,
            function,
            value: atom_bytes(program.program),
        }),
    }
}

fn quoted_apply_body(program: &SExp) -> Option<&SExp> {
    let SExp::Cons(_, operator, arguments) = program else {
        return None;
    };
    let SExp::Cons(_, quoted_program, environment_tail) = arguments.as_ref() else {
        return None;
    };
    let SExp::Cons(_, _, final_tail) = environment_tail.as_ref() else {
        return None;
    };
    let SExp::Nil(_) = final_tail.as_ref() else {
        return None;
    };
    let SExp::Cons(_, quote_operator, body) = quoted_program.as_ref() else {
        return None;
    };
    (atom_bytes(operator) == [2] && atom_bytes(quote_operator) == [1]).then_some(body.as_ref())
}

fn build_parameter_shape(
    formal: &SExp,
    directions: &mut Vec<bool>,
    state: &mut InternState,
) -> DebugParameter {
    match formal.atomize() {
        SExp::Nil(_) => DebugParameter::Nil,
        SExp::Cons(_, left, right) => {
            directions.push(false);
            let left = build_parameter_shape(left.as_ref(), directions, state);
            directions.pop();
            directions.push(true);
            let right = build_parameter_shape(right.as_ref(), directions, state);
            directions.pop();
            DebugParameter::Pair(Box::new(left), Box::new(right))
        }
        SExp::Atom(_, name) => DebugParameter::Name {
            name: state.intern_string(&String::from_utf8_lossy(&name)),
            path: parameter_path(directions),
            constraint: ParameterConstraint::Unknown,
        },
        other => DebugParameter::Name {
            name: state.intern_string(&other.to_string()),
            path: parameter_path(directions),
            constraint: ParameterConstraint::Unknown,
        },
    }
}

fn parameter_path(directions: &[bool]) -> Vec<u8> {
    let mut path = 1_u8.to_bigint().unwrap() << directions.len();
    for (index, rest) in directions.iter().enumerate() {
        if *rest {
            path += 1_u8.to_bigint().unwrap() << index;
        }
    }
    u8_from_number(path)
}

fn parameter_paths(
    parameter: &DebugParameter,
    strings: &[String],
    result: &mut Vec<(String, Vec<u8>)>,
) {
    match parameter {
        DebugParameter::Nil => {}
        DebugParameter::Name { name, path, .. } => {
            result.push((strings[*name].clone(), path.clone()));
        }
        DebugParameter::Pair(left, right) => {
            parameter_paths(left, strings, result);
            parameter_paths(right, strings, result);
        }
    }
}

fn apply_parameter_constraints(
    parameter: &mut DebugParameter,
    constraints: &mut impl Iterator<Item = ParameterConstraint>,
) {
    match parameter {
        DebugParameter::Nil => {}
        DebugParameter::Name { constraint, .. } => {
            *constraint = constraints.next().unwrap_or(ParameterConstraint::Unknown);
        }
        DebugParameter::Pair(left, right) => {
            apply_parameter_constraints(left, constraints);
            apply_parameter_constraints(right, constraints);
        }
    }
}

fn parse_formal_parameters(text: &str) -> Result<SExp, String> {
    parse_sexp(Srcloc::compiler_internal_srcloc(), text.bytes())
        .map_err(|error| format!("invalid compiler formal parameter metadata: {error:?}"))?
        .into_iter()
        .next()
        .map(|value| value.atomize())
        .ok_or_else(|| "empty compiler formal parameter metadata".to_string())
}

struct FunctionRecordSpec<'a> {
    hash: &'a str,
    name: &'a str,
    arguments: &'a str,
    left_env: bool,
}

fn add_function_record(
    program_index: &ProgramIndex<'_>,
    spec: FunctionRecordSpec<'_>,
    state: &mut InternState,
    functions: &mut Vec<DebugFunction>,
    functions_by_hash: &mut HashMap<String, usize>,
) -> Result<(), String> {
    let Some(function_program) = program_index.find_subtree_by_hash(spec.hash) else {
        return Ok(());
    };
    let formal = parse_formal_parameters(spec.arguments)?;
    let mut directions = if spec.left_env {
        vec![true]
    } else {
        Vec::new()
    };
    let mut parameters = build_parameter_shape(&formal, &mut directions, state);
    let mut paths = Vec::new();
    parameter_paths(&parameters, &state.strings, &mut paths);
    let inference_program = quoted_apply_body(function_program).unwrap_or(function_program);
    let inferred = infer_parameter_constraints_sexp(inference_program, &paths);
    apply_parameter_constraints(
        &mut parameters,
        &mut inferred.into_iter().map(|(_, constraint)| constraint),
    );
    let index = functions.len();
    functions.push(DebugFunction {
        name: state.intern_string(spec.name),
        left_env: spec.left_env,
        parameters,
    });
    functions_by_hash.insert(spec.hash.to_string(), index);
    Ok(())
}

fn build_functions(
    program_index: &ProgramIndex<'_>,
    symbols: &HashMap<String, String>,
    state: &mut InternState,
) -> Result<(Vec<DebugFunction>, HashMap<String, usize>), String> {
    let mut hashes = symbols
        .keys()
        .filter(|key| {
            key.len() == 64
                && key.bytes().all(|byte| byte.is_ascii_hexdigit())
                && symbols.get(*key).is_some_and(|value| !value.contains('('))
        })
        .cloned()
        .collect::<Vec<_>>();
    hashes.sort();
    hashes.dedup();

    let mut functions = Vec::new();
    let mut functions_by_hash = HashMap::new();
    for hash in hashes {
        add_function_record(
            program_index,
            FunctionRecordSpec {
                hash: &hash,
                name: &symbols[&hash],
                arguments: symbols
                    .get(&format!("{hash}_arguments"))
                    .map(String::as_str)
                    .unwrap_or("()"),
                left_env: symbols
                    .get(&format!("{hash}_left_env"))
                    .is_some_and(|value| value != "0" && value != "()"),
            },
            state,
            &mut functions,
            &mut functions_by_hash,
        )?;
    }

    if let Some(arguments) = symbols.get("__chia__main_arguments") {
        let hash = hex::encode(program_index.root.hash);
        if !functions_by_hash.contains_key(&hash) {
            add_function_record(
                program_index,
                FunctionRecordSpec {
                    hash: &hash,
                    name: "<main>",
                    arguments,
                    left_env: false,
                },
                state,
                &mut functions,
                &mut functions_by_hash,
            )?;
        }
    }
    Ok((functions, functions_by_hash))
}

impl DebugMetadata {
    pub fn from_program(
        program: &SExp,
        symbols: &HashMap<String, String>,
        sources: &HashMap<String, String>,
    ) -> Result<Self, String> {
        let program_bytes = serialize_program(program)?;
        Self::from_program_bytes(program, &program_bytes, symbols, sources)
    }

    pub fn from_program_bytes(
        program: &SExp,
        program_bytes: &[u8],
        symbols: &HashMap<String, String>,
        sources: &HashMap<String, String>,
    ) -> Result<Self, String> {
        if serialize_program(program)? != program_bytes {
            return Err("debug metadata program does not match exact serialized bytes".to_string());
        }
        let program_index = ProgramIndex::new(program);
        Self::from_program_index(&program_index, program_bytes, symbols, sources)
    }

    fn from_program_index(
        program_index: &ProgramIndex<'_>,
        program_bytes: &[u8],
        symbols: &HashMap<String, String>,
        sources: &HashMap<String, String>,
    ) -> Result<Self, String> {
        let mut state = InternState::default();
        let (functions, functions_by_hash) = build_functions(program_index, symbols, &mut state)?;
        let tree = build_tree(
            &program_index.root,
            symbols,
            &functions_by_hash,
            sources,
            &mut state,
        )?;
        Ok(DebugMetadata {
            program_sha256: Sha256::digest(program_bytes).into(),
            files: state.files,
            strings: state.strings,
            spans: state.spans,
            functions,
            tree,
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let mut allocator = Allocator::new();
        let root = self.to_clvm(&mut allocator)?;
        let mut stream = Stream::new(None);
        sexp_to_stream(&mut allocator, root, &mut stream);
        Ok(stream.get_value().data().clone())
    }

    pub fn decode(data: &[u8]) -> Result<Self, String> {
        let mut allocator = Allocator::new();
        let mut stream = Stream::new(Some(Bytes::new(Some(BytesFromType::Raw(data.to_vec())))));
        let root = sexp_from_stream(
            &mut allocator,
            &mut stream,
            Box::new(SimpleCreateCLVMObject {}),
        )
        .map_err(|error| format!("invalid debug metadata CLVM: {error:?}"))?
        .1;
        if stream.get_seek() != stream.get_length() {
            return Err("trailing bytes after debug metadata".to_string());
        }
        Self::from_clvm(&allocator, root)
    }

    pub fn verify_program(&self, program_bytes: &[u8]) -> Result<(), String> {
        let actual_hash: [u8; 32] = Sha256::digest(program_bytes).into();
        if actual_hash != self.program_sha256 {
            return Err("debug metadata program identity mismatch".to_string());
        }
        let mut allocator = Allocator::new();
        let mut stream = Stream::new(Some(Bytes::new(Some(BytesFromType::Raw(
            program_bytes.to_vec(),
        )))));
        let program = sexp_from_stream(
            &mut allocator,
            &mut stream,
            Box::new(SimpleCreateCLVMObject {}),
        )
        .map_err(|error| format!("invalid program CLVM: {error:?}"))?
        .1;
        if stream.get_seek() != stream.get_length() {
            return Err("trailing bytes after program".to_string());
        }
        verify_tree(&allocator, program, &self.tree)
    }

    /// Symbolize by exact structure, then by recursively peeling canonical
    /// curry wrappers. `runtime_arguments` are the top-level values from the
    /// CLVM environment; compiler-internal left environments are removed using
    /// the function record. Runtime values remain distinct from values quoted
    /// into curry.
    pub fn symbolize_frame(
        &self,
        program_bytes: &[u8],
        runtime_arguments: &[Vec<u8>],
    ) -> Result<SymbolizedFrame, String> {
        self.symbolize_frame_with_environment(program_bytes, runtime_arguments, None)
    }

    fn symbolize_frame_with_environment(
        &self,
        program_bytes: &[u8],
        runtime_arguments: &[Vec<u8>],
        runtime_tail: Option<&[u8]>,
    ) -> Result<SymbolizedFrame, String> {
        let mut allocator = Allocator::new();
        let program = decode_clvm(&mut allocator, program_bytes, "frame program")?;
        let mut bound_arguments = Vec::new();
        let (matched, fallback_function, source_span, function_index) = symbolize_node(
            &allocator,
            program,
            &self.tree,
            &self.strings,
            &mut bound_arguments,
        );
        let function = function_index
            .and_then(|index| self.functions.get(index))
            .and_then(|function| self.strings.get(function.name))
            .cloned()
            .or(fallback_function);
        let arguments = function_index
            .map(|index| {
                materialize_frame_arguments(
                    self,
                    index,
                    &bound_arguments,
                    runtime_arguments,
                    runtime_tail,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let program_hash = clvm_tree_hash(&mut allocator, program)?;
        let mut serialized_runtime_arguments = runtime_arguments.to_vec();
        serialized_runtime_arguments.extend(runtime_tail.map(<[u8]>::to_vec));
        Ok(SymbolizedFrame {
            matched,
            function,
            function_index,
            source_span,
            program_hash,
            bound_arguments,
            runtime_arguments: serialized_runtime_arguments,
            arguments,
        })
    }

    fn to_clvm(&self, allocator: &mut Allocator) -> Result<NodePtr, String> {
        let files = self
            .files
            .iter()
            .map(|file| {
                let path = atom(allocator, file.path.as_bytes())?;
                let source = atom(allocator, file.source.as_bytes())?;
                list(allocator, &[path, source])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let strings = self
            .strings
            .iter()
            .map(|value| atom(allocator, value.as_bytes()))
            .collect::<Result<Vec<_>, _>>()?;
        let spans = self
            .spans
            .iter()
            .map(|span| {
                let file = uint(allocator, span.file)?;
                let start_line = uint(allocator, span.start_line)?;
                let start_column = uint(allocator, span.start_column)?;
                let end_line = uint(allocator, span.end_line)?;
                let end_column = uint(allocator, span.end_column)?;
                list(
                    allocator,
                    &[file, start_line, start_column, end_line, end_column],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let functions = self
            .functions
            .iter()
            .map(|function| encode_function(allocator, function))
            .collect::<Result<Vec<_>, _>>()?;
        let coordinate_base = uint(allocator, 1)?;
        let tab_width = uint(allocator, DEBUG_METADATA_TAB_WIDTH)?;
        let coordinates = list(allocator, &[coordinate_base, tab_width])?;
        let files = list(allocator, &files)?;
        let strings = list(allocator, &strings)?;
        let spans = list(allocator, &spans)?;
        let functions = list(allocator, &functions)?;
        let tree = encode_tree(allocator, &self.tree)?;
        let magic = atom(allocator, MAGIC)?;
        let version = uint(allocator, DEBUG_METADATA_VERSION)?;
        let program_sha256 = atom(allocator, &self.program_sha256)?;
        list(
            allocator,
            &[
                magic,
                version,
                coordinates,
                program_sha256,
                files,
                strings,
                spans,
                functions,
                tree,
            ],
        )
    }

    fn from_clvm(allocator: &Allocator, root: NodePtr) -> Result<Self, String> {
        let root = proper_list(allocator, root)?;
        if root.len() != 9 || atom_value(allocator, root[0])? != MAGIC {
            return Err("not Chialisp structural debug metadata".to_string());
        }
        let version = decode_uint(&atom_value(allocator, root[1])?)?;
        if version != DEBUG_METADATA_VERSION {
            return Err(format!("unsupported debug metadata version {version}"));
        }
        let coordinates = proper_list(allocator, root[2])?;
        if coordinates.len() != 2
            || decode_uint(&atom_value(allocator, coordinates[0])?)? != 1
            || decode_uint(&atom_value(allocator, coordinates[1])?)? != DEBUG_METADATA_TAB_WIDTH
        {
            return Err("unsupported debug metadata coordinate convention".to_string());
        }
        let hash = atom_value(allocator, root[3])?;
        let program_sha256: [u8; 32] = hash
            .try_into()
            .map_err(|_| "program identity must be a 32-byte SHA-256".to_string())?;

        let files = proper_list(allocator, root[4])?
            .into_iter()
            .map(|entry| {
                let entry = proper_list(allocator, entry)?;
                if entry.len() != 2 {
                    return Err("malformed debug source file entry".to_string());
                }
                Ok(DebugSourceFile {
                    path: utf8(&atom_value(allocator, entry[0])?, "source path")?,
                    source: utf8(&atom_value(allocator, entry[1])?, "source content")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let strings = proper_list(allocator, root[5])?
            .into_iter()
            .map(|value| utf8(&atom_value(allocator, value)?, "debug string"))
            .collect::<Result<Vec<_>, _>>()?;
        let spans = proper_list(allocator, root[6])?
            .into_iter()
            .map(|entry| {
                let entry = proper_list(allocator, entry)?;
                if entry.len() != 5 {
                    return Err("malformed debug span entry".to_string());
                }
                let span = DebugSourceSpan {
                    file: decode_uint(&atom_value(allocator, entry[0])?)?,
                    start_line: decode_uint(&atom_value(allocator, entry[1])?)?,
                    start_column: decode_uint(&atom_value(allocator, entry[2])?)?,
                    end_line: decode_uint(&atom_value(allocator, entry[3])?)?,
                    end_column: decode_uint(&atom_value(allocator, entry[4])?)?,
                };
                if span.file >= files.len()
                    || span.start_line == 0
                    || span.start_column == 0
                    || span.end_line == 0
                    || span.end_column == 0
                {
                    return Err("debug span has an invalid table index or coordinate".to_string());
                }
                Ok(span)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let functions = proper_list(allocator, root[7])?
            .into_iter()
            .map(|node| decode_function(allocator, node, strings.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let tree = decode_tree(
            allocator,
            root[8],
            spans.len(),
            strings.len(),
            functions.len(),
        )?;
        Ok(DebugMetadata {
            program_sha256,
            files,
            strings,
            spans,
            functions,
            tree,
        })
    }
}

impl DebugMetadataCollection {
    pub fn insert(&mut self, sidecar: &[u8]) -> Result<(), String> {
        self.insert_metadata(DebugMetadata::decode(sidecar)?)
    }

    pub fn insert_metadata(&mut self, metadata: DebugMetadata) -> Result<(), String> {
        if self
            .program_identities
            .contains_key(&metadata.program_sha256)
        {
            return Err(format!(
                "duplicate debug metadata program identity {}",
                hex::encode(metadata.program_sha256)
            ));
        }

        let index = self.entries.len();
        let mut identities = HashSet::new();
        collect_debug_node_identities(&metadata.tree, &mut identities);
        for identity in identities {
            self.structural_identities
                .entry(identity)
                .or_default()
                .push(index);
        }
        self.program_identities
            .insert(metadata.program_sha256, index);
        self.entries.push(metadata);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&self, index: usize) -> Option<&DebugMetadata> {
        self.entries.get(index)
    }

    /// Select a sidecar and symbolize a serialized evaluator frame. A
    /// nonmatching frame is returned as unknown; malformed data for a frame
    /// that matches an indexed sidecar remains an error.
    pub fn symbolize_serialized_frame(
        &self,
        captured: &SerializedFrame,
    ) -> Result<SymbolizedSerializedFrame, String> {
        let mut allocator = Allocator::new();
        let program = decode_clvm(&mut allocator, &captured.program, "frame program")?;
        let (runtime_arguments, runtime_tail) =
            decode_serialized_environment(&captured.environment)?;

        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        let serialized_identity: [u8; 32] = Sha256::digest(&captured.program).into();
        if let Some(index) = self.program_identities.get(&serialized_identity) {
            let metadata = &self.entries[*index];
            let frame = metadata.symbolize_frame_with_environment(
                &captured.program,
                &runtime_arguments,
                runtime_tail.as_deref(),
            )?;
            if frame.matched == FrameMatch::Unknown {
                return Err("debug metadata program structure mismatch".to_string());
            }
            return Ok(SymbolizedSerializedFrame {
                metadata_index: Some(*index),
                frame,
            });
        }

        let mut current = program;
        loop {
            let identity = clvm_tree_hash(&mut allocator, current)?;
            if let Some(indices) = self.structural_identities.get(&identity) {
                for index in indices {
                    if seen.insert(*index) {
                        candidates.push(*index);
                    }
                }
            }
            let Some((base, _)) = canonical_curry(&allocator, current) else {
                break;
            };
            current = base;
        }

        let mut matched_error = None;
        for index in candidates {
            let metadata = &self.entries[index];
            match metadata.symbolize_frame_with_environment(
                &captured.program,
                &runtime_arguments,
                runtime_tail.as_deref(),
            ) {
                Ok(frame) if frame.matched != FrameMatch::Unknown => {
                    return Ok(SymbolizedSerializedFrame {
                        metadata_index: Some(index),
                        frame,
                    });
                }
                Ok(_) => {}
                Err(error) => {
                    matched_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = matched_error {
            return Err(error);
        }

        Ok(SymbolizedSerializedFrame {
            metadata_index: None,
            frame: SymbolizedFrame {
                matched: FrameMatch::Unknown,
                function: None,
                function_index: None,
                source_span: None,
                program_hash: clvm_tree_hash(&mut allocator, program)?,
                bound_arguments: Vec::new(),
                runtime_arguments: runtime_arguments.into_iter().chain(runtime_tail).collect(),
                arguments: Vec::new(),
            },
        })
    }

    /// Format retained frames in evaluator order (oldest to newest). The
    /// omitted count describes older frames dropped before the retained slice.
    pub fn format_captured_stack(
        &self,
        frames: &[SerializedFrame],
        omitted: usize,
        style: StackFrameStyle,
    ) -> Result<String, String> {
        let mut rendered = Vec::with_capacity(frames.len() + usize::from(omitted != 0));
        if omitted != 0 {
            rendered.push(format!(
                "<... {omitted} older frame{} omitted ...>",
                if omitted == 1 { "" } else { "s" }
            ));
        }
        for captured in frames {
            let symbolized = self.symbolize_serialized_frame(captured)?;
            let frame = if let Some(index) = symbolized.metadata_index {
                format_stack_frame(&self.entries[index], &symbolized.frame, style)?
            } else {
                format_unknown_stack_frame(&symbolized.frame, style)
            };
            rendered.push(frame);
        }
        Ok(rendered.join("\n"))
    }
}

pub fn debug_output_path(program_path: &str) -> String {
    let base = program_path
        .strip_suffix(".clvm.bin")
        .or_else(|| program_path.strip_suffix(".clvm.hex"))
        .or_else(|| program_path.strip_suffix(".hex"))
        .or_else(|| program_path.strip_suffix(".bin"))
        .unwrap_or(program_path);
    format!("{base}.debug.clvm.bin")
}

/// Compile modern Chialisp and return exact binary CLVM plus structural debug
/// metadata without converting the deployed program through hexadecimal text.
///
/// Module compilation returns the summary first, followed by one artifact for
/// every exported component. Ordinary compilation returns one artifact.
pub fn compile_with_debug(
    opts: Rc<dyn CompilerOpts>,
    content: &str,
) -> Result<Vec<DebugCompileArtifact>, CompileErr> {
    let (opts, sources) =
        capture_compiler_sources(opts.clone(), opts.filename(), content.to_string());
    let mut allocator = Allocator::new();
    let mut compiler_symbols = HashMap::new();
    let output = compile_file(
        &mut allocator,
        Rc::new(DefaultProgramRunner::new()),
        opts.clone(),
        content,
        &mut compiler_symbols,
    )?;
    let sources = sources.borrow();

    let mut programs = Vec::new();
    match output {
        CompilerOutput::Program(_, program) => programs.push((None, program)),
        CompilerOutput::Module(module) => {
            programs.push((None, module.summary.as_ref().clone()));
            for component in module.components {
                programs.push((
                    Some(String::from_utf8_lossy(&component.shortname).into_owned()),
                    component.content.as_ref().clone(),
                ));
            }
        }
    }

    programs
        .into_iter()
        .map(|(export_name, program)| {
            let program_bytes =
                serialize_program(&program).map_err(|error| CompileErr(program.loc(), error))?;
            let program_index = ProgramIndex::new(&program);
            let mut symbols = HashMap::new();
            program_index.build_symbol_table(&mut symbols);
            for (key, value) in &compiler_symbols {
                symbols.insert(key.clone(), value.clone());
            }
            let metadata = DebugMetadata::from_program_index(
                &program_index,
                &program_bytes,
                &symbols,
                &sources,
            )
            .and_then(|metadata| metadata.encode())
            .map_err(|error| CompileErr(program.loc(), error))?;
            Ok(DebugCompileArtifact {
                export_name,
                program: program_bytes,
                metadata,
                symbols,
            })
        })
        .collect()
}

fn atom(allocator: &mut Allocator, value: &[u8]) -> Result<NodePtr, String> {
    allocator
        .new_atom(value)
        .map_err(|error| format!("cannot allocate debug atom: {error:?}"))
}

fn uint(allocator: &mut Allocator, value: usize) -> Result<NodePtr, String> {
    if value == 0 {
        return atom(allocator, &[]);
    }
    let bytes = value.to_be_bytes();
    let first = bytes
        .iter()
        .position(|byte| *byte != 0)
        .unwrap_or(bytes.len() - 1);
    let mut canonical = bytes[first..].to_vec();
    if canonical[0] & 0x80 != 0 {
        canonical.insert(0, 0);
    }
    atom(allocator, &canonical)
}

fn optional_index(allocator: &mut Allocator, value: Option<usize>) -> Result<NodePtr, String> {
    uint(allocator, value.map(|index| index + 1).unwrap_or(0))
}

fn list(allocator: &mut Allocator, values: &[NodePtr]) -> Result<NodePtr, String> {
    let mut result = NodePtr::NIL;
    for value in values.iter().rev() {
        result = allocator
            .new_pair(*value, result)
            .map_err(|error| format!("cannot allocate debug list: {error:?}"))?;
    }
    Ok(result)
}

fn encode_constraint(
    allocator: &mut Allocator,
    constraint: &ParameterConstraint,
) -> Result<NodePtr, String> {
    let (tag_value, detail) = match constraint {
        ParameterConstraint::Unknown => (0, None),
        ParameterConstraint::Integer => (1, None),
        ParameterConstraint::Atom => (2, None),
        ParameterConstraint::Bytes(length) => (3, Some(uint(allocator, *length)?)),
        ParameterConstraint::Pair => (4, None),
        ParameterConstraint::ProperList => (5, None),
        ParameterConstraint::Union(values) => {
            if values.len() < 2
                || values.windows(2).any(|pair| pair[0] >= pair[1])
                || values.iter().any(|value| {
                    matches!(
                        value,
                        ParameterConstraint::Unknown | ParameterConstraint::Union(_)
                    )
                })
            {
                return Err("non-canonical parameter constraint union".to_string());
            }
            let values = values
                .iter()
                .map(|value| encode_constraint(allocator, value))
                .collect::<Result<Vec<_>, _>>()?;
            (6, Some(list(allocator, &values)?))
        }
    };
    let tag = uint(allocator, tag_value)?;
    match detail {
        Some(detail) => list(allocator, &[tag, detail]),
        None => list(allocator, &[tag]),
    }
}

fn decode_constraint(allocator: &Allocator, node: NodePtr) -> Result<ParameterConstraint, String> {
    let fields = proper_list(allocator, node)?;
    if fields.is_empty() {
        return Err("empty parameter constraint".to_string());
    }
    let tag = decode_uint(&atom_value(allocator, fields[0])?)?;
    match (tag, fields.as_slice()) {
        (0, [_]) => Ok(ParameterConstraint::Unknown),
        (1, [_]) => Ok(ParameterConstraint::Integer),
        (2, [_]) => Ok(ParameterConstraint::Atom),
        (3, [_, length]) => Ok(ParameterConstraint::Bytes(decode_uint(&atom_value(
            allocator, *length,
        )?)?)),
        (4, [_]) => Ok(ParameterConstraint::Pair),
        (5, [_]) => Ok(ParameterConstraint::ProperList),
        (6, [_, values]) => {
            let decoded = proper_list(allocator, *values)?
                .into_iter()
                .map(|value| decode_constraint(allocator, value))
                .collect::<Result<Vec<_>, _>>()?;
            if decoded.len() < 2
                || decoded.windows(2).any(|pair| pair[0] >= pair[1])
                || decoded.iter().any(|value| {
                    matches!(
                        value,
                        ParameterConstraint::Unknown | ParameterConstraint::Union(_)
                    )
                })
            {
                return Err("non-canonical parameter constraint union".to_string());
            }
            Ok(ParameterConstraint::Union(decoded))
        }
        _ => Err("unknown or malformed parameter constraint".to_string()),
    }
}

fn encode_parameter(
    allocator: &mut Allocator,
    parameter: &DebugParameter,
) -> Result<NodePtr, String> {
    match parameter {
        DebugParameter::Nil => {
            let tag = uint(allocator, 0)?;
            list(allocator, &[tag])
        }
        DebugParameter::Name {
            name,
            path,
            constraint,
        } => {
            let tag = uint(allocator, 1)?;
            let name = uint(allocator, *name)?;
            let path = atom(allocator, path)?;
            let constraint = encode_constraint(allocator, constraint)?;
            list(allocator, &[tag, name, path, constraint])
        }
        DebugParameter::Pair(left, right) => {
            let tag = uint(allocator, 2)?;
            let left = encode_parameter(allocator, left)?;
            let right = encode_parameter(allocator, right)?;
            list(allocator, &[tag, left, right])
        }
    }
}

fn decode_parameter(
    allocator: &Allocator,
    node: NodePtr,
    string_count: usize,
) -> Result<DebugParameter, String> {
    let fields = proper_list(allocator, node)?;
    if fields.is_empty() {
        return Err("empty debug parameter".to_string());
    }
    let tag = decode_uint(&atom_value(allocator, fields[0])?)?;
    match (tag, fields.as_slice()) {
        (0, [_]) => Ok(DebugParameter::Nil),
        (1, [_, name, path, constraint]) => {
            let name = decode_uint(&atom_value(allocator, *name)?)?;
            if name >= string_count {
                return Err("debug parameter name index out of range".to_string());
            }
            let path = atom_value(allocator, *path)?;
            if path.is_empty() {
                return Err("debug parameter has empty environment path".to_string());
            }
            Ok(DebugParameter::Name {
                name,
                path,
                constraint: decode_constraint(allocator, *constraint)?,
            })
        }
        (2, [_, left, right]) => Ok(DebugParameter::Pair(
            Box::new(decode_parameter(allocator, *left, string_count)?),
            Box::new(decode_parameter(allocator, *right, string_count)?),
        )),
        _ => Err("unknown or malformed debug parameter".to_string()),
    }
}

fn encode_function(allocator: &mut Allocator, function: &DebugFunction) -> Result<NodePtr, String> {
    let name = uint(allocator, function.name)?;
    let left_env = uint(allocator, usize::from(function.left_env))?;
    let parameters = encode_parameter(allocator, &function.parameters)?;
    list(allocator, &[name, left_env, parameters])
}

fn decode_function(
    allocator: &Allocator,
    node: NodePtr,
    string_count: usize,
) -> Result<DebugFunction, String> {
    let fields = proper_list(allocator, node)?;
    if fields.len() != 3 {
        return Err("malformed debug function record".to_string());
    }
    let name = decode_uint(&atom_value(allocator, fields[0])?)?;
    if name >= string_count {
        return Err("debug function name index out of range".to_string());
    }
    let left_env = decode_uint(&atom_value(allocator, fields[1])?)?;
    if left_env > 1 {
        return Err("debug function left-env flag must be boolean".to_string());
    }
    Ok(DebugFunction {
        name,
        left_env: left_env == 1,
        parameters: decode_parameter(allocator, fields[2], string_count)?,
    })
}

fn encode_tree(allocator: &mut Allocator, tree: &DebugNode) -> Result<NodePtr, String> {
    match tree {
        DebugNode::Atom {
            span,
            label,
            function,
            value,
        } => {
            let tag = uint(allocator, 0)?;
            let span = optional_index(allocator, *span)?;
            let label = optional_index(allocator, *label)?;
            let function = optional_index(allocator, *function)?;
            let value = atom(allocator, value)?;
            list(allocator, &[tag, span, label, function, value])
        }
        DebugNode::Pair {
            span,
            label,
            function,
            left,
            right,
        } => {
            let tag = uint(allocator, 1)?;
            let span = optional_index(allocator, *span)?;
            let label = optional_index(allocator, *label)?;
            let function = optional_index(allocator, *function)?;
            let left = encode_tree(allocator, left)?;
            let right = encode_tree(allocator, right)?;
            list(allocator, &[tag, span, label, function, left, right])
        }
    }
}

fn atom_value(allocator: &Allocator, node: NodePtr) -> Result<Vec<u8>, String> {
    match allocator.sexp(node) {
        clvm_rs::allocator::SExp::Atom => Ok(allocator.atom(node).as_ref().to_vec()),
        clvm_rs::allocator::SExp::Pair(_, _) => Err("expected debug metadata atom".to_string()),
    }
}

fn proper_list(allocator: &Allocator, mut node: NodePtr) -> Result<Vec<NodePtr>, String> {
    let mut result = Vec::new();
    loop {
        match allocator.sexp(node) {
            clvm_rs::allocator::SExp::Atom if allocator.atom(node).as_ref().is_empty() => {
                return Ok(result);
            }
            clvm_rs::allocator::SExp::Atom => {
                return Err("expected proper debug metadata list".to_string());
            }
            clvm_rs::allocator::SExp::Pair(left, right) => {
                result.push(left);
                node = right;
            }
        }
    }
}

fn decode_uint(bytes: &[u8]) -> Result<usize, String> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes[0] & 0x80 != 0 || (bytes.len() > 1 && bytes[0] == 0 && bytes[1] & 0x80 == 0) {
        return Err("non-canonical or negative debug metadata integer".to_string());
    }
    let mut result = 0usize;
    for byte in bytes {
        result = result
            .checked_mul(256)
            .and_then(|value| value.checked_add(*byte as usize))
            .ok_or_else(|| "debug metadata integer overflow".to_string())?;
    }
    Ok(result)
}

fn decode_optional_index(bytes: &[u8], len: usize) -> Result<Option<usize>, String> {
    let encoded = decode_uint(bytes)?;
    if encoded == 0 {
        return Ok(None);
    }
    let index = encoded - 1;
    if index >= len {
        return Err("debug metadata table index out of range".to_string());
    }
    Ok(Some(index))
}

fn utf8(bytes: &[u8], what: &str) -> Result<String, String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| format!("{what} is not valid UTF-8"))
}

fn decode_tree(
    allocator: &Allocator,
    node: NodePtr,
    span_count: usize,
    string_count: usize,
    function_count: usize,
) -> Result<DebugNode, String> {
    let fields = proper_list(allocator, node)?;
    if fields.is_empty() {
        return Err("empty debug shadow node".to_string());
    }
    let tag = decode_uint(&atom_value(allocator, fields[0])?)?;
    match tag {
        0 if fields.len() == 5 => Ok(DebugNode::Atom {
            span: decode_optional_index(&atom_value(allocator, fields[1])?, span_count)?,
            label: decode_optional_index(&atom_value(allocator, fields[2])?, string_count)?,
            function: decode_optional_index(&atom_value(allocator, fields[3])?, function_count)?,
            value: atom_value(allocator, fields[4])?,
        }),
        1 if fields.len() == 6 => Ok(DebugNode::Pair {
            span: decode_optional_index(&atom_value(allocator, fields[1])?, span_count)?,
            label: decode_optional_index(&atom_value(allocator, fields[2])?, string_count)?,
            function: decode_optional_index(&atom_value(allocator, fields[3])?, function_count)?,
            left: Box::new(decode_tree(
                allocator,
                fields[4],
                span_count,
                string_count,
                function_count,
            )?),
            right: Box::new(decode_tree(
                allocator,
                fields[5],
                span_count,
                string_count,
                function_count,
            )?),
        }),
        _ => Err("unknown or malformed debug shadow node".to_string()),
    }
}

fn verify_tree(allocator: &Allocator, program: NodePtr, tree: &DebugNode) -> Result<(), String> {
    match (allocator.sexp(program), tree) {
        (
            clvm_rs::allocator::SExp::Atom,
            DebugNode::Atom {
                value: expected, ..
            },
        ) if allocator.atom(program).as_ref() == expected => Ok(()),
        (
            clvm_rs::allocator::SExp::Pair(left, right),
            DebugNode::Pair {
                left: expected_left,
                right: expected_right,
                ..
            },
        ) => {
            verify_tree(allocator, left, expected_left)?;
            verify_tree(allocator, right, expected_right)
        }
        _ => Err("debug shadow tree does not match program structure".to_string()),
    }
}

fn decode_clvm(
    allocator: &mut Allocator,
    bytes: &[u8],
    description: &str,
) -> Result<NodePtr, String> {
    let mut stream = Stream::new(Some(Bytes::new(Some(BytesFromType::Raw(bytes.to_vec())))));
    let node = sexp_from_stream(allocator, &mut stream, Box::new(SimpleCreateCLVMObject {}))
        .map_err(|error| format!("invalid {description} CLVM: {error:?}"))?
        .1;
    if stream.get_seek() != stream.get_length() {
        return Err(format!("trailing bytes after {description}"));
    }
    Ok(node)
}

fn clvm_tree_hash(allocator: &mut Allocator, node: NodePtr) -> Result<[u8; 32], String> {
    let hash = crate::classic::clvm_tools::sha256tree::sha256tree(allocator, node);
    hash.data()
        .as_slice()
        .try_into()
        .map_err(|_| "CLVM tree hash was not 32 bytes".to_string())
}

fn collect_debug_node_identities(node: &DebugNode, identities: &mut HashSet<[u8; 32]>) -> [u8; 32] {
    let identity = match node {
        DebugNode::Atom { value, .. } => {
            let mut hasher = Sha256::new();
            hasher.update([1]);
            hasher.update(value);
            hasher.finalize().into()
        }
        DebugNode::Pair { left, right, .. } => {
            let left = collect_debug_node_identities(left, identities);
            let right = collect_debug_node_identities(right, identities);
            let mut hasher = Sha256::new();
            hasher.update([2]);
            hasher.update(left);
            hasher.update(right);
            hasher.finalize().into()
        }
    };
    identities.insert(identity);
    identity
}

fn decode_serialized_environment(
    environment: &[u8],
) -> Result<(Vec<Vec<u8>>, Option<Vec<u8>>), String> {
    let mut allocator = Allocator::new();
    let mut cursor = decode_clvm(&mut allocator, environment, "frame environment")?;
    let mut arguments = Vec::new();
    loop {
        match allocator.sexp(cursor) {
            clvm_rs::allocator::SExp::Pair(first, rest) => {
                arguments.push(
                    node_to_bytes(&allocator, first)
                        .map_err(|error| format!("cannot serialize frame argument: {error:?}"))?,
                );
                cursor = rest;
            }
            clvm_rs::allocator::SExp::Atom if allocator.atom(cursor).is_empty() => {
                return Ok((arguments, None));
            }
            clvm_rs::allocator::SExp::Atom => {
                let tail = node_to_bytes(&allocator, cursor)
                    .map_err(|error| format!("cannot serialize frame argument: {error:?}"))?;
                return Ok((arguments, Some(tail)));
            }
        }
    }
}

fn node_matches_tree(allocator: &Allocator, node: NodePtr, tree: &DebugNode) -> bool {
    match (allocator.sexp(node), tree) {
        (
            clvm_rs::allocator::SExp::Atom,
            DebugNode::Atom {
                value: expected, ..
            },
        ) => allocator.atom(node).as_ref() == expected,
        (
            clvm_rs::allocator::SExp::Pair(left, right),
            DebugNode::Pair {
                left: expected_left,
                right: expected_right,
                ..
            },
        ) => {
            node_matches_tree(allocator, left, expected_left)
                && node_matches_tree(allocator, right, expected_right)
        }
        _ => false,
    }
}

fn exact_tree_match<'a>(
    allocator: &Allocator,
    node: NodePtr,
    tree: &'a DebugNode,
) -> Option<&'a DebugNode> {
    if node_matches_tree(allocator, node, tree) {
        return Some(tree);
    }
    if let DebugNode::Pair { left, right, .. } = tree {
        exact_tree_match(allocator, node, left).or_else(|| exact_tree_match(allocator, node, right))
    } else {
        None
    }
}

fn node_label(tree: &DebugNode, strings: &[String]) -> Option<String> {
    let label = match tree {
        DebugNode::Atom { label, .. } | DebugNode::Pair { label, .. } => *label,
    };
    label.and_then(|index| strings.get(index).cloned())
}

fn node_span(tree: &DebugNode) -> Option<usize> {
    match tree {
        DebugNode::Atom { span, .. } | DebugNode::Pair { span, .. } => *span,
    }
}

fn node_function(tree: &DebugNode) -> Option<usize> {
    match tree {
        DebugNode::Atom { function, .. } | DebugNode::Pair { function, .. } => *function,
    }
}

fn pair(allocator: &Allocator, node: NodePtr) -> Option<(NodePtr, NodePtr)> {
    match allocator.sexp(node) {
        clvm_rs::allocator::SExp::Pair(left, right) => Some((left, right)),
        clvm_rs::allocator::SExp::Atom => None,
    }
}

fn atom_is(allocator: &Allocator, node: NodePtr, expected: &[u8]) -> bool {
    matches!(allocator.sexp(node), clvm_rs::allocator::SExp::Atom)
        && allocator.atom(node).as_ref() == expected
}

fn quoted_value(allocator: &Allocator, node: NodePtr) -> Option<NodePtr> {
    pair(allocator, node).and_then(|(quote, value)| {
        if atom_is(allocator, quote, &[1]) {
            Some(value)
        } else {
            None
        }
    })
}

/// Recognize only `(a (q . MOD) (c (q . ARG) ... 1))`.
fn canonical_curry(allocator: &Allocator, node: NodePtr) -> Option<(NodePtr, Vec<NodePtr>)> {
    let (apply, tail) = pair(allocator, node)?;
    if !atom_is(allocator, apply, &[2]) {
        return None;
    }
    let (quoted_program, tail) = pair(allocator, tail)?;
    let base = quoted_value(allocator, quoted_program)?;
    let (environment, end) = pair(allocator, tail)?;
    if !atom_is(allocator, end, &[]) {
        return None;
    }

    let mut cursor = environment;
    let mut arguments = Vec::new();
    loop {
        if atom_is(allocator, cursor, &[1]) {
            return Some((base, arguments));
        }
        let (cons, tail) = pair(allocator, cursor)?;
        if !atom_is(allocator, cons, &[4]) {
            return None;
        }
        let (quoted_argument, tail) = pair(allocator, tail)?;
        arguments.push(quoted_value(allocator, quoted_argument)?);
        let (next, end) = pair(allocator, tail)?;
        if !atom_is(allocator, end, &[]) {
            return None;
        }
        cursor = next;
    }
}

fn symbolize_node(
    allocator: &Allocator,
    node: NodePtr,
    metadata_tree: &DebugNode,
    strings: &[String],
    bound_arguments: &mut Vec<Vec<u8>>,
) -> (FrameMatch, Option<String>, Option<usize>, Option<usize>) {
    if let Some(exact) = exact_tree_match(allocator, node, metadata_tree) {
        return (
            FrameMatch::Exact,
            node_label(exact, strings),
            node_span(exact),
            node_function(exact),
        );
    }
    let Some((base, arguments)) = canonical_curry(allocator, node) else {
        return (FrameMatch::Unknown, None, None, None);
    };
    let (matched, function, source_span, function_index) =
        symbolize_node(allocator, base, metadata_tree, strings, bound_arguments);
    for argument in arguments {
        let Ok(bytes) = node_to_bytes(allocator, argument) else {
            return (FrameMatch::Unknown, None, None, None);
        };
        bound_arguments.push(bytes);
    }
    (
        if matched == FrameMatch::Unknown {
            FrameMatch::Unknown
        } else {
            FrameMatch::Curried
        },
        function,
        source_span,
        function_index,
    )
}

fn top_level_parameters(
    parameter: &DebugParameter,
) -> (Vec<&DebugParameter>, Option<&DebugParameter>) {
    let mut fixed = Vec::new();
    let mut cursor = parameter;
    loop {
        match cursor {
            DebugParameter::Pair(left, right) => {
                fixed.push(left.as_ref());
                cursor = right.as_ref();
            }
            DebugParameter::Nil => return (fixed, None),
            DebugParameter::Name { .. } => return (fixed, Some(cursor)),
        }
    }
}

fn bind_parameter_value(
    metadata: &DebugMetadata,
    allocator: &Allocator,
    parameter: &DebugParameter,
    value: NodePtr,
    binding: ArgumentBinding,
    result: &mut Vec<StackFrameArgument>,
) -> Result<(), String> {
    match parameter {
        DebugParameter::Nil => Ok(()),
        DebugParameter::Name {
            name, constraint, ..
        } => {
            let name = metadata
                .strings
                .get(*name)
                .ok_or_else(|| "debug parameter name index out of range".to_string())?
                .clone();
            result.push(StackFrameArgument {
                name,
                binding,
                constraint: constraint.clone(),
                value: node_to_bytes(allocator, value)
                    .map_err(|error| format!("cannot serialize frame argument: {error:?}"))?,
            });
            Ok(())
        }
        DebugParameter::Pair(left, right) => {
            let (first, rest) = pair(allocator, value)
                .ok_or_else(|| "frame value does not match destructured parameter".to_string())?;
            bind_parameter_value(metadata, allocator, left, first, binding, result)?;
            bind_parameter_value(metadata, allocator, right, rest, binding, result)
        }
    }
}

fn materialize_frame_arguments(
    metadata: &DebugMetadata,
    function_index: usize,
    bound_arguments: &[Vec<u8>],
    runtime_arguments: &[Vec<u8>],
    runtime_tail: Option<&[u8]>,
) -> Result<Vec<StackFrameArgument>, String> {
    let function = metadata
        .functions
        .get(function_index)
        .ok_or_else(|| "frame function index out of range".to_string())?;
    let mut allocator = Allocator::new();
    let actual = bound_arguments
        .iter()
        .map(|value| (ArgumentBinding::Bound, value))
        .chain(
            runtime_arguments
                .iter()
                .map(|value| (ArgumentBinding::Runtime, value)),
        )
        .map(|(binding, value)| {
            decode_clvm(&mut allocator, value, "frame argument").map(|node| (binding, node))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let runtime_tail = runtime_tail
        .map(|value| decode_clvm(&mut allocator, value, "frame argument"))
        .transpose()?;
    // A left-environment function receives its captured lexical environment
    // before its declared parameters. It is compiler plumbing, not a formal.
    let actual = &actual[usize::from(function.left_env).min(actual.len())..];
    let (fixed, tail) = top_level_parameters(&function.parameters);
    let mut result = Vec::new();
    for (parameter, (binding, value)) in fixed.iter().zip(actual.iter()) {
        bind_parameter_value(
            metadata,
            &allocator,
            parameter,
            *value,
            *binding,
            &mut result,
        )?;
    }
    if let Some(tail) = tail {
        let remaining = &actual[fixed.len().min(actual.len())..];
        let mut value = runtime_tail.unwrap_or(NodePtr::NIL);
        for (_, item) in remaining.iter().rev() {
            value = allocator
                .new_pair(*item, value)
                .map_err(|error| format!("cannot construct rest argument: {error:?}"))?;
        }
        let binding = if remaining
            .iter()
            .all(|(binding, _)| *binding == ArgumentBinding::Bound)
            && runtime_tail.is_none()
        {
            ArgumentBinding::Bound
        } else {
            ArgumentBinding::Runtime
        };
        bind_parameter_value(metadata, &allocator, tail, value, binding, &mut result)?;
    }
    Ok(result)
}

impl ParameterConstraint {
    pub fn union(self, other: ParameterConstraint) -> ParameterConstraint {
        if self == other {
            return self;
        }
        if self == ParameterConstraint::Unknown || other == ParameterConstraint::Unknown {
            return ParameterConstraint::Unknown;
        }
        let mut members = Vec::new();
        for constraint in [self, other] {
            match constraint {
                ParameterConstraint::Union(values) => members.extend(values),
                value => members.push(value),
            }
        }
        members.sort();
        members.dedup();
        ParameterConstraint::Union(members)
    }
}

impl std::fmt::Display for ParameterConstraint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterConstraint::Integer => formatter.write_str("integer"),
            ParameterConstraint::Atom => formatter.write_str("atom"),
            ParameterConstraint::Bytes(length) => write!(formatter, "bytes[{length}]"),
            ParameterConstraint::Pair => formatter.write_str("pair"),
            ParameterConstraint::ProperList => formatter.write_str("proper-list"),
            ParameterConstraint::Unknown => formatter.write_str("unknown"),
            ParameterConstraint::Union(values) => {
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        formatter.write_str("|")?;
                    }
                    value.fmt(formatter)?;
                }
                Ok(())
            }
        }
    }
}

fn primitive_name(primitive: &[u8]) -> &[u8] {
    match primitive {
        [1] => b"q",
        [5] => b"f",
        [6] => b"r",
        [9] => b"=",
        [10] => b">s",
        [11] => b"sha256",
        [12] => b"substr",
        [13] => b"strlen",
        [14] => b"concat",
        [16] => b"+",
        [17] => b"-",
        [18] => b"*",
        [19] => b"/",
        [20] => b"divmod",
        [21] => b">",
        [22] => b"ash",
        [23] => b"lsh",
        [24] => b"logand",
        [25] => b"logior",
        [26] => b"logxor",
        [27] => b"lognot",
        [29] => b"point_add",
        [30] => b"pubkey_for_exp",
        [48] => b"coinid",
        [49] => b"g1_subtract",
        [50] => b"g1_multiply",
        [51] => b"g1_negate",
        [52] => b"g2_add",
        [53] => b"g2_subtract",
        [54] => b"g2_multiply",
        [55] => b"g2_negate",
        [58] => b"bls_pairing_identity",
        [59] => b"bls_verify",
        [60] => b"modpow",
        [61] => b"%",
        [62] => b"keccak256",
        name => name,
    }
}

/// Return only constraints guaranteed by a primitive use. Argument indices are
/// zero-based. Unsupported or semantically polymorphic positions are unknown.
pub fn primitive_parameter_constraint(
    primitive: &[u8],
    argument_index: usize,
) -> ParameterConstraint {
    let primitive = primitive_name(primitive);
    match primitive {
        b"f" | b"r" if argument_index == 0 => ParameterConstraint::Pair,
        b"=" | b">s" | b"sha256" | b"strlen" | b"concat" | b"keccak256" => {
            ParameterConstraint::Atom
        }
        b"substr" if argument_index == 0 => ParameterConstraint::Atom,
        b"substr" => ParameterConstraint::Integer,
        b"+" | b"-" | b"*" | b"/" | b"divmod" | b">" | b"ash" | b"lsh" | b"logand" | b"logior"
        | b"logxor" | b"lognot" | b"modpow" | b"%" => ParameterConstraint::Integer,
        b"pubkey_for_exp" if argument_index == 0 => ParameterConstraint::Integer,
        b"point_add" | b"g1_subtract" | b"g1_negate" => ParameterConstraint::Bytes(48),
        b"g1_multiply" if argument_index == 0 => ParameterConstraint::Bytes(48),
        b"g1_multiply" => ParameterConstraint::Integer,
        b"g2_add" | b"g2_subtract" | b"g2_negate" => ParameterConstraint::Bytes(96),
        b"g2_multiply" if argument_index == 0 => ParameterConstraint::Bytes(96),
        b"g2_multiply" => ParameterConstraint::Integer,
        b"coinid" if argument_index < 2 => ParameterConstraint::Bytes(32),
        b"coinid" if argument_index == 2 => ParameterConstraint::Integer,
        b"bls_pairing_identity" | b"bls_verify" if argument_index.is_multiple_of(2) => {
            ParameterConstraint::Bytes(48)
        }
        b"bls_pairing_identity" | b"bls_verify" => ParameterConstraint::Bytes(96),
        _ => ParameterConstraint::Unknown,
    }
}

/// Infer constraints when a parameter path is passed directly to a primitive.
/// Indirect flows and polymorphic primitive positions remain unknown.
pub fn infer_parameter_constraints(
    program: &[u8],
    parameters: &[(String, Vec<u8>)],
) -> Result<Vec<(String, ParameterConstraint)>, String> {
    let mut allocator = Allocator::new();
    let program = decode_clvm(&mut allocator, program, "constraint program")?;
    let mut inferred = vec![None; parameters.len()];
    infer_constraints_node(&allocator, program, parameters, &mut inferred);
    Ok(parameters
        .iter()
        .enumerate()
        .map(|(index, (name, _))| {
            (
                name.clone(),
                inferred[index]
                    .clone()
                    .unwrap_or(ParameterConstraint::Unknown),
            )
        })
        .collect())
}

fn infer_parameter_constraints_sexp(
    program: &SExp,
    parameters: &[(String, Vec<u8>)],
) -> Vec<(String, ParameterConstraint)> {
    let mut inferred = vec![None; parameters.len()];
    infer_constraints_sexp_node(program, parameters, &mut inferred);
    parameters
        .iter()
        .enumerate()
        .map(|(index, (name, _))| {
            (
                name.clone(),
                inferred[index]
                    .clone()
                    .unwrap_or(ParameterConstraint::Unknown),
            )
        })
        .collect()
}

fn infer_constraints_sexp_node(
    node: &SExp,
    parameters: &[(String, Vec<u8>)],
    inferred: &mut [Option<ParameterConstraint>],
) {
    let SExp::Cons(_, operator, arguments) = node else {
        return;
    };
    let operator_value = match operator.as_ref() {
        SExp::Cons(_, _, _) => {
            infer_constraints_sexp_node(operator, parameters, inferred);
            Vec::new()
        }
        value => atom_bytes(value),
    };
    if primitive_name(&operator_value) == b"q" {
        return;
    }
    let mut arguments = arguments.as_ref();
    let mut argument_index = 0usize;
    while let SExp::Cons(_, argument, rest) = arguments {
        match argument.as_ref() {
            SExp::Cons(_, _, _) => {
                infer_constraints_sexp_node(argument, parameters, inferred);
            }
            value => record_inferred_constraints(
                &operator_value,
                &atom_bytes(value),
                argument_index,
                parameters,
                inferred,
            ),
        }
        arguments = rest;
        argument_index += 1;
    }
}

fn record_inferred_constraints(
    operator: &[u8],
    argument: &[u8],
    argument_index: usize,
    parameters: &[(String, Vec<u8>)],
    inferred: &mut [Option<ParameterConstraint>],
) {
    for (parameter_index, (_, path)) in parameters.iter().enumerate() {
        if argument == path {
            let constraint = primitive_parameter_constraint(operator, argument_index);
            if constraint != ParameterConstraint::Unknown {
                inferred[parameter_index] = Some(
                    inferred[parameter_index]
                        .clone()
                        .map(|current| current.union(constraint.clone()))
                        .unwrap_or(constraint),
                );
            }
        }
    }
}

fn infer_constraints_node(
    allocator: &Allocator,
    node: NodePtr,
    parameters: &[(String, Vec<u8>)],
    inferred: &mut [Option<ParameterConstraint>],
) {
    let Some((operator, mut arguments)) = pair(allocator, node) else {
        return;
    };
    let operator_value = match allocator.sexp(operator) {
        clvm_rs::allocator::SExp::Atom => allocator.atom(operator).as_ref().to_vec(),
        clvm_rs::allocator::SExp::Pair(_, _) => {
            infer_constraints_node(allocator, operator, parameters, inferred);
            Vec::new()
        }
    };
    // Do not interpret quoted data as executable primitive use.
    if primitive_name(&operator_value) == b"q" {
        return;
    }
    let mut argument_index = 0usize;
    while let Some((argument, rest)) = pair(allocator, arguments) {
        if let clvm_rs::allocator::SExp::Atom = allocator.sexp(argument) {
            let argument_value = allocator.atom(argument);
            record_inferred_constraints(
                &operator_value,
                argument_value.as_ref(),
                argument_index,
                parameters,
                inferred,
            );
        } else {
            infer_constraints_node(allocator, argument, parameters, inferred);
        }
        arguments = rest;
        argument_index += 1;
    }
}

/// Render canonical CLVM serialization using the same disassembly convention
/// used for `brun` values.
pub fn render_clvm_value(serialized: &[u8]) -> Result<String, String> {
    let mut allocator = Allocator::new();
    let value = decode_clvm(&mut allocator, serialized, "frame value")?;
    Ok(disassemble_with_kw(
        &allocator,
        value,
        keyword_from_atom(crate::classic::clvm::OPERATORS_LATEST_VERSION),
    ))
}

fn expand_tabs(line: &str) -> String {
    let mut result = String::new();
    let mut column = 1usize;
    for character in line.chars() {
        if character == '\t' {
            let next = (column + DEBUG_METADATA_TAB_WIDTH) & !(DEBUG_METADATA_TAB_WIDTH - 1);
            result.extend(std::iter::repeat_n(' ', next - column));
            column = next;
        } else {
            result.push(character);
            column += 1;
        }
    }
    result
}

fn format_unknown_stack_frame(frame: &SymbolizedFrame, style: StackFrameStyle) -> String {
    let function = format!("<unknown:{}>", hex::encode(frame.program_hash));
    match style {
        StackFrameStyle::Lisp => format!("({function})"),
        StackFrameStyle::Python => format!("{function}()"),
    }
}

pub fn format_stack_frame(
    metadata: &DebugMetadata,
    frame: &SymbolizedFrame,
    style: StackFrameStyle,
) -> Result<String, String> {
    let function = frame
        .function
        .clone()
        .unwrap_or_else(|| format!("<unknown:{}>", hex::encode(frame.program_hash)));
    let rendered_arguments = frame
        .arguments
        .iter()
        .map(|argument| {
            render_clvm_value(&argument.value).map(|value| match style {
                StackFrameStyle::Lisp => {
                    format!("({} {} {})", argument.name, argument.constraint, value)
                }
                StackFrameStyle::Python => {
                    format!("{}: {} = {}", argument.name, argument.constraint, value)
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = match style {
        StackFrameStyle::Lisp => {
            if rendered_arguments.is_empty() {
                format!("({function})")
            } else {
                format!("({function} {})", rendered_arguments.join(" "))
            }
        }
        StackFrameStyle::Python => format!("{function}({})", rendered_arguments.join(", ")),
    };
    let bound_names = frame
        .arguments
        .iter()
        .filter(|argument| argument.binding == ArgumentBinding::Bound)
        .map(|argument| argument.name.as_str())
        .collect::<Vec<_>>();
    if !bound_names.is_empty() {
        output.push_str(match style {
            StackFrameStyle::Lisp => " ; bound: ",
            StackFrameStyle::Python => "  # bound: ",
        });
        output.push_str(&bound_names.join(", "));
    }

    if let Some(span_index) = frame.source_span {
        let span = metadata
            .spans
            .get(span_index)
            .ok_or_else(|| "frame source span index out of range".to_string())?;
        let file = metadata
            .files
            .get(span.file)
            .ok_or_else(|| "frame source file index out of range".to_string())?;
        if let Some(line) = file.source.lines().nth(span.start_line - 1) {
            let expanded = expand_tabs(line);
            let width = if span.end_line == span.start_line {
                span.end_column.saturating_sub(span.start_column).max(1)
            } else {
                expanded
                    .chars()
                    .count()
                    .saturating_sub(span.start_column - 1)
                    .max(1)
            };
            output.push_str(&format!(
                "\n  --> {}:{}:{}\n   |\n{:>3}| {}\n   | {}{}",
                file.path,
                span.start_line,
                span.start_column,
                span.start_line,
                expanded,
                " ".repeat(span.start_column - 1),
                "^".repeat(width)
            ));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::sexp::parse_sexp;

    #[derive(Clone)]
    struct InMemoryCompilerOpts {
        opts: Rc<dyn CompilerOpts>,
        files: Rc<HashMap<String, (String, Vec<u8>)>>,
    }

    impl HasCompilerOptsDelegation for InMemoryCompilerOpts {
        fn compiler_opts(&self) -> Rc<dyn CompilerOpts> {
            self.opts.clone()
        }

        fn update_compiler_opts<F: FnOnce(Rc<dyn CompilerOpts>) -> Rc<dyn CompilerOpts>>(
            &self,
            f: F,
        ) -> Rc<dyn CompilerOpts> {
            Rc::new(Self {
                opts: f(self.opts.clone()),
                files: self.files.clone(),
            })
        }

        fn override_read_new_file(
            &self,
            inc_from: String,
            filename: String,
        ) -> Result<(String, Vec<u8>), CompileErr> {
            self.files
                .get(&filename)
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| self.opts.read_new_file(inc_from, filename))
        }
    }

    fn shadow_to_node(allocator: &mut Allocator, tree: &DebugNode) -> NodePtr {
        match tree {
            DebugNode::Atom { value, .. } => allocator.new_atom(value).expect("shadow atom"),
            DebugNode::Pair { left, right, .. } => {
                let left = shadow_to_node(allocator, left);
                let right = shadow_to_node(allocator, right);
                allocator.new_pair(left, right).expect("shadow pair")
            }
        }
    }

    fn function_shadow(tree: &DebugNode, function: usize) -> Option<&DebugNode> {
        let own_function = match tree {
            DebugNode::Atom { function: own, .. } | DebugNode::Pair { function: own, .. } => *own,
        };
        if own_function == Some(function) {
            return Some(tree);
        }
        if let DebugNode::Pair { left, right, .. } = tree {
            function_shadow(left, function).or_else(|| function_shadow(right, function))
        } else {
            None
        }
    }

    fn shadow_bytes(tree: &DebugNode) -> Vec<u8> {
        let mut allocator = Allocator::new();
        let node = shadow_to_node(&mut allocator, tree);
        node_to_bytes(&allocator, node).expect("serialize shadow")
    }

    fn atom_serialization(value: &[u8]) -> Vec<u8> {
        let mut allocator = Allocator::new();
        let node = allocator.new_atom(value).expect("value atom");
        node_to_bytes(&allocator, node).expect("serialize value atom")
    }

    fn serialized_sexp(text: &str) -> Vec<u8> {
        let value = parse_sexp(Srcloc::start("*value*"), text.bytes())
            .expect("parse value")
            .remove(0);
        serialize_program(value.as_ref()).expect("serialize value")
    }

    fn serialized_environment(values: &[Vec<u8>], tail: Option<&[u8]>) -> Vec<u8> {
        let mut allocator = Allocator::new();
        let mut environment = tail
            .map(|value| decode_clvm(&mut allocator, value, "test environment tail"))
            .transpose()
            .expect("environment tail")
            .unwrap_or(NodePtr::NIL);
        for value in values.iter().rev() {
            let value =
                decode_clvm(&mut allocator, value, "test environment value").expect("value");
            environment = allocator
                .new_pair(value, environment)
                .expect("environment pair");
        }
        node_to_bytes(&allocator, environment).expect("serialize environment")
    }

    fn canonical_curry_bytes(program: &[u8], bound: &[Vec<u8>]) -> Vec<u8> {
        let mut allocator = Allocator::new();
        let program = decode_clvm(&mut allocator, program, "test curry program").expect("program");
        let mut environment = allocator.new_atom(&[1]).expect("runtime path");
        for value in bound.iter().rev() {
            let value = decode_clvm(&mut allocator, value, "test bound value").expect("value");
            let quote = allocator.new_atom(&[1]).expect("quote");
            let quoted = allocator.new_pair(quote, value).expect("quoted value");
            let cons = allocator.new_atom(&[4]).expect("cons");
            let nil = NodePtr::NIL;
            let environment_tail = allocator
                .new_pair(environment, nil)
                .expect("environment tail");
            let quoted_tail = allocator
                .new_pair(quoted, environment_tail)
                .expect("quoted tail");
            environment = allocator.new_pair(cons, quoted_tail).expect("curry cons");
        }
        let quote = allocator.new_atom(&[1]).expect("quote");
        let quoted_program = allocator.new_pair(quote, program).expect("quoted program");
        let apply = allocator.new_atom(&[2]).expect("apply");
        let nil = NodePtr::NIL;
        let environment_tail = allocator
            .new_pair(environment, nil)
            .expect("environment list");
        let program_tail = allocator
            .new_pair(quoted_program, environment_tail)
            .expect("program tail");
        let curried = allocator.new_pair(apply, program_tail).expect("curried");
        node_to_bytes(&allocator, curried).expect("serialize curry")
    }

    fn parameter_leaves<'a>(
        parameter: &'a DebugParameter,
        result: &mut Vec<(usize, &'a ParameterConstraint)>,
    ) {
        match parameter {
            DebugParameter::Nil => {}
            DebugParameter::Name {
                name, constraint, ..
            } => result.push((*name, constraint)),
            DebugParameter::Pair(left, right) => {
                parameter_leaves(left, result);
                parameter_leaves(right, result);
            }
        }
    }

    fn sexp_node_count(program: &SExp) -> usize {
        match program {
            SExp::Cons(_, left, right) => {
                1 + sexp_node_count(left.as_ref()) + sexp_node_count(right.as_ref())
            }
            _ => 1,
        }
    }

    fn fixture() -> (SExp, HashMap<String, String>, HashMap<String, String>) {
        let source = "(+ X 1)";
        let program = parse_sexp(Srcloc::start("fixture.clsp"), source.bytes())
            .expect("parse fixture")
            .remove(0);
        let program = program.as_ref().clone();
        let mut symbols = HashMap::new();
        let hash = tree_hash_hex(&program);
        symbols.insert(hash.clone(), "fixture".to_string());
        symbols.insert(format!("{hash}_arguments"), "(X Y)".to_string());
        let mut sources = HashMap::new();
        sources.insert("fixture.clsp".to_string(), source.to_string());
        (program, symbols, sources)
    }

    #[test]
    fn structural_metadata_round_trips_and_verifies_exact_program() {
        let (program, mut symbols, sources) = fixture();
        let absent_hash = "00".repeat(32);
        symbols.insert(absent_hash.clone(), "absent".to_string());
        symbols.insert(format!("{absent_hash}_arguments"), "(Z)".to_string());
        let program_bytes = serialize_program(&program).expect("serialize program");
        let metadata =
            DebugMetadata::from_program_bytes(&program, &program_bytes, &symbols, &sources)
                .expect("build metadata");
        let encoded = metadata.encode().expect("encode metadata");
        let decoded = DebugMetadata::decode(&encoded).expect("decode metadata");

        assert_eq!(decoded, metadata);
        decoded
            .verify_program(&program_bytes)
            .expect("verify exact program");
        assert_eq!(decoded.files.len(), 1);
        assert_eq!(decoded.files[0].source, "(+ X 1)");
        assert_eq!(decoded.strings, vec!["X", "Y", "fixture"]);
        assert_eq!(decoded.functions.len(), 1);
        assert!(decoded.spans.len() < 8, "spans should be interned");
    }

    #[test]
    fn program_index_hashes_once_and_uses_constant_time_symbol_lookups() {
        let (program, _, sources) = fixture();
        let program_index = ProgramIndex::new(&program);
        assert_eq!(program_index.hash_visits, sexp_node_count(&program));
        let mut legacy_symbols = HashMap::new();
        crate::compiler::debug::build_symbol_table_mut(&mut legacy_symbols, &program);
        let mut indexed_symbols = HashMap::new();
        program_index.build_symbol_table(&mut indexed_symbols);
        assert_eq!(indexed_symbols, legacy_symbols);
        let zero_program = parse_sexp(Srcloc::start("*zero*"), "(0 . 0)".bytes())
            .expect("parse zero program")
            .remove(0);
        let zero_index = ProgramIndex::new(zero_program.as_ref());
        let mut legacy_zero_symbols = HashMap::new();
        crate::compiler::debug::build_symbol_table_mut(
            &mut legacy_zero_symbols,
            zero_program.as_ref(),
        );
        let mut indexed_zero_symbols = HashMap::new();
        zero_index.build_symbol_table(&mut indexed_zero_symbols);
        assert_eq!(indexed_zero_symbols, legacy_zero_symbols);

        let mut symbols = HashMap::new();
        let present_hash = hex::encode(program_index.root.hash);
        symbols.insert(present_hash.clone(), "fixture".to_string());
        symbols.insert(format!("{present_hash}_arguments"), "(X)".to_string());
        for index in 0_u32..1_000 {
            let absent_hash = Sha256::digest(index.to_be_bytes());
            symbols.insert(hex::encode(absent_hash), format!("absent_{index}"));
        }
        let expected_lookups = symbols
            .keys()
            .filter(|key| {
                key.len() == 64
                    && key.bytes().all(|byte| byte.is_ascii_hexdigit())
                    && symbols.get(*key).is_some_and(|value| !value.contains('('))
            })
            .count();

        let mut state = InternState::default();
        let (_, functions_by_hash) =
            build_functions(&program_index, &symbols, &mut state).expect("build functions");
        assert_eq!(program_index.subtree_lookups.get(), expected_lookups);
        assert_eq!(program_index.hash_visits, sexp_node_count(&program));
        assert_eq!(functions_by_hash.len(), 1);

        build_tree(
            &program_index.root,
            &symbols,
            &functions_by_hash,
            &sources,
            &mut state,
        )
        .expect("build indexed shadow tree");
        assert_eq!(program_index.hash_visits, sexp_node_count(&program));
    }

    #[test]
    fn structural_metadata_rejects_different_program() {
        let (program, symbols, sources) = fixture();
        let metadata =
            DebugMetadata::from_program(&program, &symbols, &sources).expect("build metadata");
        let other = parse_sexp(Srcloc::start("fixture.clsp"), "(+ X 2)".bytes())
            .expect("parse other")
            .remove(0);
        let other_bytes = serialize_program(other.as_ref()).expect("serialize other");
        assert_eq!(
            metadata.verify_program(&other_bytes),
            Err("debug metadata program identity mismatch".to_string())
        );
    }

    #[test]
    fn structural_metadata_rejects_shadow_topology_mismatch_after_identity_match() {
        let (program, symbols, sources) = fixture();
        let program_bytes = serialize_program(&program).expect("serialize program");
        let mut metadata =
            DebugMetadata::from_program(&program, &symbols, &sources).expect("build metadata");
        metadata.tree = DebugNode::Atom {
            span: None,
            label: None,
            function: None,
            value: Vec::new(),
        };

        assert_eq!(
            metadata.verify_program(&program_bytes),
            Err("debug shadow tree does not match program structure".to_string())
        );
    }

    #[test]
    fn decoder_rejects_non_current_version() {
        let mut allocator = Allocator::new();
        let magic = atom(&mut allocator, MAGIC).expect("magic");
        let version = uint(&mut allocator, DEBUG_METADATA_VERSION + 1).expect("version");
        let coordinate_base = uint(&mut allocator, 1).expect("coordinate base");
        let tab_width = uint(&mut allocator, DEBUG_METADATA_TAB_WIDTH).expect("tab width");
        let coordinates = list(&mut allocator, &[coordinate_base, tab_width]).expect("coordinates");
        let hash = atom(&mut allocator, &[0; 32]).expect("hash");
        let root = list(
            &mut allocator,
            &[
                magic,
                version,
                coordinates,
                hash,
                NodePtr::NIL,
                NodePtr::NIL,
                NodePtr::NIL,
                NodePtr::NIL,
                NodePtr::NIL,
            ],
        )
        .expect("root");
        let mut stream = Stream::new(None);
        sexp_to_stream(&mut allocator, root, &mut stream);
        assert_eq!(
            DebugMetadata::decode(stream.get_value().data()),
            Err("unsupported debug metadata version 2".to_string())
        );
    }

    #[test]
    fn debug_output_names_are_deterministic() {
        assert_eq!(debug_output_path("foo.clvm.bin"), "foo.debug.clvm.bin");
        assert_eq!(debug_output_path("foo.hex"), "foo.debug.clvm.bin");
        assert_eq!(
            debug_output_path("/tmp/foo.clvm.hex"),
            "/tmp/foo.debug.clvm.bin"
        );
    }

    #[test]
    fn compiler_api_returns_exact_binary_and_metadata() {
        let source = "(mod (X) (include *standard-cl-23*) (+ X 1))";
        let opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("api.clsp"),
        );
        let artifacts = compile_with_debug(opts, source).expect("compile with debug metadata");
        assert_eq!(artifacts.len(), 1);
        assert!(!artifacts[0].program.is_empty());
        assert!(!artifacts[0].metadata.is_empty());
        DebugMetadata::decode(&artifacts[0].metadata)
            .expect("decode compiler metadata")
            .verify_program(&artifacts[0].program)
            .expect("verify compiler program");
    }

    #[test]
    fn compiler_api_captures_search_path_include_once() {
        let temp = tempfile::tempdir().expect("temp directory");
        let include_dir = temp.path().join("include");
        std::fs::create_dir(&include_dir).expect("create include directory");
        let include_source = "((defun included (X) (+ X 7)))\n";
        std::fs::write(include_dir.join("helper.clib"), include_source).expect("write include");
        let source = "(mod (X) (include *standard-cl-23*) (include helper.clib) (included X))";
        let opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("main.clsp"),
        )
        .set_search_paths(&[include_dir.to_string_lossy().into_owned()]);

        let artifact = compile_with_debug(opts, source)
            .expect("compile search-path include")
            .remove(0);
        let metadata = DebugMetadata::decode(&artifact.metadata).expect("decode metadata");
        let included = metadata
            .files
            .iter()
            .filter(|file| file.source == include_source)
            .collect::<Vec<_>>();

        assert_eq!(included.len(), 1, "included source must be interned once");
        assert_eq!(included[0].path, "helper.clib");
        assert_eq!(
            metadata
                .files
                .iter()
                .filter(|file| file.path == "helper.clib")
                .count(),
            1
        );
    }

    #[test]
    fn compiler_api_captures_in_memory_include_across_option_clones() {
        let include_source = "((defun included (X) (+ X 11)))\n";
        let files = HashMap::from([(
            "memory.clib".to_string(),
            (
                "memory://resolved/memory.clib".to_string(),
                include_source.as_bytes().to_vec(),
            ),
        )]);
        let base: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("memory-main.clsp"),
        );
        let opts: Rc<dyn CompilerOpts> = Rc::new(InMemoryCompilerOpts {
            opts: base,
            files: Rc::new(files),
        });
        let source = "(mod (X) (include *standard-cl-23*) (include memory.clib) (included X))";

        let artifact = compile_with_debug(opts, source)
            .expect("compile in-memory include")
            .remove(0);
        let metadata = DebugMetadata::decode(&artifact.metadata).expect("decode metadata");
        let included = metadata
            .files
            .iter()
            .filter(|file| file.source == include_source)
            .collect::<Vec<_>>();

        assert_eq!(included.len(), 1, "in-memory source must be interned once");
        assert_eq!(included[0].path, "memory.clib");
    }

    #[test]
    fn default_debug_compile_matches_normal_optimized_cl23_compile() {
        let source = indoc! {"
            (mod (N)
              (include *standard-cl-23*)
              (defun optimized (X)
                (+ X 0))
              (optimized N))
        "};
        let filename = "option-parity.clsp";
        let base_opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new(filename),
        );

        let mut normal_allocator = Allocator::new();
        let normal = crate::classic::clvm_tools::clvmc::compile_clvm_text_maybe_opt(
            &mut normal_allocator,
            false,
            base_opts.clone(),
            &mut HashMap::new(),
            source,
            filename,
            true,
        )
        .expect("normal compile");
        let normal_bytes =
            node_to_bytes(&normal_allocator, normal).expect("serialize normal compile");

        let mut option_allocator = Allocator::new();
        let debug_opts = crate::classic::clvm_tools::clvmc::compiler_opts_for_source(
            &mut option_allocator,
            base_opts,
            source,
            false,
        )
        .expect("derive CLI-equivalent options");
        let artifact = compile_with_debug(debug_opts, source)
            .expect("debug compile")
            .remove(0);

        assert_eq!(artifact.program, normal_bytes);
        DebugMetadata::decode(&artifact.metadata)
            .expect("decode debug metadata")
            .verify_program(&normal_bytes)
            .expect("metadata matches normal program");
    }

    #[test]
    fn compile_with_debug_preserves_explicit_optimizer_options() {
        let source = indoc! {"
            (mod (N)
              (include *standard-cl-23*)
              (defun optimized (X)
                (+ X 0))
              (optimized N))
        "};
        let dialect = KNOWN_DIALECTS["*standard-cl-23*"].accepted.clone();
        let base = Rc::new(crate::compiler::compiler::DefaultCompilerOpts::new(
            "explicit-options.clsp",
        ))
        .set_dialect(dialect);
        let unoptimized = compile_with_debug(base.set_optimize(false), source)
            .expect("explicit unoptimized compile")
            .remove(0);
        let optimized = compile_with_debug(base.set_optimize(true), source)
            .expect("explicit optimized compile")
            .remove(0);

        assert_ne!(unoptimized.program, optimized.program);
    }

    #[test]
    fn symbolizes_exact_and_recursive_curry_frames() {
        let (program, symbols, sources) = fixture();
        let metadata =
            DebugMetadata::from_program(&program, &symbols, &sources).expect("build metadata");
        let exact_bytes = serialize_program(&program).expect("serialize exact frame");
        let runtime = vec![vec![0x80]];
        let exact = metadata
            .symbolize_frame(&exact_bytes, &runtime)
            .expect("symbolize exact");
        assert_eq!(exact.matched, FrameMatch::Exact);
        assert_eq!(exact.function.as_deref(), Some("fixture"));
        assert!(exact.bound_arguments.is_empty());
        assert_eq!(exact.runtime_arguments, runtime);

        let inner_text = format!("(2 (1 . {program}) (4 (1 . 42) 1))");
        let inner = parse_sexp(Srcloc::start("curry.clvm"), inner_text.bytes())
            .expect("parse inner curry")
            .remove(0);
        let outer_text = format!("(2 (1 . {inner}) (4 (1 . 99) 1))");
        let outer = parse_sexp(Srcloc::start("curry.clvm"), outer_text.bytes())
            .expect("parse outer curry")
            .remove(0);
        let outer_bytes = serialize_program(outer.as_ref()).expect("serialize curry");
        let curried = metadata
            .symbolize_frame(&outer_bytes, &[])
            .expect("symbolize curry");
        assert_eq!(curried.matched, FrameMatch::Curried);
        assert_eq!(curried.function.as_deref(), Some("fixture"));
        assert_eq!(curried.bound_arguments, vec![vec![42], vec![99]]);
    }

    #[test]
    fn collection_selects_exact_and_recursively_curried_sidecars_in_argument_order() {
        let (program, symbols, sources) = fixture();
        let metadata =
            DebugMetadata::from_program(&program, &symbols, &sources).expect("fixture metadata");
        let program_bytes = serialize_program(&program).expect("fixture program");

        let other = parse_sexp(Srcloc::start("other.clsp"), "(* Z 2)".bytes())
            .expect("parse other")
            .remove(0);
        let other_hash = tree_hash_hex(other.as_ref());
        let other_symbols = HashMap::from([
            (other_hash.clone(), "other".to_string()),
            (format!("{other_hash}_arguments"), "(Z)".to_string()),
        ]);
        let other_sources = HashMap::from([("other.clsp".to_string(), "(* Z 2)".to_string())]);
        let other_metadata =
            DebugMetadata::from_program(other.as_ref(), &other_symbols, &other_sources)
                .expect("other metadata");

        let mut collection = DebugMetadataCollection::default();
        collection
            .insert_metadata(other_metadata)
            .expect("insert other");
        collection
            .insert_metadata(metadata.clone())
            .expect("insert fixture");
        assert_eq!(collection.len(), 2);
        assert!(!collection.is_empty());

        let exact = collection
            .symbolize_serialized_frame(&SerializedFrame {
                program: program_bytes.clone(),
                environment: serialized_sexp("(5 6)"),
            })
            .expect("exact frame");
        assert_eq!(exact.metadata_index, Some(1));
        assert_eq!(exact.frame.matched, FrameMatch::Exact);

        let inner = canonical_curry_bytes(&program_bytes, &[vec![42]]);
        let outer = canonical_curry_bytes(&inner, &[vec![99]]);
        let curried = collection
            .symbolize_serialized_frame(&SerializedFrame {
                program: outer,
                environment: serialized_sexp("()"),
            })
            .expect("recursive curry frame");
        assert_eq!(curried.metadata_index, Some(1));
        assert_eq!(curried.frame.matched, FrameMatch::Curried);
        assert_eq!(curried.frame.bound_arguments, vec![vec![42], vec![99]]);
        assert_eq!(
            curried
                .frame
                .arguments
                .iter()
                .map(|argument| (argument.name.as_str(), argument.value.as_slice()))
                .collect::<Vec<_>>(),
            vec![("X", &[42][..]), ("Y", &[99][..])]
        );

        let mut inconsistent = metadata.clone();
        let duplicate = collection.insert_metadata(metadata).unwrap_err();
        assert!(duplicate.contains("duplicate debug metadata program identity"));

        inconsistent.tree = DebugNode::Atom {
            span: None,
            label: None,
            function: None,
            value: vec![1],
        };
        let mut malformed_collection = DebugMetadataCollection::default();
        malformed_collection
            .insert_metadata(inconsistent)
            .expect("insert structurally inconsistent metadata");
        assert_eq!(
            malformed_collection
                .symbolize_serialized_frame(&SerializedFrame {
                    program: program_bytes,
                    environment: serialized_sexp("(5 6)"),
                })
                .unwrap_err(),
            "debug metadata program structure mismatch"
        );
    }

    #[test]
    fn captured_environment_preserves_left_env_and_destructured_parameters() {
        let source = indoc! {"
            (mod (MAIN)
              (include *standard-cl-23*)
              (defun destructured ((N . P) B)
                (c (+ N 1) (c (f P) (c (coinid B B N) ()))))
              (destructured MAIN MAIN))
        "};
        let opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("captured-destructured.clsp"),
        );
        let artifact = compile_with_debug(opts, source)
            .expect("compile destructured metadata")
            .remove(0);
        let metadata = DebugMetadata::decode(&artifact.metadata).expect("decode metadata");
        let function_index = metadata
            .functions
            .iter()
            .position(|function| metadata.strings[function.name] == "destructured")
            .expect("destructured function");
        assert!(metadata.functions[function_index].left_env);
        let function_program = shadow_bytes(
            function_shadow(&metadata.tree, function_index).expect("function program"),
        );
        let pair = serialized_sexp("(5 7 . 8)");
        let bytes32 = atom_serialization(&[0x22; 32]);
        let environment =
            serialized_environment(&[atom_serialization(b"captured"), pair, bytes32], None);

        let mut collection = DebugMetadataCollection::default();
        collection
            .insert(&artifact.metadata)
            .expect("insert metadata");
        let symbolized = collection
            .symbolize_serialized_frame(&SerializedFrame {
                program: function_program,
                environment,
            })
            .expect("captured frame must not fail destructured binding");
        assert_eq!(
            symbolized
                .frame
                .arguments
                .iter()
                .map(|argument| argument.name.as_str())
                .collect::<Vec<_>>(),
            vec!["N", "P", "B"]
        );
        let rendered = format_stack_frame(
            collection
                .get(symbolized.metadata_index.expect("metadata index"))
                .expect("metadata"),
            &symbolized.frame,
            StackFrameStyle::Python,
        )
        .expect("render captured frame");
        assert!(
            rendered.starts_with("destructured(N: integer = 5, P: pair = (l . 8), B: bytes[32] = ")
        );
        assert!(
            !rendered.contains("frame value does not match destructured parameter"),
            "{rendered}"
        );

        let malformed = collection
            .symbolize_serialized_frame(&SerializedFrame {
                program: shadow_bytes(
                    function_shadow(&collection.get(0).expect("metadata").tree, function_index)
                        .expect("function program"),
                ),
                environment: serialized_environment(
                    &[
                        atom_serialization(b"captured"),
                        atom_serialization(b"not-a-pair"),
                        atom_serialization(&[0x22; 32]),
                    ],
                    None,
                ),
            })
            .unwrap_err();
        assert_eq!(
            malformed,
            "frame value does not match destructured parameter"
        );
    }

    #[test]
    fn captured_environment_preserves_proper_and_improper_rest_values() {
        let source = "(c A REST)";
        let program = parse_sexp(Srcloc::start("rest.clsp"), source.bytes())
            .expect("parse rest fixture")
            .remove(0);
        let hash = tree_hash_hex(program.as_ref());
        let symbols = HashMap::from([
            (hash.clone(), "resty".to_string()),
            (format!("{hash}_arguments"), "(A . REST)".to_string()),
        ]);
        let sources = HashMap::from([("rest.clsp".to_string(), source.to_string())]);
        let metadata = DebugMetadata::from_program(program.as_ref(), &symbols, &sources)
            .expect("rest metadata");
        let program = serialize_program(program.as_ref()).expect("rest program");
        let mut collection = DebugMetadataCollection::default();
        collection
            .insert_metadata(metadata)
            .expect("insert rest metadata");

        let proper = collection
            .symbolize_serialized_frame(&SerializedFrame {
                program: program.clone(),
                environment: serialized_sexp("(42 43 44)"),
            })
            .expect("proper environment");
        assert_eq!(
            render_clvm_value(&proper.frame.arguments[1].value).unwrap(),
            "(43 44)"
        );

        let improper = collection
            .symbolize_serialized_frame(&SerializedFrame {
                program,
                environment: serialized_sexp("(42 43 . 44)"),
            })
            .expect("improper environment");
        assert_eq!(
            render_clvm_value(&improper.frame.arguments[1].value).unwrap(),
            "(43 . 44)"
        );
    }

    #[test]
    fn stack_formatting_keeps_unknown_frames_and_marks_older_omissions() {
        let (program, symbols, sources) = fixture();
        let metadata = DebugMetadata::from_program(&program, &symbols, &sources).expect("metadata");
        let known = SerializedFrame {
            program: serialize_program(&program).expect("known program"),
            environment: serialized_sexp("(5 6)"),
        };
        let unknown = SerializedFrame {
            program: serialized_sexp("(- 9 3)"),
            environment: serialized_sexp("()"),
        };
        let mut collection = DebugMetadataCollection::default();
        collection
            .insert_metadata(metadata)
            .expect("insert metadata");

        let rendered = collection
            .format_captured_stack(&[known.clone(), unknown, known], 2, StackFrameStyle::Python)
            .expect("partial stack");
        assert!(rendered.starts_with("<... 2 older frames omitted ...>\n"));
        let first = rendered.find("fixture(").expect("oldest known frame");
        let unknown = rendered.find("<unknown:").expect("unknown fallback");
        let newest = rendered.rfind("fixture(").expect("newest known frame");
        assert!(first < unknown && unknown < newest, "{rendered}");
    }

    #[test]
    fn rejects_noncanonical_curry_and_retains_unknown_hash() {
        let (program, symbols, sources) = fixture();
        let metadata =
            DebugMetadata::from_program(&program, &symbols, &sources).expect("build metadata");
        // The environment terminates in nil rather than runtime path 1.
        let text = format!("(2 (1 . {program}) (4 (1 . 42) ()))");
        let unknown = parse_sexp(Srcloc::start("unknown.clvm"), text.bytes())
            .expect("parse unknown")
            .remove(0);
        let unknown_bytes = serialize_program(unknown.as_ref()).expect("serialize unknown");
        let frame = metadata
            .symbolize_frame(&unknown_bytes, &[])
            .expect("symbolize unknown");
        assert_eq!(frame.matched, FrameMatch::Unknown);
        assert_eq!(frame.function, None);
        assert_ne!(frame.program_hash, [0; 32]);
        assert!(frame.bound_arguments.is_empty());
    }

    #[test]
    fn primitive_constraints_are_conservative_and_union_canonically() {
        assert_eq!(
            primitive_parameter_constraint(b"+", 0),
            ParameterConstraint::Integer
        );
        assert_eq!(
            primitive_parameter_constraint(&[5], 0),
            ParameterConstraint::Pair
        );
        assert_eq!(
            primitive_parameter_constraint(b"coinid", 1),
            ParameterConstraint::Bytes(32)
        );
        assert_eq!(
            primitive_parameter_constraint(b"c", 0),
            ParameterConstraint::Unknown
        );
        assert_eq!(
            ParameterConstraint::Pair.union(ParameterConstraint::Atom),
            ParameterConstraint::Union(vec![ParameterConstraint::Atom, ParameterConstraint::Pair])
        );
        assert_eq!(
            ParameterConstraint::Integer.union(ParameterConstraint::Unknown),
            ParameterConstraint::Unknown
        );
        let mut allocator = Allocator::new();
        assert!(encode_constraint(
            &mut allocator,
            &ParameterConstraint::Union(
                vec![ParameterConstraint::Pair, ParameterConstraint::Atom,]
            ),
        )
        .is_err());
        let pair =
            encode_constraint(&mut allocator, &ParameterConstraint::Pair).expect("pair constraint");
        let atom =
            encode_constraint(&mut allocator, &ParameterConstraint::Atom).expect("atom constraint");
        let values = list(&mut allocator, &[pair, atom]).expect("union values");
        let union_tag = uint(&mut allocator, 6).expect("union tag");
        let union = list(&mut allocator, &[union_tag, values]).expect("union constraint");
        assert!(decode_constraint(&allocator, union).is_err());

        let program = parse_sexp(Srcloc::start("*constraints*"), "(+ 2 (f 5))".bytes())
            .expect("parse constraint program")
            .remove(0);
        let parameters = [
            ("X".to_string(), vec![2]),
            ("Y".to_string(), vec![5]),
            ("Z".to_string(), vec![11]),
        ];
        let inferred = infer_parameter_constraints(
            &serialize_program(program.as_ref()).expect("serialize constraints"),
            &parameters,
        )
        .expect("infer constraints");
        assert_eq!(
            inferred,
            vec![
                ("X".to_string(), ParameterConstraint::Integer),
                ("Y".to_string(), ParameterConstraint::Pair),
                ("Z".to_string(), ParameterConstraint::Unknown),
            ]
        );
        assert_eq!(
            infer_parameter_constraints_sexp(program.as_ref(), &parameters),
            inferred,
            "direct SExp inference must match the public serialized API"
        );
    }

    #[test]
    fn renders_brun_values_and_conventional_frames_with_source() {
        let source = "\t(+ X 1)";
        let program = parse_sexp(Srcloc::start("tabs.clsp"), source.bytes())
            .expect("parse tab fixture")
            .remove(0);
        let program = program.as_ref().clone();
        let mut symbols = HashMap::new();
        let hash = tree_hash_hex(&program);
        symbols.insert(hash.clone(), "tabbed".to_string());
        symbols.insert(format!("{hash}_arguments"), "(X)".to_string());
        let mut sources = HashMap::new();
        sources.insert("tabs.clsp".to_string(), source.to_string());
        let metadata = DebugMetadata::from_program(&program, &symbols, &sources).expect("metadata");
        let program_bytes = serialize_program(&program).expect("program bytes");
        let frame = metadata
            .symbolize_frame(&program_bytes, &[vec![5]])
            .expect("frame");
        let python =
            format_stack_frame(&metadata, &frame, StackFrameStyle::Python).expect("python frame");
        assert!(python.starts_with("tabbed(X: unknown = 5)"));
        assert!(python.contains("--> tabs.clsp:1:8"), "{python}");
        assert!(python.contains("        (+ X 1)"));
        assert!(python.contains("        ^"));

        let lisp =
            format_stack_frame(&metadata, &frame, StackFrameStyle::Lisp).expect("lisp frame");
        assert!(lisp.starts_with("(tabbed (X unknown 5))"));

        let value = parse_sexp(Srcloc::start("*value*"), "(hello . 5)".bytes())
            .expect("parse value")
            .remove(0);
        assert_eq!(
            render_clvm_value(&serialize_program(value.as_ref()).expect("serialize value"))
                .expect("render value"),
            "(\"hello\" . 5)"
        );
    }

    #[test]
    fn utf8_spans_and_excerpts_use_display_columns_with_tabs() {
        let source = "\t(\"é\" 参数)";
        let program = parse_sexp(Srcloc::start("utf8.clsp"), source.bytes())
            .expect("parse UTF-8 fixture")
            .remove(0);
        let program = program.as_ref().clone();
        let hash = tree_hash_hex(&program);
        let mut symbols = HashMap::new();
        symbols.insert(hash.clone(), "fünc".to_string());
        symbols.insert(format!("{hash}_arguments"), "(参数)".to_string());
        let mut sources = HashMap::new();
        sources.insert("utf8.clsp".to_string(), source.to_string());
        let metadata = DebugMetadata::from_program(&program, &symbols, &sources).expect("metadata");
        let frame = metadata
            .symbolize_frame(
                &serialize_program(&program).expect("program bytes"),
                &[atom_serialization("值".as_bytes())],
            )
            .expect("frame");
        let span = &metadata.spans[frame.source_span.expect("source span")];
        assert_eq!(
            (span.start_line, span.start_column),
            (1, 8),
            "tab convention remains one-based with eight-column stops"
        );
        assert_eq!(
            (span.end_line, span.end_column),
            (1, 12),
            "non-ASCII characters occupy one display column"
        );
        assert!(
            metadata.spans.iter().any(|span| {
                span.start_line == 1
                    && span.start_column == 13
                    && span.end_line == 1
                    && span.end_column == 15
            }),
            "the two-character non-ASCII parameter should occupy columns 13 through 15"
        );

        let rendered =
            format_stack_frame(&metadata, &frame, StackFrameStyle::Python).expect("render frame");
        assert!(
            rendered.starts_with("fünc(参数: unknown = 0xe580bc)"),
            "{rendered}"
        );
        assert!(rendered.contains("--> utf8.clsp:1:8"), "{rendered}");
        assert!(rendered.contains("        (\"é\" 参数)"), "{rendered}");
        assert!(rendered.contains("       ^^^^"), "{rendered}");
    }

    #[test]
    fn many_line_span_indexing_does_not_rescan_source_prefixes() {
        const LINE_COUNT: usize = 20_000;
        let source = "éX\n".repeat(LINE_COUNT);
        let path = Rc::new("many-lines.clsp".to_string());
        let mut sources = HashMap::new();
        sources.insert(path.as_ref().clone(), source);
        let mut state = InternState::default();

        for line in 1..=LINE_COUNT {
            state
                .intern_span(&Srcloc::new(path.clone(), line, 3), &sources)
                .expect("intern span");
        }

        assert_eq!(state.source_indices[0].line_starts.len(), LINE_COUNT + 1);
        assert_eq!(state.spans.len(), LINE_COUNT);
        assert_eq!(
            state.spans.last(),
            Some(&DebugSourceSpan {
                file: 0,
                start_line: LINE_COUNT,
                start_column: 2,
                end_line: LINE_COUNT,
                end_column: 3,
            })
        );
    }

    #[test]
    fn compiler_metadata_interns_files_sources_strings_and_spans() {
        let source = indoc! {"
            (mod (X)
              (include *standard-cl-23*)
              (defun first (X) (+ X X))
              (defun second (X) (+ X X))
              (+ (first X) (second X)))
        "};
        let opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("dedup.clsp"),
        );
        let artifact = compile_with_debug(opts, source)
            .expect("compile dedup fixture")
            .remove(0);
        let metadata = DebugMetadata::decode(&artifact.metadata).expect("decode metadata");

        assert!(
            metadata.files.len() >= 2,
            "fixture should retain both user and included macro sources"
        );
        assert_eq!(
            metadata
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            metadata.files.len(),
            "file/source records must be interned"
        );
        assert_eq!(
            metadata
                .strings
                .iter()
                .map(String::as_str)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            metadata.strings.len(),
            "repeated parameter and symbol names must be interned"
        );
        assert_eq!(
            metadata
                .spans
                .iter()
                .map(|span| (
                    span.file,
                    span.start_line,
                    span.start_column,
                    span.end_line,
                    span.end_column,
                ))
                .collect::<std::collections::HashSet<_>>()
                .len(),
            metadata.spans.len(),
            "repeated source spans must be interned"
        );
        assert_eq!(
            metadata
                .strings
                .iter()
                .filter(|value| value.as_str() == "X")
                .count(),
            1
        );
    }

    #[test]
    fn destructured_parameters_round_trip_bind_infer_and_render() {
        let source = indoc! {"
            (mod (MAIN)
              (include *standard-cl-23*)
              (defun destructured ((N . P) B)
                (c (+ N 1) (c (f P) (c (coinid B B N) ()))))
              (destructured MAIN MAIN))
        "};
        let opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("destructured.clsp"),
        );
        let artifact = compile_with_debug(opts, source)
            .expect("compile destructured metadata")
            .remove(0);
        let metadata = DebugMetadata::decode(&artifact.metadata).expect("decode metadata");
        assert_eq!(
            DebugMetadata::decode(&metadata.encode().expect("re-encode metadata"))
                .expect("decode re-encoded metadata"),
            metadata
        );
        let function_index = metadata
            .functions
            .iter()
            .position(|function| metadata.strings[function.name] == "destructured")
            .expect("destructured function record");
        let mut leaves = Vec::new();
        parameter_leaves(&metadata.functions[function_index].parameters, &mut leaves);
        assert_eq!(
            leaves
                .iter()
                .map(|(name, constraint)| (metadata.strings[*name].as_str(), (*constraint).clone()))
                .collect::<Vec<_>>(),
            vec![
                ("N", ParameterConstraint::Integer),
                ("P", ParameterConstraint::Pair),
                ("B", ParameterConstraint::Bytes(32)),
            ]
        );

        let program = shadow_bytes(
            function_shadow(&metadata.tree, function_index).expect("destructured structural node"),
        );
        let pair = serialize_program(
            parse_sexp(Srcloc::start("*value*"), "(5 7 . 8)".bytes())
                .expect("destructured value")
                .remove(0)
                .as_ref(),
        )
        .expect("serialize destructured value");
        let frame = metadata
            .symbolize_frame(
                &program,
                &[
                    atom_serialization(b"captured"),
                    pair,
                    atom_serialization(&[0x22; 32]),
                ],
            )
            .expect("symbolize destructured frame");
        assert_eq!(
            frame
                .arguments
                .iter()
                .map(|argument| (
                    argument.name.as_str(),
                    argument.binding,
                    argument.constraint.clone(),
                ))
                .collect::<Vec<_>>(),
            vec![
                ("N", ArgumentBinding::Runtime, ParameterConstraint::Integer),
                ("P", ArgumentBinding::Runtime, ParameterConstraint::Pair),
                (
                    "B",
                    ArgumentBinding::Runtime,
                    ParameterConstraint::Bytes(32),
                ),
            ]
        );
        let rendered =
            format_stack_frame(&metadata, &frame, StackFrameStyle::Python).expect("render");
        assert!(
            rendered.starts_with("destructured(N: integer = 5, P: pair = (l . 8), B: bytes[32] = "),
            "{rendered}"
        );
    }

    #[test]
    fn optimized_function_retains_user_source_location() {
        let source = indoc! {"
            (mod (N)
              (include *standard-cl-23*)
              (defun optimized (X)
                (+ X 0))
              (optimized N))
        "};
        let base = crate::compiler::compiler::DefaultCompilerOpts::new("optimized.clsp");
        let unoptimized = compile_with_debug(Rc::new(base.clone()), source)
            .expect("compile unoptimized fixture")
            .remove(0);
        let optimized = compile_with_debug(Rc::new(base).set_optimize(true), source)
            .expect("compile optimized fixture")
            .remove(0);
        assert_ne!(
            optimized.program, unoptimized.program,
            "fixture must exercise a final-tree optimization"
        );

        let metadata = DebugMetadata::decode(&optimized.metadata).expect("decode metadata");
        let function_index = metadata
            .functions
            .iter()
            .position(|function| metadata.strings[function.name] == "optimized")
            .expect("optimized function record");
        let function_program = shadow_bytes(
            function_shadow(&metadata.tree, function_index).expect("optimized structural node"),
        );
        let frame = metadata
            .symbolize_frame(
                &function_program,
                &[atom_serialization(b"captured"), vec![5]],
            )
            .expect("symbolize optimized function");
        let rendered =
            format_stack_frame(&metadata, &frame, StackFrameStyle::Python).expect("render");
        assert!(rendered.contains("--> optimized.clsp:4:"), "{rendered}");
        assert!(rendered.contains("    (+ X 0))"), "{rendered}");
    }

    #[test]
    fn compiled_sidecar_owns_typed_exact_and_curried_frame_arguments() {
        let source = indoc! {"
            (mod (MAIN)
              (include *standard-cl-23*)
              (defun typed (N P B U)
                (c (+ N 1) (c (f P) (c (coinid B B N) (c U ())))))
              (defun wrapper (N P B U) (typed N P B U))
              (wrapper MAIN MAIN MAIN MAIN))
        "};
        let opts: Rc<dyn CompilerOpts> = Rc::new(
            crate::compiler::compiler::DefaultCompilerOpts::new("typed.clsp"),
        );
        let artifact = compile_with_debug(opts, source)
            .expect("compile typed debug metadata")
            .remove(0);
        let metadata = DebugMetadata::decode(&artifact.metadata).expect("decode typed metadata");
        metadata
            .verify_program(&artifact.program)
            .expect("verify typed program");

        let typed_index = metadata
            .functions
            .iter()
            .position(|function| metadata.strings[function.name] == "typed")
            .expect("typed function record");
        let typed = &metadata.functions[typed_index];
        assert!(typed.left_env);
        let mut leaves = Vec::new();
        parameter_leaves(&typed.parameters, &mut leaves);
        let typed_program = shadow_bytes(
            function_shadow(&metadata.tree, typed_index).expect("typed structural node"),
        );
        assert_eq!(
            leaves
                .iter()
                .map(|(name, constraint)| (metadata.strings[*name].as_str(), (*constraint).clone()))
                .collect::<Vec<_>>(),
            vec![
                ("N", ParameterConstraint::Integer),
                ("P", ParameterConstraint::Pair),
                ("B", ParameterConstraint::Bytes(32)),
                ("U", ParameterConstraint::Unknown),
            ]
        );

        let pair_value = serialize_program(
            parse_sexp(Srcloc::start("*value*"), "(7 . 8)".bytes())
                .expect("pair value")
                .remove(0)
                .as_ref(),
        )
        .expect("serialize pair");
        let bytes32 = atom_serialization(&[0x11; 32]);
        let unknown = atom_serialization(b"opaque");
        let left_env = atom_serialization(b"captured");
        let exact = metadata
            .symbolize_frame(
                &typed_program,
                &[
                    left_env.clone(),
                    vec![5],
                    pair_value.clone(),
                    bytes32.clone(),
                    unknown.clone(),
                ],
            )
            .expect("symbolize exact typed function");
        assert_eq!(exact.matched, FrameMatch::Exact);
        assert_eq!(exact.function.as_deref(), Some("typed"));
        assert_eq!(
            exact
                .arguments
                .iter()
                .map(|argument| (
                    argument.name.as_str(),
                    argument.binding,
                    argument.constraint.clone()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("N", ArgumentBinding::Runtime, ParameterConstraint::Integer),
                ("P", ArgumentBinding::Runtime, ParameterConstraint::Pair),
                (
                    "B",
                    ArgumentBinding::Runtime,
                    ParameterConstraint::Bytes(32)
                ),
                ("U", ArgumentBinding::Runtime, ParameterConstraint::Unknown),
            ]
        );

        let curried_program = canonical_curry_bytes(&typed_program, &[left_env, vec![5]]);
        let curried = metadata
            .symbolize_frame(&curried_program, &[pair_value, bytes32, unknown])
            .expect("symbolize curried typed function");
        assert_eq!(curried.matched, FrameMatch::Curried);
        assert_eq!(curried.arguments[0].name, "N");
        assert_eq!(curried.arguments[0].binding, ArgumentBinding::Bound);
        assert!(curried.arguments[1..]
            .iter()
            .all(|argument| argument.binding == ArgumentBinding::Runtime));
        let rendered =
            format_stack_frame(&metadata, &curried, StackFrameStyle::Python).expect("render");
        assert!(
            rendered.starts_with("typed(N: integer = 5, P: pair = (l . 8), B: bytes[32] = "),
            "{rendered}"
        );
        assert!(rendered.contains("U: unknown = \"opaque\""));
        assert!(rendered.contains("# bound: N"));
        assert!(rendered.contains("--> typed.clsp:"));
    }
}
