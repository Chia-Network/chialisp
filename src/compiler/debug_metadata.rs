//! Versioned, structural debug metadata for serialized CLVM programs.
//!
//! The persisted form is CLVM serialization, not a hash-indexed side table:
//!
//! ```text
//! ("CHIALISP_DEBUG" 1 (1 8) PROGRAM_SHA256
//!   ((PATH FULL_UTF8_SOURCE) ...)
//!   (STRING ...)
//!   ((FILE START_LINE START_COLUMN END_LINE END_COLUMN) ...)
//!   SHADOW_TREE)
//! ```
//!
//! Coordinates are one-based display columns. Tabs advance to the next
//! one-based 8-column tab stop, matching [`Srcloc::advance`]. Span ends are
//! exclusive. The shadow tree uses `(0 SPAN STRING ATOM)` for atoms and
//! `(1 SPAN STRING LEFT RIGHT)` for pairs. Table references are encoded as
//! `index + 1`; zero means absent.

use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr};
use clvm_rs::serde::node_to_bytes;
use sha2::{Digest, Sha256};

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::serialize::{sexp_from_stream, sexp_to_stream, SimpleCreateCLVMObject};
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::compiler::{compile_file, ADVANCED_MACROS, STANDARD_MACROS};
use crate::compiler::comptypes::{CompileErr, CompilerOpts, CompilerOutput};
use crate::compiler::debug::build_symbol_table_mut;
use crate::compiler::dialect::KNOWN_DIALECTS;
use crate::compiler::sexp::SExp;
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
        value: Vec<u8>,
    },
    Pair {
        span: Option<usize>,
        label: Option<usize>,
        left: Box<DebugNode>,
        right: Box<DebugNode>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugMetadata {
    pub program_sha256: [u8; 32],
    pub files: Vec<DebugSourceFile>,
    pub strings: Vec<String>,
    pub spans: Vec<DebugSourceSpan>,
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
    /// CLVM tree hash of the executable frame, retained even when unknown.
    pub program_hash: [u8; 32],
    /// Canonical CLVM serialization of values quoted into curry wrappers.
    pub bound_arguments: Vec<Vec<u8>>,
    /// Canonical CLVM serialization of arguments supplied by the caller.
    pub runtime_arguments: Vec<Vec<u8>>,
}

#[derive(Default)]
struct InternState {
    files: Vec<DebugSourceFile>,
    file_indices: HashMap<String, usize>,
    strings: Vec<String>,
    string_indices: HashMap<String, usize>,
    spans: Vec<DebugSourceSpan>,
    span_indices: HashMap<(usize, usize, usize, usize, usize), usize>,
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

fn tree_hash_hex(program: &SExp) -> String {
    hex::encode(crate::compiler::clvm::sha256tree(Rc::new(program.clone())))
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
        .or_else(|| fs::read_to_string(path).ok())
        .ok_or_else(|| format!("debug metadata has no UTF-8 source for {path}"))
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
        self.files.push(DebugSourceFile {
            path: path.to_string(),
            source: source_for_file(path, supplied)?,
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
        let key = (file, loc.line, loc.col, end_line, end_column);
        if let Some(index) = self.span_indices.get(&key) {
            return Ok(*index);
        }
        let index = self.spans.len();
        self.spans.push(DebugSourceSpan {
            file,
            start_line: loc.line,
            start_column: loc.col,
            end_line,
            end_column,
        });
        self.span_indices.insert(key, index);
        Ok(index)
    }
}

fn build_tree(
    program: &SExp,
    symbols: &HashMap<String, String>,
    sources: &HashMap<String, String>,
    state: &mut InternState,
) -> Result<DebugNode, String> {
    let span = Some(state.intern_span(&program.loc(), sources)?);
    let label = symbols
        .get(&tree_hash_hex(program))
        .filter(|name| !name.contains('('))
        .map(|name| state.intern_string(name));
    match program {
        SExp::Cons(_, left, right) => Ok(DebugNode::Pair {
            span,
            label,
            left: Box::new(build_tree(left, symbols, sources, state)?),
            right: Box::new(build_tree(right, symbols, sources, state)?),
        }),
        _ => Ok(DebugNode::Atom {
            span,
            label,
            value: atom_bytes(program),
        }),
    }
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
        let mut state = InternState::default();
        let tree = build_tree(program, symbols, sources, &mut state)?;
        Ok(DebugMetadata {
            program_sha256: Sha256::digest(program_bytes).into(),
            files: state.files,
            strings: state.strings,
            spans: state.spans,
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
    /// curry wrappers. Caller-supplied runtime arguments remain distinct from
    /// values quoted into curry.
    pub fn symbolize_frame(
        &self,
        program_bytes: &[u8],
        runtime_arguments: &[Vec<u8>],
    ) -> Result<SymbolizedFrame, String> {
        let mut allocator = Allocator::new();
        let program = decode_clvm(&mut allocator, program_bytes, "frame program")?;
        let mut bound_arguments = Vec::new();
        let (matched, function) = symbolize_node(
            &allocator,
            program,
            &self.tree,
            &self.strings,
            &mut bound_arguments,
        );
        let hash = crate::classic::clvm_tools::sha256tree::sha256tree(&mut allocator, program);
        let program_hash: [u8; 32] = hash
            .data()
            .as_slice()
            .try_into()
            .map_err(|_| "CLVM tree hash was not 32 bytes".to_string())?;
        Ok(SymbolizedFrame {
            matched,
            function,
            program_hash,
            bound_arguments,
            runtime_arguments: runtime_arguments.to_vec(),
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
        let coordinate_base = uint(allocator, 1)?;
        let tab_width = uint(allocator, DEBUG_METADATA_TAB_WIDTH)?;
        let coordinates = list(allocator, &[coordinate_base, tab_width])?;
        let files = list(allocator, &files)?;
        let strings = list(allocator, &strings)?;
        let spans = list(allocator, &spans)?;
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
                tree,
            ],
        )
    }

    fn from_clvm(allocator: &Allocator, root: NodePtr) -> Result<Self, String> {
        let root = proper_list(allocator, root)?;
        if root.len() != 8 || atom_value(allocator, root[0])? != MAGIC {
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
        let tree = decode_tree(allocator, root[7], spans.len(), strings.len())?;
        Ok(DebugMetadata {
            program_sha256,
            files,
            strings,
            spans,
            tree,
        })
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
    let mut allocator = Allocator::new();
    let mut compiler_symbols = HashMap::new();
    let output = compile_file(
        &mut allocator,
        Rc::new(DefaultProgramRunner::new()),
        opts.clone(),
        content,
        &mut compiler_symbols,
    )?;
    let mut sources = HashMap::new();
    sources.insert(opts.filename(), content.to_string());
    sources.insert("*macros*".to_string(), STANDARD_MACROS.to_string());
    if opts.dialect().strict {
        sources.insert("*macros*".to_string(), ADVANCED_MACROS.to_string());
    }

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
            let mut symbols = HashMap::new();
            build_symbol_table_mut(&mut symbols, &program);
            for (key, value) in &compiler_symbols {
                symbols.insert(key.clone(), value.clone());
            }
            let metadata =
                DebugMetadata::from_program_bytes(&program, &program_bytes, &symbols, &sources)
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

fn encode_tree(allocator: &mut Allocator, tree: &DebugNode) -> Result<NodePtr, String> {
    match tree {
        DebugNode::Atom { span, label, value } => {
            let tag = uint(allocator, 0)?;
            let span = optional_index(allocator, *span)?;
            let label = optional_index(allocator, *label)?;
            let value = atom(allocator, value)?;
            list(allocator, &[tag, span, label, value])
        }
        DebugNode::Pair {
            span,
            label,
            left,
            right,
        } => {
            let tag = uint(allocator, 1)?;
            let span = optional_index(allocator, *span)?;
            let label = optional_index(allocator, *label)?;
            let left = encode_tree(allocator, left)?;
            let right = encode_tree(allocator, right)?;
            list(allocator, &[tag, span, label, left, right])
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
) -> Result<DebugNode, String> {
    let fields = proper_list(allocator, node)?;
    if fields.is_empty() {
        return Err("empty debug shadow node".to_string());
    }
    let tag = decode_uint(&atom_value(allocator, fields[0])?)?;
    match tag {
        0 if fields.len() == 4 => Ok(DebugNode::Atom {
            span: decode_optional_index(&atom_value(allocator, fields[1])?, span_count)?,
            label: decode_optional_index(&atom_value(allocator, fields[2])?, string_count)?,
            value: atom_value(allocator, fields[3])?,
        }),
        1 if fields.len() == 5 => Ok(DebugNode::Pair {
            span: decode_optional_index(&atom_value(allocator, fields[1])?, span_count)?,
            label: decode_optional_index(&atom_value(allocator, fields[2])?, string_count)?,
            left: Box::new(decode_tree(allocator, fields[3], span_count, string_count)?),
            right: Box::new(decode_tree(allocator, fields[4], span_count, string_count)?),
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
) -> (FrameMatch, Option<String>) {
    if let Some(exact) = exact_tree_match(allocator, node, metadata_tree) {
        return (FrameMatch::Exact, node_label(exact, strings));
    }
    let Some((base, arguments)) = canonical_curry(allocator, node) else {
        return (FrameMatch::Unknown, None);
    };
    let (matched, function) =
        symbolize_node(allocator, base, metadata_tree, strings, bound_arguments);
    for argument in arguments {
        let Ok(bytes) = node_to_bytes(allocator, argument) else {
            return (FrameMatch::Unknown, None);
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
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::sexp::parse_sexp;

    fn fixture() -> (SExp, HashMap<String, String>, HashMap<String, String>) {
        let source = "(+ X 1)";
        let program = parse_sexp(Srcloc::start("fixture.clsp"), source.bytes())
            .expect("parse fixture")
            .remove(0);
        let program = program.as_ref().clone();
        let mut symbols = HashMap::new();
        symbols.insert(tree_hash_hex(&program), "fixture".to_string());
        let mut sources = HashMap::new();
        sources.insert("fixture.clsp".to_string(), source.to_string());
        (program, symbols, sources)
    }

    #[test]
    fn structural_metadata_round_trips_and_verifies_exact_program() {
        let (program, symbols, sources) = fixture();
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
        assert_eq!(decoded.strings, vec!["fixture"]);
        assert!(decoded.spans.len() < 8, "spans should be interned");
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
}
