use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::chia_dialect::{ChiaDialect, NO_UNKNOWN_OPS};
use clvm_rs::cost::Cost;
use clvm_rs::dialect::{Dialect, OperatorSet};
use clvm_rs::error::EvalErr;
use clvm_rs::reduction::{Reduction, Response};
use clvm_rs::run_program::run_program_with_pre_eval;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType, Stream};
use crate::classic::clvm::OPERATORS_LATEST_VERSION;

use crate::classic::clvm::keyword_from_atom;
use crate::classic::clvm::sexp::proper_list;

use crate::classic::clvm_tools::binutils::{assemble_from_ir, disassemble_to_ir_with_kw};
use crate::classic::clvm_tools::ir::reader::read_ir;
use crate::classic::clvm_tools::ir::writer::write_ir_to_stream;
use crate::classic::clvm_tools::sha256tree::TreeHash;
use crate::classic::clvm_tools::stages::stage_0::{
    DefaultProgramRunner, OriginalDialect, RunProgramOption, TRunProgram,
};
use crate::classic::clvm_tools::stages::stage_2::compile::do_com_prog_for_dialect;
use crate::classic::clvm_tools::stages::stage_2::optimize::do_optimize;

use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::sexp::decode_string;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocatorRefOrTreeHash {
    AllocatorRef(NodePtr),
    TreeHash(TreeHash),
}

impl AllocatorRefOrTreeHash {
    pub fn new_from_nodeptr(n: NodePtr) -> Self {
        AllocatorRefOrTreeHash::AllocatorRef(n)
    }

    pub fn new_from_sexp(allocator: &mut Allocator, n: NodePtr) -> Self {
        AllocatorRefOrTreeHash::TreeHash(TreeHash::new_from_sexp(allocator, n))
    }
}

pub struct CompilerOperatorsInternal {
    source_file: String,
    search_paths: Vec<String>,
    symbols_extra_info: bool,
    compile_outcomes: RefCell<HashMap<String, String>>,
    runner: RefCell<Rc<dyn TRunProgram>>,
    opt_memo: RefCell<HashMap<AllocatorRefOrTreeHash, NodePtr>>,
    // A compiler opts as in the modern compiler.  If present, try using its
    // file system interface to read files.
    compiler_opts: RefCell<Option<Rc<dyn CompilerOpts>>>,
    // The version of the operators selected by the user.  version 1 includes bls.
    operators_version: RefCell<Option<usize>>,
}

/// Given a list of search paths, find a full path to a file whose partial name
/// is given.  If the file can't be found in any search path, use the expression
/// the user gave to cause the file to be searched for in the error result.
/// They're searched in order so repetition doesn't do anything. (suggested Q+A)
pub fn full_path_for_filename(
    parent_sexp: NodePtr,
    filename: &str,
    search_paths: &[String],
) -> Result<String, EvalErr> {
    for path in search_paths.iter() {
        let mut path_buf = PathBuf::new();
        path_buf.push(path);
        path_buf.push(filename);
        let f_path = path_buf.as_path();
        if f_path.exists() {
            return f_path
                .to_str()
                .map(|x| x.to_owned())
                .map(Ok)
                .unwrap_or_else(|| {
                    Err(EvalErr::InternalError(
                        parent_sexp,
                        format!("could not compute absolute path for the combination of search path {path} and file name {filename} during text conversion from path_buf")
                    ))
                });
        }
    }

    Err(EvalErr::InternalError(
        parent_sexp,
        "can't open file".to_string(),
    ))
}

pub struct CompilerOperators {
    parent: Rc<CompilerOperatorsInternal>,
}

// This wrapper object is here to hold a drop trait that untangles CompilerOperatorsInternal.
// The design of this code requires the lifetime of the compiler object (via Rc<dyn TRunProgram>)
// to be uncertain from rust's perspective (i'm not given a lifetime parameter for the runnable
// passed to clvmr, so I chose this way to mitigate the lack of specifiable lifetime as passing
// down a reference became very hairy.  The downside is that a few objects had become fixed in
// an Rc cycle.  The drop trait below corrects that.
impl CompilerOperators {
    pub fn new(source_file: &str, search_paths: Vec<String>, symbols_extra_info: bool) -> Self {
        CompilerOperators {
            parent: Rc::new(CompilerOperatorsInternal::new(
                source_file,
                search_paths,
                symbols_extra_info,
            )),
        }
    }
}

impl Drop for CompilerOperators {
    fn drop(&mut self) {
        self.parent.neutralize();
    }
}

impl CompilerOperatorsInternal {
    pub fn new(source_file: &str, search_paths: Vec<String>, symbols_extra_info: bool) -> Self {
        let base_runner = Rc::new(DefaultProgramRunner::new());
        CompilerOperatorsInternal {
            source_file: source_file.to_owned(),
            search_paths,
            symbols_extra_info,
            compile_outcomes: RefCell::new(HashMap::new()),
            runner: RefCell::new(base_runner),
            opt_memo: RefCell::new(HashMap::new()),
            compiler_opts: RefCell::new(None),
            operators_version: RefCell::new(None),
        }
    }

    pub fn neutralize(&self) {
        self.set_runner(Rc::new(DefaultProgramRunner::new()));
        self.set_compiler_opts(None);
    }

    fn symbols_extra_info(&self, allocator: &mut Allocator) -> Response {
        if self.symbols_extra_info {
            Ok(Reduction(1, allocator.new_atom(&[1])?))
        } else {
            Ok(Reduction(1, NodePtr::NIL))
        }
    }

    fn set_runner(&self, runner: Rc<dyn TRunProgram>) {
        self.runner.replace(runner);
    }

    fn get_runner(&self) -> Rc<dyn TRunProgram> {
        let borrow: Ref<'_, Rc<dyn TRunProgram>> = self.runner.borrow();
        borrow.clone()
    }

    fn set_compiler_opts(&self, opts: Option<Rc<dyn CompilerOpts>>) {
        self.compiler_opts.replace(opts);
    }

    fn get_compiler_opts(&self) -> Option<Rc<dyn CompilerOpts>> {
        let borrow: Ref<'_, Option<Rc<dyn CompilerOpts>>> = self.compiler_opts.borrow();
        borrow.clone()
    }

    fn get_operators_version(&self) -> Option<usize> {
        let borrow: Ref<'_, Option<usize>> = self.operators_version.borrow();
        *borrow
    }

    // Return the extension operator system to use while compiling based on user
    // preference.
    fn get_operators_extension(&self) -> OperatorSet {
        let ops_version = self
            .get_operators_version()
            .unwrap_or(OPERATORS_LATEST_VERSION);
        match ops_version {
            0 => OperatorSet::Default,
            1 => OperatorSet::Bls,
            _ => OperatorSet::Keccak,
        }
    }

    fn set_operators_version(&self, ver: Option<usize>) {
        self.operators_version.replace(ver);
    }

    fn read(&self, allocator: &mut Allocator, sexp: NodePtr) -> Response {
        // Given a string containing the data in the file to parse, parse it or
        // return EvalErr.
        let parse_file_content = |allocator: &mut Allocator, content: &String| {
            read_ir(content)
                .map_err(|e| EvalErr::InternalError(NodePtr::NIL, e.to_string()))
                .and_then(|ir| {
                    assemble_from_ir(allocator, Rc::new(ir)).map(|ir_sexp| Reduction(1, ir_sexp))
                })
        };

        match allocator.sexp(sexp) {
            SExp::Pair(f, _) => match allocator.sexp(f) {
                SExp::Atom => {
                    let atom = allocator.atom(f);
                    let filename =
                        Bytes::new(Some(BytesFromType::Raw(atom.as_ref().to_vec()))).decode();
                    // Use the read interface in CompilerOpts if we have one.
                    if let Some(opts) = self.get_compiler_opts() {
                        if let Ok((_, content)) =
                            opts.read_new_file(self.source_file.clone(), filename.clone())
                        {
                            return parse_file_content(allocator, &decode_string(&content));
                        }
                    }

                    // Use the filesystem like normal if the opts couldn't find
                    // the file.
                    fs::read_to_string(&filename)
                        .map_err(|_| {
                            EvalErr::InternalError(
                                NodePtr::NIL,
                                format!("Failed to read file {filename}"),
                            )
                        })
                        .and_then(|content| parse_file_content(allocator, &content))
                }
                _ => Err(EvalErr::InternalError(
                    NodePtr::NIL,
                    "filename is not an atom".to_string(),
                )),
            },
            _ => Err(EvalErr::InternalError(
                NodePtr::NIL,
                "given a program that is an atom".to_string(),
            )),
        }
    }

    fn write(&self, allocator: &Allocator, sexp: NodePtr) -> Response {
        if let SExp::Pair(filename_sexp, r) = allocator.sexp(sexp) {
            if let SExp::Pair(data, _) = allocator.sexp(r) {
                if let SExp::Atom = allocator.sexp(filename_sexp) {
                    let filename_atom = allocator.atom(filename_sexp);
                    let filename_bytes =
                        Bytes::new(Some(BytesFromType::Raw(filename_atom.as_ref().to_vec())));
                    let ir = disassemble_to_ir_with_kw(
                        allocator,
                        data,
                        keyword_from_atom(self.get_disassembly_ver()),
                        true,
                    );
                    let mut stream = Stream::new(None);
                    write_ir_to_stream(Rc::new(ir), &mut stream);
                    return fs::write(filename_bytes.decode(), stream.get_value().decode())
                        .map_err(|_| {
                            EvalErr::InternalError(
                                sexp,
                                format!("failed to write {}", filename_bytes.decode()),
                            )
                        })
                        .map(|_| Reduction(1, NodePtr::NIL));
                }
            }
        }

        Err(EvalErr::InternalError(
            sexp,
            "failed to write data".to_string(),
        ))
    }

    fn get_compile_filename(&self, allocator: &mut Allocator) -> Response {
        let converted_filename = allocator.new_atom(self.source_file.as_bytes())?;
        Ok(Reduction(1, converted_filename))
    }

    fn get_include_paths(&self, allocator: &mut Allocator) -> Response {
        let mut converted_search_paths = NodePtr::NIL;
        for s in self.search_paths.iter().rev() {
            let search_path_string = allocator.new_atom(s.as_bytes())?;
            converted_search_paths =
                allocator.new_pair(search_path_string, converted_search_paths)?;
        }

        Ok(Reduction(1, converted_search_paths))
    }

    fn get_full_path_for_filename(&self, allocator: &mut Allocator, sexp: NodePtr) -> Response {
        let convert_filename = |allocator: &mut Allocator, full_name: &String| {
            let converted_filename = allocator.new_atom(full_name.as_bytes())?;
            Ok(Reduction(1, converted_filename))
        };

        if let SExp::Pair(l, _r) = allocator.sexp(sexp) {
            if let SExp::Atom = allocator.sexp(l) {
                // l most relevant in scope.
                let atom = allocator.atom(l);
                let filename =
                    Bytes::new(Some(BytesFromType::Raw(atom.as_ref().to_vec()))).decode();
                // If we have a compiler opts injected, let that handle reading
                // files.  The name will bubble up to the _read function.
                if self.get_compiler_opts().is_some() {
                    return convert_filename(allocator, &filename);
                }

                let full_name = full_path_for_filename(sexp, &filename, &self.search_paths)?;
                return convert_filename(allocator, &full_name);
            }
        }

        Err(EvalErr::InternalError(sexp, "can't open file".to_string()))
    }

    fn get_source_file(&self, allocator: &mut Allocator) -> Result<Reduction, EvalErr> {
        let file_name_bytes = Bytes::new(Some(BytesFromType::String(self.source_file.clone())));
        let new_atom = allocator.new_atom(file_name_bytes.data())?;
        Ok(Reduction(1, new_atom))
    }

    pub fn set_symbol_table(
        &self,
        allocator: &Allocator,
        table: NodePtr,
    ) -> Result<Reduction, EvalErr> {
        if let Some(symtable) =
            proper_list(allocator, table, true).and_then(|t| proper_list(allocator, t[0], true))
        {
            for kv in symtable.iter() {
                if let SExp::Pair(hash, name) = allocator.sexp(*kv) {
                    if let (SExp::Atom, SExp::Atom) = (allocator.sexp(hash), allocator.sexp(name)) {
                        // hash and name in scope.
                        let hash_atom = allocator.atom(hash);
                        let hash_text =
                            Bytes::new(Some(BytesFromType::Raw(hash_atom.as_ref().to_vec())))
                                .decode();
                        let name_atom = allocator.atom(name);
                        let name_text =
                            Bytes::new(Some(BytesFromType::Raw(name_atom.as_ref().to_vec())))
                                .decode();

                        self.compile_outcomes.replace_with(|co| {
                            let mut result = co.clone();
                            result.insert(hash_text, name_text);
                            result
                        });
                    }
                }
            }
        }

        Ok(Reduction(1, NodePtr::NIL))
    }

    fn get_disassembly_ver(&self) -> usize {
        self.get_compiler_opts()
            .and_then(|o| o.disassembly_ver())
            .unwrap_or(OPERATORS_LATEST_VERSION)
    }
}

impl Dialect for CompilerOperatorsInternal {
    fn quote_kw(&self) -> u32 {
        1
    }

    fn apply_kw(&self) -> u32 {
        2
    }

    fn softfork_kw(&self) -> u32 {
        36
    }

    // The softfork operator comes with an extension argument.
    fn softfork_extension(&self, ext: u32) -> OperatorSet {
        match ext {
            0 => OperatorSet::Bls,
            // new extensions go here
            _ => OperatorSet::Default,
        }
    }

    fn op(
        &self,
        allocator: &mut Allocator,
        op: NodePtr,
        sexp: NodePtr,
        max_cost: Cost,
        _extension: OperatorSet,
    ) -> Response {
        let new_operators = self
            .get_operators_version()
            .unwrap_or(OPERATORS_LATEST_VERSION);
        let base_dialect = if new_operators > 0 {
            let rc: Box<dyn Dialect> = Box::new(ChiaDialect::new(NO_UNKNOWN_OPS));
            rc
        } else {
            let rc: Box<dyn Dialect> = Box::new(OriginalDialect::new(NO_UNKNOWN_OPS));
            rc
        };
        // Ensure we have at least the bls extensions available.
        // The extension passed in above is based on the state of whether
        // we're approaching from within softfork...  As the compiler author
        // we're overriding this so the user can specify these in the compile
        // context...  Even when compiling code to go inside softfork, the
        // compiler doesn't itself run in a softfork.
        let extensions_to_clvmr_during_compile = self.get_operators_extension();

        match allocator.sexp(op) {
            SExp::Atom => {
                // use of op obvious.
                let op_atom = allocator.atom(op);
                let opbuf = op_atom.as_ref();
                if opbuf == b"_read" {
                    self.read(allocator, sexp)
                } else if opbuf == b"_write" {
                    self.write(allocator, sexp)
                } else if opbuf == b"com" {
                    do_com_prog_for_dialect(self.get_runner(), allocator, sexp)
                } else if opbuf == b"opt" {
                    do_optimize(self.get_runner(), allocator, &self.opt_memo, sexp)
                } else if opbuf == b"_set_symbol_table" {
                    self.set_symbol_table(allocator, sexp)
                } else if opbuf == b"_get_compile_filename" {
                    self.get_compile_filename(allocator)
                } else if opbuf == b"_get_include_paths" {
                    self.get_include_paths(allocator)
                } else if opbuf == b"_full_path_for_name" {
                    self.get_full_path_for_filename(allocator, sexp)
                } else if opbuf == b"_symbols_extra_info" {
                    self.symbols_extra_info(allocator)
                } else if opbuf == b"_get_source_file" {
                    self.get_source_file(allocator)
                } else {
                    base_dialect.op(
                        allocator,
                        op,
                        sexp,
                        max_cost,
                        extensions_to_clvmr_during_compile,
                    )
                }
            }
            _ => base_dialect.op(
                allocator,
                op,
                sexp,
                max_cost,
                extensions_to_clvmr_during_compile,
            ),
        }
    }

    fn allow_unknown_ops(&self) -> bool {
        false
    }
}

impl CompilerOperatorsInternal {
    pub fn get_compiles(&self) -> HashMap<String, String> {
        self.compile_outcomes.borrow().clone()
    }
}

impl CompilerOperators {
    pub fn get_compiles(&self) -> HashMap<String, String> {
        self.parent.get_compiles()
    }

    pub fn set_compiler_opts(&self, opts: Option<Rc<dyn CompilerOpts>>) {
        self.parent.set_compiler_opts(opts);
    }

    pub fn set_operators_version(&self, ver: Option<usize>) {
        self.parent.set_operators_version(ver);
    }
}

impl TRunProgram for CompilerOperatorsInternal {
    fn run_program(
        &self,
        allocator: &mut Allocator,
        program: NodePtr,
        args: NodePtr,
        option: Option<RunProgramOption>,
    ) -> Response {
        let max_cost = option.as_ref().and_then(|o| o.max_cost).unwrap_or(0);
        run_program_with_pre_eval(
            allocator,
            self,
            program,
            args,
            max_cost,
            option.and_then(|o| o.pre_eval_f),
        )
    }
}

impl TRunProgram for CompilerOperators {
    fn run_program(
        &self,
        allocator: &mut Allocator,
        program: NodePtr,
        args: NodePtr,
        option: Option<RunProgramOption>,
    ) -> Response {
        self.parent.run_program(allocator, program, args, option)
    }
}

pub fn run_program_for_search_paths(
    source_file: &str,
    search_paths: &[String],
    symbols_extra_info: bool,
) -> Rc<CompilerOperators> {
    let ops = Rc::new(CompilerOperators::new(
        source_file,
        search_paths.to_vec(),
        symbols_extra_info,
    ));
    ops.parent.set_runner(ops.parent.clone());
    ops
}
