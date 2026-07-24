use std::fs;
use std::rc::Rc;

use clvm_rs::error::EvalErr;
use clvmr::allocator::NodePtr;

use crate::classic::clvm::__type_compatibility__::{Bytes, Stream, UnvalidatedBytesFromType};
use crate::classic::clvm::serialize::{sexp_from_stream, SimpleCreateCLVMObject};
use crate::classic::clvm::sexp::{proper_list, rest};
use crate::classic::clvm_tools::stages::assemble;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::classic::clvm_tools::stages::stage_2::abstraction::{
    ASExp, BufCarrier, ClError, ClassicAllocator,
};
use crate::classic::clvm_tools::stages::stage_2::compile::get_search_paths;
use crate::classic::clvm_tools::stages::stage_2::helpers::quote;
use crate::classic::clvm_tools::stages::stage_2::operators::full_path_for_filename;

use crate::compiler::sexp::decode_string;

/// An object that represents file contents that were found when fulfilling a
/// form that requested some data be embedded at compile time in this program.
pub struct PresentFile {
    pub data: Vec<u8>,
    pub full_path: String,
    pub search_paths: Vec<String>,
}

/// Given u8 data from a hex file, build an sexp from it.
/// This is used for the compile-file and embed-file feature.
pub fn convert_hex_to_sexp<A: ClassicAllocator>(
    allocator: &mut A,
    parent_sexp: &A::NodePtr,
    file_data: &[u8],
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(parent_sexp);
    let content_bytes = Bytes::new_validated(Some(UnvalidatedBytesFromType::Hex(decode_string(
        file_data,
    ))))
    .map_err(|e| {
        ClError(
            loc.clone(),
            EvalErr::InternalError(NodePtr::NIL, e.to_string()),
        )
    })?;
    let mut reader_stream = Stream::new(Some(content_bytes));
    let incoming_data = sexp_from_stream(
        allocator.allocator(),
        &mut reader_stream,
        Box::new(SimpleCreateCLVMObject {}),
    )
    .map_err(|e| ClError(loc.clone(), e))?;
    allocator.import(loc, incoming_data.1)
}

/// Given a runner (which in the case of classic, contains the search paths as
/// reading a file is done by evaluating a clvm program on this special compile
/// time runner), try to find a file to embed given its name.  Try to report an
/// error nicely by using the form the user gave (parent_sexp) in the error
/// report.
pub fn read_file<A: ClassicAllocator>(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut A,
    parent_sexp: &A::NodePtr,
    filename: &str,
) -> Result<PresentFile, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(parent_sexp);
    let search_paths = get_search_paths(runner, loc.clone(), allocator)?;
    let full_path = full_path_for_filename(allocator, parent_sexp, filename, &search_paths)?;

    let export = allocator.export(parent_sexp);
    fs::read(full_path.clone())
        .map_err(|x| {
            ClError(
                loc,
                EvalErr::InternalError(export, format!("error reading {full_path}: {x:?}")),
            )
        })
        .map(|data| PresentFile {
            data,
            full_path,
            search_paths,
        })
}

/// Given an sexp representing an embedding preprocessor form of some kind such
/// as (embed-file constant-name kind filename)
/// or (compile-file constant-name filename)
/// Return the resulting constant name and a quoted expression suitable for use
/// as a constant or an error if the file wasn't found.
pub fn process_embed_file<A: ClassicAllocator>(
    allocator: &mut A,
    runner: Rc<dyn TRunProgram>,
    declaration_sexp: &A::NodePtr,
) -> Result<(Vec<u8>, A::NodePtr), ClError>
where
    A::NodePtr: Clone,
{
    // Include the file's contents in the constant pool.
    // The user can specify the format to read:
    //
    // bin
    // hex
    // sexp
    let rest_of_decl = rest(allocator, declaration_sexp)?;
    let loc = allocator.loc(declaration_sexp);
    let export = allocator.export(declaration_sexp);
    if let Some(l) = proper_list(allocator, &rest_of_decl, true) {
        if l.len() != 3 {
            let loc = allocator.loc(declaration_sexp);
            let dec_export = allocator.export(declaration_sexp);
            return Err(ClError(
                loc,
                EvalErr::InternalError(dec_export, "must have a type and a name".to_string()),
            ));
        }

        if let (ASExp::Atom, ASExp::Atom, ASExp::Atom) = (
            allocator.sexp(&l[0]),
            allocator.sexp(&l[1]),
            allocator.sexp(&l[2]),
        ) {
            // Note: we don't want to keep borrowing here because we
            // need the mutable borrow below.
            let name_atom = allocator.atom(&l[0]);
            let kind_atom = allocator.atom(&l[1]);
            let filename_atom = allocator.atom(&l[2]);
            let name_buf = name_atom.as_ref().to_vec();
            let kind_buf = kind_atom.as_ref().to_vec();
            let filename_buf = filename_atom.as_ref().to_vec();
            let file_data = if kind_buf == b"bin" {
                let file = read_file(
                    runner,
                    allocator,
                    declaration_sexp,
                    &decode_string(&filename_buf),
                )?;
                allocator.new_atom(loc, &file.data)?
            } else if kind_buf == b"hex" {
                let file = read_file(
                    runner,
                    allocator,
                    declaration_sexp,
                    &decode_string(&filename_buf),
                )?;
                convert_hex_to_sexp(allocator, declaration_sexp, &file.data)?
            } else if kind_buf == b"sexp" {
                let file = read_file(
                    runner,
                    allocator,
                    declaration_sexp,
                    &decode_string(&filename_buf),
                )?;
                let assembled = assemble(allocator.allocator(), &decode_string(&file.data))
                    .map_err(|e| ClError(loc.clone(), e))?;
                allocator.import(loc, assembled)?
            } else {
                return Err(ClError(
                    loc,
                    EvalErr::InternalError(export, "no such embed kind".to_string()),
                ));
            };

            Ok((name_buf.to_vec(), quote(allocator, &file_data)?))
        } else {
            Err(ClError(
                loc,
                EvalErr::InternalError(export, "malformed embed-file".to_string()),
            ))
        }
    } else {
        Err(ClError(
            loc,
            EvalErr::InternalError(export, "must be a proper list".to_string()),
        ))
    }
}
