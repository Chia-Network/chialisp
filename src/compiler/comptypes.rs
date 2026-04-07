use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use serde::Serialize;

use clvm_rs::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::clvm::{sha256tree, truthy};
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::sexp::{decode_string, enlist, SExp};
use crate::compiler::srcloc::Srcloc;

// Note: only used in tests, not normally dependencies.
#[cfg(test)]
use crate::compiler::compiler::DefaultCompilerOpts;
#[cfg(test)]
use crate::compiler::frontend::compile_bodyform;
#[cfg(test)]
use crate::compiler::sexp::parse_sexp;

/// The basic error type.  It contains a Srcloc identifying coordinates of the
/// error in the source file and a message.  It probably should be made even better
/// but this works ok.
#[derive(Clone, Debug)]
pub struct CompileErr(pub Srcloc, pub String);

impl From<(Srcloc, String)> for CompileErr {
    fn from(err: (Srcloc, String)) -> Self {
        CompileErr(err.0, err.1)
    }
}

/// A structure carrying a compilation result to give it a distinct type from
/// chialisp input.  It's used by codegen.
#[derive(Clone, Debug)]
pub struct CompiledCode(pub Srcloc, pub Rc<SExp>);

/// A description of an inlined function for use during inline expansion.
/// This is used only by PrimaryCodegen.
#[derive(Clone, Debug)]
pub struct InlineFunction {
    pub name: Vec<u8>,
    pub args: Rc<SExp>,
    pub body: Rc<BodyForm>,
}

impl InlineFunction {
    pub fn to_sexp(&self) -> Rc<SExp> {
        Rc::new(SExp::Cons(
            self.body.loc(),
            self.args.clone(),
            self.body.to_sexp(),
        ))
    }
}

/// Specifies the type of application that any form (X ...) invokes in an
/// expression position.
pub enum Callable {
    /// The expression is a macro expansion (list, if etc.)
    CallMacro(Srcloc, SExp),
    /// The expression invokes an env defun.
    CallDefun(Srcloc, SExp),
    /// The expression expands and inline function.
    CallInline(Srcloc, InlineFunction),
    /// The expression addresses a clvm primitive (such as a, c, f, =)
    CallPrim(Srcloc, SExp),
    /// The expression is a (com ...) invokcation (normally used in macros).
    RunCompiler,
    /// The expression is an (@ n) form that directly references the environment.
    EnvPath,
}

/// Given a slice of SExp values, generate a proper list containing them.
pub fn list_to_cons(l: Srcloc, list: &[Rc<SExp>]) -> SExp {
    if list.is_empty() {
        return SExp::Nil(l);
    }

    let mut result = SExp::Nil(l);
    for i_reverse in 0..list.len() {
        let i = list.len() - i_reverse - 1;
        result = SExp::Cons(list[i].loc(), list[i].clone(), Rc::new(result));
    }

    result
}

/// Specifies the pattern that is destructured in let bindings.
#[derive(Clone, Debug, Serialize)]
pub enum BindingPattern {
    /// The whole expression is bound to this name.
    Name(Vec<u8>),
    /// Specifies a tree of atoms into which the value will be destructured.
    Complex(Rc<SExp>),
}

/// If present, states an intention for desugaring of this let form to favor
/// inlining or functions.
#[derive(Clone, Debug, Serialize)]
pub enum LetFormInlineHint {
    NoChoice,
    Inline(Srcloc),
    NonInline(Srcloc),
}

/// A binding from a (let ...) form.  Specifies the name of the bound variable
/// the location of the whole binding form, the location of the name atom (nl)
/// and the body as a BodyForm (which are chialisp expressions).
#[derive(Clone, Debug, Serialize)]
pub struct Binding {
    /// Overall location of the form.
    pub loc: Srcloc,
    /// Location of the name atom specifically.
    pub nl: Srcloc,
    /// Specifies the pattern which is extracted from the expression, which can
    /// be a Name (a single name names the whole subexpression) or Complex which
    /// can destructure and is used in code that extends cl21 past the definition
    /// of the language at that point.
    pub pattern: BindingPattern,
    /// The expression the binding refers to.
    pub body: Rc<BodyForm>,
}

/// Determines how a let binding is bound.  Parallel means that the bindings do
/// not depend on each other and aren't in scope for each other.  Sequential
/// is like lisp's let* form in that each binding has the previous ones in scope
/// for itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum LetFormKind {
    Parallel,
    Sequential,
    Assign,
}

/// Information about a let form.  Encapsulates everything except whether it's
/// parallel or sequential, which is left in the BodyForm itself.
#[derive(Clone, Debug, Serialize)]
pub struct LetData {
    /// The location of the form overall.
    pub loc: Srcloc,
    /// The location specifically of the let or let* keyword.
    pub kw: Option<Srcloc>,
    /// Inline hint.
    pub inline_hint: Option<LetFormInlineHint>,
    /// The bindings introduced.
    pub bindings: Vec<Rc<Binding>>,
    /// The expression evaluated in the context of all the bindings.
    pub body: Rc<BodyForm>,
}

/// Describes a lambda used in an expression.
#[derive(Clone, Debug, Serialize)]
pub struct LambdaData {
    pub loc: Srcloc,
    pub kw: Option<Srcloc>,
    pub capture_args: Rc<SExp>,
    pub captures: Rc<BodyForm>,
    pub args: Rc<SExp>,
    pub body: Rc<BodyForm>,
}

#[derive(Clone, Debug, Serialize)]
pub enum BodyForm {
    /// A let or let* form (depending on LetFormKind).
    Let(LetFormKind, Box<LetData>),
    /// An explicitly quoted constant of some kind.
    Quoted(SExp),
    /// An undiferentiated "value" of some kind in the source language.
    /// If this refers to an atom, then it is a variable reference of some kind,
    /// otherwise it refers to a self-quoting value (like a quoted string or int).
    Value(SExp),
    /// An application of some kind, parsed from a proper list.
    /// This is a proper list because of the ambiguity of the final value in an
    /// improper list.  While it's possible to treat a final atom
    ///
    /// (x y . z)
    ///
    /// as an argument that matches a tail argument, there's no way to write
    ///
    /// (x y . (+ 1 z))
    ///
    /// So tail improper calls aren't allowed.  In real lisp, (apply ...) can
    /// generate them if needed.
    Call(Srcloc, Vec<Rc<BodyForm>>, Option<Rc<BodyForm>>),
    /// (mod ...) can be used in chialisp as an expression, in which it returns
    /// the compiled code.  Here, it contains a CompileForm, which represents
    /// the full significant input of a program (yielded by frontend()).
    Mod(Srcloc, CompileForm),
    /// A lambda form (lambda (...) ...)
    ///
    /// The lambda arguments are in two parts:
    ///
    /// (lambda ((& captures) real args) ...)
    ///
    /// Where the parts in captures are captured from the hosting environment.
    /// Captures are optional.
    /// The real args are given in the indicated shape when the lambda is applied
    /// with the 'a' operator.
    Lambda(Box<LambdaData>),
}

/// Convey information about synthetically generated helper forms.
#[derive(Clone, Debug, Serialize)]
pub enum SyntheticType {
    NoInlinePreference,
    MaybeRecursive,
    WantInline,
    WantNonInline,
}

/// The information needed to know about a defun.  Whether it's inline is left in
/// the HelperForm.
#[derive(Clone, Debug, Serialize)]
pub struct DefunData {
    /// The location of the helper form.
    pub loc: Srcloc,
    /// The name of the defun.
    pub name: Vec<u8>,
    /// The location of the keyword used in the defun.
    pub kw: Option<Srcloc>,
    /// The location of the name of the defun.
    pub nl: Srcloc,
    /// The arguments as originally given by the user.
    pub orig_args: Rc<SExp>,
    /// The argument spec for the defun with any renaming.
    pub args: Rc<SExp>,
    /// The body expression of the defun.
    pub body: Rc<BodyForm>,
    /// Whether this defun was created during desugaring.
    pub synthetic: Option<SyntheticType>,
}

/// Specifies the information extracted from a macro definition allowing the
/// compiler to expand code using it.
#[derive(Clone, Debug, Serialize)]
pub struct DefmacData {
    /// The location of the macro.
    pub loc: Srcloc,
    /// The name of the macro.
    pub name: Vec<u8>,
    /// The locaton of the keyword used to define the macro.
    pub kw: Option<Srcloc>,
    /// The location of the macro's name.
    pub nl: Srcloc,
    /// The argument spec.
    pub args: Rc<SExp>,
    /// The program appearing in the macro definition.
    pub program: Rc<CompileForm>,
    /// Whether this is an an advanced macro.
    pub advanced: bool,
}

/// Information from a constant definition.
#[derive(Clone, Debug, Serialize)]
pub struct DefconstData {
    /// The location of the constant form.
    pub loc: Srcloc,
    /// Specifies whether the constant is a simple quoted sexp or is specified
    /// by an expression.  This allows us to delay constant evaluation until we
    /// have the whole program.
    pub kind: ConstantKind,
    /// The name of constant.
    pub name: Vec<u8>,
    /// The location of the keyword in the definition.
    pub kw: Option<Srcloc>,
    /// The location of the name in the definition.
    pub nl: Srcloc,
    /// The location of the body expression, whatever it is.
    pub body: Rc<BodyForm>,
    /// This constant should exist in the left env rather than be inlined.
    pub tabled: bool,
}

/// Specifies where a constant is the classic kind (unevaluated) or a proper
/// expression.
#[derive(Clone, Debug, Serialize)]
pub enum ConstantKind {
    Complex,
    Simple,
}

/// HelperForm is a toplevel binding of some kind.
/// Helpers are the (defconst ...) (defun ...) (defun-inline ...) (defmacro ...)
/// forms from the source code and "help" the program do its job.  They're
/// individually parsable and represent the atomic units of the program.
#[derive(Clone, Debug, Serialize)]
pub enum HelperForm {
    /// A constant definition (see DefconstData).
    Defconstant(DefconstData),
    /// A macro definition (see DefmacData).
    Defmacro(DefmacData),
    /// A function definition (see DefunData).
    Defun(bool, Box<DefunData>),
}

/// To what purpose is the file included.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum IncludeProcessType {
    /// Include the bytes on disk as an atom.
    Bin,
    /// Parse the hex on disk and present it as a clvm value.
    Hex,
    /// Read clvm in s-expression form as a clvm value.
    SExpression,
}

/// A description of an include form.  Here, records the locations of the various
/// parts of the include so they can be marked in the language server and be
/// subject to other kind of reporting if desired.
#[derive(Clone, Debug, Serialize)]
pub struct IncludeDesc {
    /// Location of the keyword introducing this form.
    pub kw: Srcloc,
    /// Location of the name of the file.
    pub nl: Srcloc,
    /// The relative path to a target or a special directive name.
    pub name: Vec<u8>,
    pub kind: Option<IncludeProcessType>,
}

impl IncludeDesc {
    pub fn to_sexp(&self) -> Rc<SExp> {
        Rc::new(SExp::Cons(
            self.kw.clone(),
            Rc::new(SExp::Atom(self.kw.clone(), b"include".to_vec())),
            Rc::new(SExp::QuotedString(self.nl.clone(), b'"', self.name.clone())),
        ))
    }
}

/// An encoding of a complete program.  This includes all the include forms
/// traversed (for marking in a language server), the argument spec of the program,
/// the list of helper declarations and the expression serving as the "main"
/// program.
#[derive(Clone, Debug, Serialize)]
pub struct CompileForm {
    /// Location of the form that was collected into this object.
    pub loc: Srcloc,
    /// List of include directives.
    pub include_forms: Vec<IncludeDesc>,
    /// Argument spec.
    pub args: Rc<SExp>,
    /// List of declared helpers encountered.  Unless the CompilerOpts is directed
    /// to preserve all helpers, helpers not used by a toplevel defun or the main
    /// expression (those needed by the finished code) are not included.  The
    /// set_frontend_check_live method of CompilerOpts allows this to be changed.
    pub helpers: Vec<HelperForm>,
    /// The expression the program evaluates, using the declared helpers.
    pub exp: Rc<BodyForm>,
}

/// Represents a call to a defun, used by code generation.
#[derive(Clone, Debug)]
pub struct DefunCall {
    pub required_env: Rc<SExp>,
    pub code: Rc<SExp>,
}

/// PrimaryCodegen is an object used by codegen to accumulate and use state needed
/// during code generation.  It's mostly used internally.
#[derive(Clone, Debug)]
pub struct PrimaryCodegen {
    pub prims: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    pub constants: HashMap<Vec<u8>, Rc<SExp>>,
    pub tabled_constants: HashMap<Vec<u8>, Rc<SExp>>,
    pub macros: HashMap<Vec<u8>, Rc<SExp>>,
    pub inlines: HashMap<Vec<u8>, InlineFunction>,
    pub defuns: HashMap<Vec<u8>, DefunCall>,
    pub parentfns: HashSet<Vec<u8>>,
    pub env: Rc<SExp>,
    pub to_process: Vec<HelperForm>,
    pub original_helpers: Vec<HelperForm>,
    pub final_expr: Rc<BodyForm>,
    pub final_code: Option<CompiledCode>,
    pub function_symbols: HashMap<String, String>,
    pub left_env: bool,
}

/// The CompilerOpts specifies global options used during compilation.
/// CompilerOpts is used whenever interaction with the compilation infrastructure
/// is needed that has options or needs guidance.
pub trait CompilerOpts {
    /// The toplevel file begin compiled.
    fn filename(&self) -> String;
    /// A PrimaryCodegen that can be donated to downstream use.  It can be the
    /// case that the state of compilation needs to be passed down in a specific
    /// form, such as when a lambda is used (coming soon), when evaluating
    /// complex constants, and into (com ...) forms.  This allows the CompilerOpts
    /// to carry this info across boundaries into a new context.
    fn code_generator(&self) -> Option<PrimaryCodegen>;
    /// Get the dialect declared in the toplevel program.
    fn dialect(&self) -> AcceptedDialect;
    /// Disassembly version (for disassembly style serialization)
    fn disassembly_ver(&self) -> Option<usize>;
    /// Specifies whether code is being generated on behalf of an inner defun in
    /// the program.
    fn in_defun(&self) -> bool;
    /// Specifies whether the standard environment is injected (list, if etc).
    fn stdenv(&self) -> bool;
    /// Specifies whether certain basic optimizations are done during and after
    /// code generation.
    fn optimize(&self) -> bool;
    /// Specifies whether the frontend code is to be optimized before code
    /// generation.  This can simplify code from the user and decide on inlining
    /// of desugared forms.
    fn frontend_opt(&self) -> bool;
    /// Specifies whether forms not reachable at runtime are included in the
    /// resulting CompileForm.
    fn frontend_check_live(&self) -> bool;
    /// Specifies the shape of the environment to use.  This allows injection of
    /// the parent program's left environment when some form is compiled in the
    /// parent's context.
    fn start_env(&self) -> Option<Rc<SExp>>;
    /// Specifies the map of primitives provided during this compilation.
    fn prim_map(&self) -> Rc<HashMap<Vec<u8>, Rc<SExp>>>;
    /// Specifies the search paths we're carrying.
    fn get_search_paths(&self) -> Vec<String>;
    /// Specifies flags that were passed down to various consumers.  This is
    /// open ended for various purposes, such as diagnostics.
    fn diag_flags(&self) -> Rc<HashSet<usize>>;

    /// Set the dialect.
    fn set_dialect(&self, dialect: AcceptedDialect) -> Rc<dyn CompilerOpts>;
    /// Set search paths.
    fn set_search_paths(&self, dirs: &[String]) -> Rc<dyn CompilerOpts>;
    /// Set disassembly version for.
    fn set_disassembly_ver(&self, ver: Option<usize>) -> Rc<dyn CompilerOpts>;
    /// Set whether we're compiling on behalf of a defun.
    fn set_in_defun(&self, new_in_defun: bool) -> Rc<dyn CompilerOpts>;
    /// Set whether to inject the standard environment.
    fn set_stdenv(&self, new_stdenv: bool) -> Rc<dyn CompilerOpts>;
    /// Set whether to run codegen optimization.
    fn set_optimize(&self, opt: bool) -> Rc<dyn CompilerOpts>;
    /// Set whether to run frontend optimization.
    fn set_frontend_opt(&self, opt: bool) -> Rc<dyn CompilerOpts>;
    /// Set whether to filter out each HelperForm that isn't reachable at
    /// run time.
    fn set_frontend_check_live(&self, check: bool) -> Rc<dyn CompilerOpts>;
    /// Set the codegen object to be used downstream.
    fn set_code_generator(&self, new_compiler: PrimaryCodegen) -> Rc<dyn CompilerOpts>;
    /// Set the environment shape to assume.
    fn set_start_env(&self, start_env: Option<Rc<SExp>>) -> Rc<dyn CompilerOpts>;
    /// Set the primitive map in use so we can add custom primitives.
    fn set_prim_map(&self, new_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>) -> Rc<dyn CompilerOpts>;
    /// Set the flags this CompilerOpts holds.  Consumers can examine these.
    fn set_diag_flags(&self, new_flags: Rc<HashSet<usize>>) -> Rc<dyn CompilerOpts>;

    /// Using the search paths list we have, try to read a file by name,
    /// Returning the expanded path to the file and its content.
    fn read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr>;

    /// Given a parsed SExp, compile it as an independent program based on the
    /// settings given here.  The result is bare generated code.
    fn compile_program(
        &self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        sexp: Rc<SExp>,
        symbol_table: &mut HashMap<String, String>,
    ) -> Result<SExp, CompileErr>;
}

/// A trait that simplifies implementing one's own CompilerOpts personality.
/// This specifies to a CompilerOptsDelegator that this object contains a
/// CompilerOpts that it uses for most of what it does, allowing end users
/// to opt into a default implementation of all the methods via
/// CompilerOptsDelegator and override only what's desired.
pub trait HasCompilerOptsDelegation {
    /// Get this object's inner CompilerOpts.
    fn compiler_opts(&self) -> Rc<dyn CompilerOpts>;
    /// Call a function that updates this object's CompilerOpts and use the
    /// update our own object with the result.  Return the new wrapper.
    fn update_compiler_opts<F: FnOnce(Rc<dyn CompilerOpts>) -> Rc<dyn CompilerOpts>>(
        &self,
        f: F,
    ) -> Rc<dyn CompilerOpts>;

    // Defaults.
    fn override_filename(&self) -> String {
        self.compiler_opts().filename()
    }
    fn override_code_generator(&self) -> Option<PrimaryCodegen> {
        self.compiler_opts().code_generator()
    }
    fn override_dialect(&self) -> AcceptedDialect {
        self.compiler_opts().dialect()
    }
    fn override_disassembly_ver(&self) -> Option<usize> {
        self.compiler_opts().disassembly_ver()
    }
    fn override_in_defun(&self) -> bool {
        self.compiler_opts().in_defun()
    }
    fn override_stdenv(&self) -> bool {
        self.compiler_opts().stdenv()
    }
    fn override_optimize(&self) -> bool {
        self.compiler_opts().optimize()
    }
    fn override_frontend_opt(&self) -> bool {
        self.compiler_opts().frontend_opt()
    }
    fn override_frontend_check_live(&self) -> bool {
        self.compiler_opts().frontend_check_live()
    }
    fn override_start_env(&self) -> Option<Rc<SExp>> {
        self.compiler_opts().start_env()
    }
    fn override_prim_map(&self) -> Rc<HashMap<Vec<u8>, Rc<SExp>>> {
        self.compiler_opts().prim_map()
    }
    fn override_get_search_paths(&self) -> Vec<String> {
        self.compiler_opts().get_search_paths()
    }
    fn override_diag_flags(&self) -> Rc<HashSet<usize>> {
        self.compiler_opts().diag_flags()
    }

    fn override_set_dialect(&self, dialect: AcceptedDialect) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_dialect(dialect))
    }
    fn override_set_search_paths(&self, dirs: &[String]) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_search_paths(dirs))
    }
    fn override_set_disassembly_ver(&self, ver: Option<usize>) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_disassembly_ver(ver))
    }
    fn override_set_in_defun(&self, new_in_defun: bool) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_in_defun(new_in_defun))
    }
    fn override_set_stdenv(&self, new_stdenv: bool) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_stdenv(new_stdenv))
    }
    fn override_set_optimize(&self, opt: bool) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_optimize(opt))
    }
    fn override_set_frontend_opt(&self, opt: bool) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_frontend_opt(opt))
    }
    fn override_set_frontend_check_live(&self, check: bool) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_frontend_check_live(check))
    }
    fn override_set_code_generator(&self, new_compiler: PrimaryCodegen) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_code_generator(new_compiler))
    }
    fn override_set_start_env(&self, start_env: Option<Rc<SExp>>) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_start_env(start_env))
    }
    fn override_set_diag_flags(&self, flags: Rc<HashSet<usize>>) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_diag_flags(flags))
    }
    fn override_set_prim_map(
        &self,
        new_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    ) -> Rc<dyn CompilerOpts> {
        self.update_compiler_opts(|o| o.set_prim_map(new_map))
    }
    fn override_read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr> {
        self.compiler_opts().read_new_file(inc_from, filename)
    }
    fn override_compile_program(
        &self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        sexp: Rc<SExp>,
        symbol_table: &mut HashMap<String, String>,
    ) -> Result<SExp, CompileErr> {
        self.compiler_opts()
            .compile_program(allocator, runner, sexp, symbol_table)
    }
}

impl<T: HasCompilerOptsDelegation> CompilerOpts for T {
    // Defaults.
    fn filename(&self) -> String {
        self.override_filename()
    }
    fn code_generator(&self) -> Option<PrimaryCodegen> {
        self.override_code_generator()
    }
    fn dialect(&self) -> AcceptedDialect {
        self.override_dialect()
    }
    fn disassembly_ver(&self) -> Option<usize> {
        self.override_disassembly_ver()
    }
    fn in_defun(&self) -> bool {
        self.override_in_defun()
    }
    fn stdenv(&self) -> bool {
        self.override_stdenv()
    }
    fn optimize(&self) -> bool {
        self.override_optimize()
    }
    fn frontend_opt(&self) -> bool {
        self.override_frontend_opt()
    }
    fn frontend_check_live(&self) -> bool {
        self.override_frontend_check_live()
    }
    fn start_env(&self) -> Option<Rc<SExp>> {
        self.override_start_env()
    }
    fn prim_map(&self) -> Rc<HashMap<Vec<u8>, Rc<SExp>>> {
        self.override_prim_map()
    }
    fn get_search_paths(&self) -> Vec<String> {
        self.override_get_search_paths()
    }
    fn diag_flags(&self) -> Rc<HashSet<usize>> {
        self.override_diag_flags()
    }

    fn set_dialect(&self, dialect: AcceptedDialect) -> Rc<dyn CompilerOpts> {
        self.override_set_dialect(dialect)
    }
    fn set_search_paths(&self, dirs: &[String]) -> Rc<dyn CompilerOpts> {
        self.override_set_search_paths(dirs)
    }
    fn set_disassembly_ver(&self, ver: Option<usize>) -> Rc<dyn CompilerOpts> {
        self.override_set_disassembly_ver(ver)
    }
    fn set_in_defun(&self, new_in_defun: bool) -> Rc<dyn CompilerOpts> {
        self.override_set_in_defun(new_in_defun)
    }
    fn set_stdenv(&self, new_stdenv: bool) -> Rc<dyn CompilerOpts> {
        self.override_set_stdenv(new_stdenv)
    }
    fn set_optimize(&self, opt: bool) -> Rc<dyn CompilerOpts> {
        self.override_set_optimize(opt)
    }
    fn set_frontend_opt(&self, opt: bool) -> Rc<dyn CompilerOpts> {
        self.override_set_frontend_opt(opt)
    }
    fn set_frontend_check_live(&self, check: bool) -> Rc<dyn CompilerOpts> {
        self.override_set_frontend_check_live(check)
    }
    fn set_code_generator(&self, new_compiler: PrimaryCodegen) -> Rc<dyn CompilerOpts> {
        self.override_set_code_generator(new_compiler)
    }
    fn set_start_env(&self, start_env: Option<Rc<SExp>>) -> Rc<dyn CompilerOpts> {
        self.override_set_start_env(start_env)
    }
    fn set_prim_map(&self, new_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>) -> Rc<dyn CompilerOpts> {
        self.override_set_prim_map(new_map)
    }
    fn set_diag_flags(&self, new_flags: Rc<HashSet<usize>>) -> Rc<dyn CompilerOpts> {
        self.override_set_diag_flags(new_flags)
    }
    fn read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr> {
        self.override_read_new_file(inc_from, filename)
    }
    fn compile_program(
        &self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        sexp: Rc<SExp>,
        symbol_table: &mut HashMap<String, String>,
    ) -> Result<SExp, CompileErr> {
        self.override_compile_program(allocator, runner, sexp, symbol_table)
    }
}

/// Frontend uses this to accumulate frontend forms, used internally.
#[derive(Debug)]
pub struct ModAccum {
    pub loc: Srcloc,
    pub includes: Vec<IncludeDesc>,
    pub helpers: Vec<HelperForm>,
    pub exp_form: Option<CompileForm>,
}

/// A specification of a function call including elements useful for evaluation.
#[derive(Debug, Clone)]
pub struct CallSpec<'a> {
    pub loc: Srcloc,
    pub name: &'a [u8],
    pub args: &'a [Rc<BodyForm>],
    pub tail: Option<Rc<BodyForm>>,
    pub original: Rc<BodyForm>,
}

/// Raw callspec for use in codegen.
#[derive(Debug, Clone)]
pub struct RawCallSpec<'a> {
    pub loc: Srcloc,
    pub args: &'a [Rc<BodyForm>],
    pub tail: Option<Rc<BodyForm>>,
    pub original: Rc<BodyForm>,
}

/// A pair of arguments and an optional tail for function calls.  The tail is
/// a function tail given by a final &rest argument.
#[derive(Debug, Default, Clone)]
pub struct ArgsAndTail {
    pub args: Vec<Rc<BodyForm>>,
    pub tail: Option<Rc<BodyForm>>,
}

impl ModAccum {
    pub fn set_final(&self, c: &CompileForm) -> Self {
        ModAccum {
            loc: self.loc.clone(),
            includes: self.includes.clone(),
            helpers: self.helpers.clone(),
            exp_form: Some(c.clone()),
        }
    }

    pub fn add_include(&self, i: IncludeDesc) -> Self {
        let mut new_includes = self.includes.clone();
        new_includes.push(i);
        ModAccum {
            loc: self.loc.clone(),
            includes: new_includes,
            helpers: self.helpers.clone(),
            exp_form: self.exp_form.clone(),
        }
    }

    pub fn add_helper(&self, h: HelperForm) -> Self {
        let mut hs = self.helpers.clone();
        hs.push(h);

        ModAccum {
            loc: self.loc.clone(),
            includes: self.includes.clone(),
            helpers: hs,
            exp_form: self.exp_form.clone(),
        }
    }

    pub fn new(loc: Srcloc) -> ModAccum {
        ModAccum {
            loc,
            includes: Vec::new(),
            helpers: Vec::new(),
            exp_form: None,
        }
    }
}

impl CompileForm {
    /// Get the location of the compileform.
    pub fn loc(&self) -> Srcloc {
        self.loc.clone()
    }

    /// Express the contents as an SExp.  This SExp does not come with a keyword
    /// but starts at the arguments, since CompileForm objects are used in the
    /// encoding of several other types.
    pub fn to_sexp(&self) -> Rc<SExp> {
        let mut sexp_forms: Vec<Rc<SExp>> = self.helpers.iter().map(|x| x.to_sexp()).collect();
        sexp_forms.push(self.exp.to_sexp());

        Rc::new(SExp::Cons(
            self.loc.clone(),
            self.args.clone(),
            Rc::new(list_to_cons(self.loc.clone(), &sexp_forms)),
        ))
    }

    /// Given a set of helpers by name, remove them.
    pub fn remove_helpers(&self, names: &HashSet<Vec<u8>>) -> CompileForm {
        CompileForm {
            loc: self.loc.clone(),
            args: self.args.clone(),
            include_forms: self.include_forms.clone(),
            helpers: self
                .helpers
                .iter()
                .filter(|h| !names.contains(h.name()))
                .cloned()
                .collect(),
            exp: self.exp.clone(),
        }
    }

    /// Given a list of helpers, introduce them in this CompileForm, removing
    /// conflicting predecessors.
    pub fn replace_helpers(&self, helpers: &[HelperForm]) -> CompileForm {
        let mut new_names = HashSet::new();
        for h in helpers.iter() {
            new_names.insert(h.name());
        }
        let mut new_helpers: Vec<HelperForm> = self
            .helpers
            .iter()
            .filter(|h| !new_names.contains(h.name()))
            .cloned()
            .collect();
        new_helpers.append(&mut helpers.to_vec());

        CompileForm {
            loc: self.loc.clone(),
            include_forms: self.include_forms.clone(),
            args: self.args.clone(),
            helpers: new_helpers,
            exp: self.exp.clone(),
        }
    }
}

pub fn generate_defmacro_sexp(mac: &DefmacData) -> Rc<SExp> {
    if mac.advanced {
        Rc::new(SExp::Cons(
            mac.loc.clone(),
            Rc::new(SExp::atom_from_string(mac.loc.clone(), "defmac")),
            Rc::new(SExp::Cons(
                mac.loc.clone(),
                Rc::new(SExp::atom_from_vec(mac.nl.clone(), &mac.name)),
                Rc::new(SExp::Cons(
                    mac.loc.clone(),
                    mac.args.clone(),
                    Rc::new(SExp::Cons(
                        mac.loc.clone(),
                        mac.program.exp.to_sexp(),
                        Rc::new(SExp::Nil(mac.loc.clone())),
                    )),
                )),
            )),
        ))
    } else {
        Rc::new(SExp::Cons(
            mac.loc.clone(),
            Rc::new(SExp::atom_from_string(mac.loc.clone(), "defmacro")),
            Rc::new(SExp::Cons(
                mac.loc.clone(),
                Rc::new(SExp::atom_from_vec(mac.nl.clone(), &mac.name)),
                mac.program.to_sexp(),
            )),
        ))
    }
}

impl HelperForm {
    /// Get a reference to the HelperForm's name.
    pub fn name(&self) -> &Vec<u8> {
        match self {
            HelperForm::Defconstant(defc) => &defc.name,
            HelperForm::Defmacro(mac) => &mac.name,
            HelperForm::Defun(_, defun) => &defun.name,
        }
    }

    /// Get the location of the HelperForm's name.
    pub fn name_loc(&self) -> &Srcloc {
        match self {
            HelperForm::Defconstant(defc) => &defc.nl,
            HelperForm::Defmacro(mac) => &mac.nl,
            HelperForm::Defun(_, defun) => &defun.nl,
        }
    }

    /// Return a general location for the whole HelperForm.
    pub fn loc(&self) -> Srcloc {
        match self {
            HelperForm::Defconstant(defc) => defc.loc.clone(),
            HelperForm::Defmacro(mac) => mac.loc.clone(),
            HelperForm::Defun(_, defun) => defun.loc.clone(),
        }
    }

    /// Convert the HelperForm to an SExp.  These render into a form that can
    /// be re-parsed if needed.
    pub fn to_sexp(&self) -> Rc<SExp> {
        match self {
            HelperForm::Defconstant(defc) => match defc.kind {
                ConstantKind::Simple => Rc::new(list_to_cons(
                    defc.loc.clone(),
                    &[
                        Rc::new(SExp::atom_from_string(defc.loc.clone(), "defconstant")),
                        Rc::new(SExp::atom_from_vec(defc.loc.clone(), &defc.name)),
                        defc.body.to_sexp(),
                    ],
                )),
                ConstantKind::Complex => Rc::new(list_to_cons(
                    defc.loc.clone(),
                    &[
                        Rc::new(SExp::atom_from_string(defc.loc.clone(), "defconst")),
                        Rc::new(SExp::atom_from_vec(defc.loc.clone(), &defc.name)),
                        defc.body.to_sexp(),
                    ],
                )),
            },
            HelperForm::Defmacro(mac) => generate_defmacro_sexp(mac),
            HelperForm::Defun(inline, defun) => {
                let di_string = "defun-inline".to_string();
                let d_string = "defun".to_string();
                Rc::new(list_to_cons(
                    defun.loc.clone(),
                    &[
                        Rc::new(SExp::atom_from_string(
                            defun.loc.clone(),
                            if *inline { &di_string } else { &d_string },
                        )),
                        Rc::new(SExp::atom_from_vec(defun.nl.clone(), &defun.name)),
                        defun.args.clone(),
                        defun.body.to_sexp(),
                    ],
                ))
            }
        }
    }
}

fn compose_lambda_serialized_form(ldata: &LambdaData) -> Rc<SExp> {
    let lambda_kw = Rc::new(SExp::Atom(ldata.loc.clone(), b"lambda".to_vec()));
    let amp_kw = Rc::new(SExp::Atom(ldata.loc.clone(), b"&".to_vec()));
    let arguments = if truthy(ldata.capture_args.clone()) {
        Rc::new(SExp::Cons(
            ldata.loc.clone(),
            Rc::new(SExp::Cons(
                ldata.loc.clone(),
                amp_kw,
                ldata.capture_args.clone(),
            )),
            ldata.args.clone(),
        ))
    } else {
        ldata.args.clone()
    };
    let rest_of_body = Rc::new(SExp::Cons(
        ldata.loc.clone(),
        ldata.body.to_sexp(),
        Rc::new(SExp::Nil(ldata.loc.clone())),
    ));

    Rc::new(SExp::Cons(
        ldata.loc.clone(),
        lambda_kw,
        Rc::new(SExp::Cons(ldata.loc.clone(), arguments, rest_of_body)),
    ))
}

fn compose_let(marker: &[u8], letdata: &LetData) -> Rc<SExp> {
    let translated_bindings: Vec<Rc<SExp>> = letdata.bindings.iter().map(|x| x.to_sexp()).collect();
    let bindings_cons = list_to_cons(letdata.loc.clone(), &translated_bindings);
    let translated_body = letdata.body.to_sexp();
    let kw_loc = letdata.kw.clone().unwrap_or_else(|| letdata.loc.clone());
    Rc::new(SExp::Cons(
        letdata.loc.clone(),
        Rc::new(SExp::Atom(kw_loc, marker.to_vec())),
        Rc::new(SExp::Cons(
            letdata.loc.clone(),
            Rc::new(bindings_cons),
            Rc::new(SExp::Cons(
                letdata.loc.clone(),
                translated_body,
                Rc::new(SExp::Nil(letdata.loc.clone())),
            )),
        )),
    ))
}

fn compose_assign(letdata: &LetData) -> Rc<SExp> {
    let mut result = Vec::new();
    let kw_loc = letdata.kw.clone().unwrap_or_else(|| letdata.loc.clone());
    result.push(Rc::new(SExp::Atom(kw_loc, b"assign".to_vec())));
    for b in letdata.bindings.iter() {
        // Binding pattern
        match &b.pattern {
            BindingPattern::Name(v) => {
                result.push(Rc::new(SExp::Atom(b.nl.clone(), v.to_vec())));
            }
            BindingPattern::Complex(c) => {
                result.push(c.clone());
            }
        }

        // Binding body.
        result.push(b.body.to_sexp());
    }

    result.push(letdata.body.to_sexp());
    Rc::new(enlist(letdata.loc.clone(), &result))
}

fn get_let_marker_text(kind: &LetFormKind, letdata: &LetData) -> Vec<u8> {
    match (kind, letdata.inline_hint.as_ref()) {
        (LetFormKind::Sequential, _) => b"let*".to_vec(),
        (LetFormKind::Parallel, _) => b"let".to_vec(),
        (LetFormKind::Assign, Some(LetFormInlineHint::Inline(_))) => b"assign-inline".to_vec(),
        (LetFormKind::Assign, Some(LetFormInlineHint::NonInline(_))) => b"assign-lambda".to_vec(),
        (LetFormKind::Assign, _) => b"assign".to_vec(),
    }
}

impl BodyForm {
    /// Get the general location of the BodyForm.
    pub fn loc(&self) -> Srcloc {
        match self {
            BodyForm::Let(_, letdata) => letdata.loc.clone(),
            BodyForm::Quoted(a) => a.loc(),
            BodyForm::Call(loc, _, _) => loc.clone(),
            BodyForm::Value(a) => a.loc(),
            BodyForm::Mod(kl, program) => kl.ext(&program.loc),
            BodyForm::Lambda(ldata) => ldata.loc.ext(&ldata.body.loc()),
        }
    }

    /// Convert the expression to its SExp form.  These should be reparsable but
    /// may change when desugaring requires it if re-serialization is needed
    /// afterward.
    pub fn to_sexp(&self) -> Rc<SExp> {
        match self {
            BodyForm::Let(LetFormKind::Assign, letdata) => compose_assign(letdata),
            BodyForm::Let(kind, letdata) => {
                let marker = get_let_marker_text(kind, letdata);
                compose_let(&marker, letdata)
            }
            BodyForm::Quoted(body) => Rc::new(SExp::Cons(
                body.loc(),
                Rc::new(SExp::atom_from_string(body.loc(), "q")),
                Rc::new(body.clone()),
            )),
            BodyForm::Value(body) => Rc::new(body.clone()),
            BodyForm::Call(loc, exprs, tail) => {
                let mut converted: Vec<Rc<SExp>> = exprs.iter().map(|x| x.to_sexp()).collect();
                if let Some(t) = tail.as_ref() {
                    converted.push(Rc::new(SExp::Atom(t.loc(), "&rest".as_bytes().to_vec())));
                    converted.push(t.to_sexp());
                }
                Rc::new(list_to_cons(loc.clone(), &converted))
            }
            BodyForm::Mod(loc, program) => Rc::new(SExp::Cons(
                loc.clone(),
                Rc::new(SExp::Atom(loc.clone(), b"mod".to_vec())),
                program.to_sexp(),
            )),
            BodyForm::Lambda(ldata) => compose_lambda_serialized_form(ldata),
        }
    }
}

// Note: in cfg(test), this will not be part of the finished binary.
// Also: not a test in itself, just named test so for at least some readers,
// its association with test infrastructure will be apparent.
#[cfg(test)]
fn test_parse_bodyform_to_frontend(bf: &str) {
    let name = "*test*";
    let loc = Srcloc::start(name);
    let opts = Rc::new(DefaultCompilerOpts::new(name));
    let parsed = parse_sexp(loc, bf.bytes()).expect("should parse");
    let bodyform = compile_bodyform(opts, parsed[0].clone()).expect("should compile");
    assert_eq!(bodyform.to_sexp(), parsed[0]);
}

// Inline unit tests for sexp serialization.
#[test]
fn test_mod_serialize_regular_mod() {
    test_parse_bodyform_to_frontend("(mod (X) (+ X 1))");
}

#[test]
fn test_mod_serialize_simple_lambda() {
    test_parse_bodyform_to_frontend("(lambda (X) (+ X 1))");
}

impl Binding {
    /// Express the binding as it would be used in a let form.
    pub fn to_sexp(&self) -> Rc<SExp> {
        let pat = match &self.pattern {
            BindingPattern::Name(name) => Rc::new(SExp::atom_from_vec(self.loc.clone(), name)),
            BindingPattern::Complex(sexp) => sexp.clone(),
        };
        Rc::new(SExp::Cons(
            self.loc.clone(),
            pat,
            Rc::new(SExp::Cons(
                self.loc.clone(),
                self.body.to_sexp(),
                Rc::new(SExp::Nil(self.loc.clone())),
            )),
        ))
    }

    /// Get the general location of the binding.
    pub fn loc(&self) -> Srcloc {
        self.loc.clone()
    }
}

impl CompiledCode {
    /// Get the general location the code was compiled from.
    pub fn loc(&self) -> Srcloc {
        self.0.clone()
    }
}

impl PrimaryCodegen {
    pub fn add_constant(&self, name: &[u8], value: Rc<SExp>) -> Self {
        let mut codegen_copy = self.clone();
        codegen_copy.constants.insert(name.to_owned(), value);
        codegen_copy
    }

    pub fn add_tabled_constant(&self, name: &[u8], value: Rc<SExp>) -> Self {
        let mut codegen_copy = self.clone();
        codegen_copy.tabled_constants.insert(name.to_owned(), value);
        codegen_copy
    }

    pub fn add_macro(&self, name: &[u8], value: Rc<SExp>) -> Self {
        let mut codegen_copy = self.clone();
        codegen_copy.macros.insert(name.to_owned(), value);
        codegen_copy
    }

    pub fn add_inline(&self, name: &[u8], value: &InlineFunction) -> Self {
        let mut codegen_copy = self.clone();
        codegen_copy.inlines.insert(name.to_owned(), value.clone());
        codegen_copy
    }

    pub fn add_defun(&self, name: &[u8], args: Rc<SExp>, value: DefunCall, left_env: bool) -> Self {
        let mut codegen_copy = self.clone();
        codegen_copy.defuns.insert(name.to_owned(), value.clone());
        let hash = sha256tree(value.code);
        let hash_str = Bytes::new(Some(BytesFromType::Raw(hash))).hex();
        let name = Bytes::new(Some(BytesFromType::Raw(name.to_owned()))).decode();
        codegen_copy.function_symbols.insert(hash_str.clone(), name);
        if left_env {
            codegen_copy
                .function_symbols
                .insert(format!("{hash_str}_left_env"), "1".to_string());
        }
        codegen_copy
            .function_symbols
            .insert(format!("{hash_str}_arguments"), args.to_string());
        codegen_copy
    }

    pub fn set_env(&self, env: Rc<SExp>) -> Self {
        let mut codegen_copy = self.clone();
        codegen_copy.env = env;
        codegen_copy
    }
}

pub fn with_heading(l: Srcloc, name: &str, body: Rc<SExp>) -> SExp {
    SExp::Cons(l.clone(), Rc::new(SExp::atom_from_string(l, name)), body)
}

pub fn cons_of_string_map<X>(
    l: Srcloc,
    cvt_body: &dyn Fn(&X) -> Rc<SExp>,
    map: &HashMap<Vec<u8>, X>,
) -> SExp {
    // Thanks: https://users.rust-lang.org/t/sort-hashmap-data-by-keys/37095/3
    let mut v: Vec<_> = map.iter().collect();
    v.sort_by(|x, y| x.0.cmp(y.0));

    let sorted_converted: Vec<Rc<SExp>> = v
        .iter()
        .map(|x| {
            Rc::new(SExp::Cons(
                l.clone(),
                Rc::new(SExp::QuotedString(l.clone(), b'\"', x.0.to_vec())),
                Rc::new(SExp::Cons(
                    l.clone(),
                    cvt_body(x.1),
                    Rc::new(SExp::Nil(l.clone())),
                )),
            ))
        })
        .collect();

    list_to_cons(l, &sorted_converted)
}

pub fn map_m<T, U, E, F>(mut f: F, list: &[T]) -> Result<Vec<U>, E>
where
    F: FnMut(&T) -> Result<U, E>,
{
    let mut result = Vec::new();
    for e in list {
        let val = f(e)?;
        result.push(val);
    }
    Ok(result)
}

pub fn map_m_reverse<T, U, E, F>(mut f: F, list: &[T]) -> Result<Vec<U>, E>
where
    F: FnMut(&T) -> Result<U, E>,
{
    let mut result = Vec::new();
    for e in list {
        let val = f(e)?;
        result.push(val);
    }
    Ok(result.into_iter().rev().collect())
}

pub fn fold_m<R, T, E>(f: &dyn Fn(&R, &T) -> Result<R, E>, start: R, list: &[T]) -> Result<R, E> {
    let mut res: R = start;
    for elt in list.iter() {
        res = f(&res, elt)?;
    }
    Ok(res)
}

pub fn join_vecs_to_string(sep: Vec<u8>, vecs: &[Vec<u8>]) -> String {
    let mut s = Vec::new();
    let mut comma = Vec::new();

    for elt in vecs {
        s.append(&mut comma.clone());
        s.append(&mut elt.to_vec());
        if comma.is_empty() {
            comma.clone_from(&sep);
        }
    }

    decode_string(&s)
}
