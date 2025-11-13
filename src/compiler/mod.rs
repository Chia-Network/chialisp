/// Chialisp debugging.
pub mod cldb;
pub mod cldb_hierarchy;
/// CLVM running.
pub mod clvm;
mod codegen;
/// CompilerOpts which is the main holder of toplevel compiler state.
#[allow(clippy::module_inception)]
pub mod compiler;
/// Types used by compilation, mainly frontend directed, including.
/// - BodyForm - The type of frontend expressions.
/// - CompileErr - The type of errors from compilation.
/// - CompileForm - The type of finished (mod ) forms before code generation.
/// - HelperForm - The type of declarations like macros, constants and functions.
pub mod comptypes;
pub mod debug;
/// Utilities for chialisp dialect choice
pub mod dialect;
pub mod evaluate;
pub mod frontend;
#[cfg(any(test, feature = "fuzz"))]
pub mod fuzz;
pub mod gensym;
mod inline;
mod lambda;
pub mod optimize;
pub mod preprocessor;
pub mod prims;
pub mod rename;
pub mod repl;
pub mod resolve;
pub mod runtypes;
pub mod sexp;
pub mod srcloc;
pub mod stackvisit;
pub mod usecheck;

use clvmr::allocator::Allocator;
use std::collections::HashMap;
use std::mem::swap;
use std::rc::Rc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic;

use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompilerOpts, DefunData, HelperForm, IncludeDesc,
    PrimaryCodegen,
};
use crate::compiler::optimize::Optimization;
use crate::compiler::sexp::SExp;

lazy_static! {
    pub static ref INDENT: AtomicUsize = AtomicUsize::new(0);
}

/// An object which represents the standard set of mutable items passed down the
/// stack when compiling chialisp.
pub struct BasicCompileContext {
    pub allocator: Allocator,
    pub runner: Rc<dyn TRunProgram>,
    pub symbols: HashMap<String, String>,
    pub optimizer: Box<dyn Optimization>,
    pub includes: Vec<IncludeDesc>,
}

impl BasicCompileContext {
    /// Get a mutable allocator reference from this compile context. The
    /// allocator is used any time we need to execute pure CLVM operators, such
    /// as when evaluating macros or constant folding any chialisp expression.
    pub fn allocator(&mut self) -> &mut Allocator {
        &mut self.allocator
    }

    /// Get the runner this compile context carries. This is used with the
    /// allocator above to execute pure CLVM when needed either on behalf of a
    /// macro or constant folding.
    pub fn runner(&self) -> Rc<dyn TRunProgram> {
        self.runner.clone()
    }

    /// Get the list of includes traversed while processing the current compile.
    pub fn includes(&mut self) -> &mut Vec<IncludeDesc> {
        &mut self.includes
    }

    /// Get the mutable symbol store this compile context carries. During
    /// compilation, the compiler records the relationships between objects in
    /// the source code and emitted CLVM expressions, along with other useful
    /// information.
    ///
    /// There are times when we're in a subcompile (such as mod expressions when
    /// the compile context needs to do swap in or out symbols or transform them
    /// on behalf of the child.
    pub fn symbols(&mut self) -> &mut HashMap<String, String> {
        &mut self.symbols
    }

    /// Called after frontend parsing and preprocessing when we have a complete
    /// picture of the user's intended semantics.
    pub fn frontend_optimization(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        cf: CompileForm,
    ) -> Result<CompileForm, CompileErr> {
        let runner = self.runner.clone();
        self.optimizer
            .frontend_optimization(&mut self.allocator, runner, opts, cf)
    }

    pub fn post_desugar_optimization(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        cf: CompileForm,
    ) -> Result<CompileForm, CompileErr> {
        let runner = self.runner.clone();
        self.optimizer
            .post_desugar_optimization(&mut self.allocator, runner, opts, cf)
    }

    /// Shrink the program prior to generating the final environment map and
    /// doing other codegen tasks.  This also serves as a tree-shaking pass.
    pub fn start_of_codegen_optimization(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        to_optimize: StartOfCodegenOptimization,
    ) -> Result<StartOfCodegenOptimization, CompileErr> {
        let runner = self.runner.clone();
        self.optimizer
            .start_of_codegen_optimization(&mut self.allocator, runner, opts, to_optimize)
    }

    /// Note: must take measures to ensure that the symbols are changed along
    /// with any code that's changed.  It's likely better to do optimizations
    /// at other stages, such as post_codegen_function_optimize.
    pub fn post_codegen_output_optimize(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        generated: SExp,
    ) -> Result<SExp, CompileErr> {
        self.optimizer.post_codegen_output_optimize(opts, generated)
    }

    /// Called when a full macro program optimization is used.
    pub fn macro_optimization(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        code: Rc<SExp>,
    ) -> Result<Rc<SExp>, CompileErr> {
        self.optimizer
            .macro_optimization(&mut self.allocator, self.runner.clone(), opts, code)
    }

    /// Called to transform a defun before generating code from it.
    /// Returns a new bodyform.
    pub fn pre_codegen_function_optimize(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        codegen: &PrimaryCodegen,
        defun: &DefunData,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        self.optimizer.defun_body_optimization(
            &mut self.allocator,
            self.runner.clone(),
            opts,
            codegen,
            defun,
        )
    }

    /// Called to transform the function body after code generation.
    pub fn post_codegen_function_optimize(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        helper: Option<&HelperForm>,
        code: Rc<SExp>,
    ) -> Result<Rc<SExp>, CompileErr> {
        self.optimizer.post_codegen_function_optimize(
            &mut self.allocator,
            self.runner.clone(),
            opts,
            helper,
            code,
        )
    }

    /// Call in final_codegen to get the final main bodyform to generate
    /// code from.
    pub fn pre_final_codegen_optimize(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        codegen: &PrimaryCodegen,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        self.optimizer.pre_final_codegen_optimize(
            &mut self.allocator,
            self.runner.clone(),
            opts,
            codegen,
        )
    }

    /// Given allocator, runner and symbols, move the mutable objects into this
    /// BasicCompileContext so it can own them and pass a single mutable
    /// reference to itself down the stack. This allows these objects to be
    /// queried and used by appropriate machinery.
    pub fn new(
        allocator: Allocator,
        runner: Rc<dyn TRunProgram>,
        symbols: HashMap<String, String>,
        optimizer: Box<dyn Optimization>,
        includes: Vec<IncludeDesc>,
    ) -> Self {
        BasicCompileContext {
            allocator,
            runner,
            symbols,
            optimizer,
            includes,
        }
    }
}

/// A wrapper that owns a BasicCompileContext and remembers a mutable reference
/// to an allocator and symbols.  It is used as a container to swap out these
/// objects for new ones used in an inner compile context.  This is used when
/// a subcompile occurs such as when a macro is compiled to CLVM to be executed
/// or an inner mod is compiled.
pub struct CompileContextWrapper<'a> {
    pub allocator: &'a mut Allocator,
    pub symbols: &'a mut HashMap<String, String>,
    pub includes: &'a mut Vec<IncludeDesc>,
    pub context: BasicCompileContext,
}

impl<'a> CompileContextWrapper<'a> {
    /// Given an allocator, runner and symbols, hold the mutable references from
    /// the code above, swapping content into a new BasicCompileContext this
    /// object contains.
    ///
    /// The new and drop methods both rely on the object's private 'switch' method
    /// which swaps the mutable reference to allocator and symbols that the caller
    /// holds with the new empty allocator and hashmap held by the inner
    /// BasicCompileContext.  This allows us to pin the mutable references here,
    /// ensuring that this object is the only consumer of these objects when in
    /// use, while allowing a new BasicCompileContext to be passed down.  The user
    /// may inspect, copy, modify etc the inner context before allowing the
    /// CompileContextWrapper object to be dropped, which will put the modified
    /// objects back in the mutable references given by the user.
    ///
    /// This object does more in the current (nightly) code, such as carrying the
    /// optimizer, which is modified when an inner compile has a different sigil
    /// and must be optimized differently.
    pub fn new(
        allocator: &'a mut Allocator,
        runner: Rc<dyn TRunProgram>,
        symbols: &'a mut HashMap<String, String>,
        optimizer: Box<dyn Optimization>,
        includes: &'a mut Vec<IncludeDesc>,
    ) -> Self {
        let bcc = BasicCompileContext {
            allocator: Allocator::new(),
            runner,
            symbols: HashMap::new(),
            optimizer,
            includes: Vec::new(),
        };
        let mut wrapper = CompileContextWrapper {
            allocator,
            symbols,
            context: bcc,
            includes,
        };
        wrapper.switch();
        wrapper
    }

    pub fn from_context(
        context: &'a mut BasicCompileContext,
        symbols: &'a mut HashMap<String, String>,
    ) -> Self {
        let runner = context.runner();
        let optimizer = context.optimizer.duplicate();
        let bcc = BasicCompileContext {
            allocator: Allocator::new(),
            runner,
            symbols: HashMap::new(),
            optimizer,
            includes: Vec::new(),
        };
        let mut wrapper = CompileContextWrapper {
            allocator: &mut context.allocator,
            symbols,
            context: bcc,
            includes: &mut context.includes,
        };
        wrapper.switch();
        wrapper
    }

    /// Swap allocator and symbols with the ones in self.context.  This has the
    /// effect of making the inner context hold the same information that would
    /// have been passed down in these members had it come from the caller's
    /// perspective.  Useful when compile context has more fields and needs
    /// to change for a consumer down the stack.
    fn switch(&mut self) {
        swap(self.allocator, &mut self.context.allocator);
        swap(self.symbols, &mut self.context.symbols);
        swap(self.includes, &mut self.context.includes);
    }
}

/// Drop CompileContextWrapper reverts the contained objects back to the ones
/// owned by the caller.
impl Drop for CompileContextWrapper<'_> {
    fn drop(&mut self) {
        self.switch();
    }
}

/// Describes the unique inputs and outputs available at the start of code
/// generation.
#[derive(Debug, Clone)]
pub struct StartOfCodegenOptimization {
    program: CompileForm,
    code_generator: PrimaryCodegen,
}

struct Indent {
    indent: usize
}

impl Default for Indent {
    fn default() -> Indent {
        let indent = INDENT.fetch_add(1, atomic::Ordering::SeqCst);
        Indent { indent }
    }
}

impl Drop for Indent {
    fn drop(&mut self) {
        INDENT.fetch_sub(1, atomic::Ordering::SeqCst);
    }
}

impl std::fmt::Display for Indent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        for i in 0..self.indent {
            formatter.write_str(" ")?;
        }

        Ok(())
    }
}
