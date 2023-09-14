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
pub mod gensym;
mod inline;
pub mod optimize;
pub mod preprocessor;
pub mod prims;
pub mod rename;
pub mod repl;
pub mod runtypes;
pub mod sexp;
pub mod srcloc;
pub mod stackvisit;
pub mod usecheck;

use clvmr::allocator::Allocator;
use std::collections::HashMap;
use std::mem::swap;
use std::rc::Rc;

use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

/// An object which represents the standard set of mutable items passed down the
/// stack when compiling chialisp.
pub struct BasicCompileContext {
    pub allocator: Allocator,
    pub runner: Rc<dyn TRunProgram>,
    pub symbols: HashMap<String, String>,
}

impl BasicCompileContext {
    /// Get a mutable allocator reference from this compile context. The
    /// allocator is used any time we need to execute pure CLVM operators, such
    /// as when evaluating macros or constant folding any chialisp expression.
    fn allocator(&mut self) -> &mut Allocator {
        &mut self.allocator
    }

    /// Get the runner this compile context carries. This is used with the
    /// allocator above to execute pure CLVM when needed either on behalf of a
    /// macro or constant folding.
    fn runner(&self) -> Rc<dyn TRunProgram> {
        self.runner.clone()
    }

    /// Get the mutable symbol store this compile context carries. During
    /// compilation, the compiler records the relationships between objects in
    /// the source code and emitted CLVM expressions, along with other useful
    /// information.
    ///
    /// There are times when we're in a subcompile (such as mod expressions when
    /// the compile context needs to do swap in or out symbols or transform them
    /// on behalf of the child.
    fn symbols(&mut self) -> &mut HashMap<String, String> {
        &mut self.symbols
    }

    /// Given allocator, runner and symbols, move the mutable objects into this
    /// BasicCompileContext so it can own them and pass a single mutable
    /// reference to itself down the stack. This allows these objects to be
    /// queried and used by appropriate machinery.
    pub fn new(
        allocator: Allocator,
        runner: Rc<dyn TRunProgram>,
        symbols: HashMap<String, String>,
    ) -> Self {
        BasicCompileContext {
            allocator,
            runner,
            symbols,
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
    ) -> Self {
        let bcc = BasicCompileContext {
            allocator: Allocator::new(),
            runner,
            symbols: HashMap::new(),
        };
        let mut wrapper = CompileContextWrapper {
            allocator,
            symbols,
            context: bcc,
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
    }
}

/// Drop CompileContextWrapper reverts the contained objects back to the ones
/// owned by the caller.
impl<'a> Drop for CompileContextWrapper<'a> {
    fn drop(&mut self) {
        self.switch();
    }
}
