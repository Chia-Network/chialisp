Compiler theory of operation
==

Basic structure and where to look for specific things
--

_About compilers in general_

What's been suggested is an overview of code compilation overall, which includes
a good number of elements that are common or where there are a limited number of
useful choices.

Most code production in compilers starts with a user editing one or more source
files in a text editor.  These text files are usually stored in files on disk
in a place accessible to the program or programs that perform compilation, which
really just means reading the source text in some way and writing out code in
another language in some way, although most of the time the terms 'translation'
or 'transpilation' are used for when the other target is also a language people
consider to be high enough level that it's considered productive to write directly
by humans.  'Compilation' is normally used when the target is a language that
is complex and less accessible.  There are definitely overlaps, as a good number
of people write assembler for smaller CPUs more or less as a high level language
and a good many compilers treat javascript purely as a VM for executing code on.
In the case of chialisp, 'chialisp' can be thought of a mid-level language in the
vein of C++ and the target is CLVM, the virtual machine that chia nodes use to
evaluate puzzles.  CLVM is quite unergonomic to write by hand given the lack
of the ability to name things and the need to compute numeric paths through
CLVM values (more about this later) to do useful work.

So there's really only a few choices a compiler can make in structural terms:

- Read all inputs, write output (golang)
- Read input a bit at a time and possibly produce output a bit at a time (C)

All chialisp compilers I'm aware of read the whole input first.  Like C, chialisp
has a textual include file mechanism which allows the programmer to name a file
to include from a set of specified include paths.  The name given is expected to
be a literal fragment of a filename (as in C).  When each include file is found,
a single form is read from it, and that is expected to take the form of a list
of chialisp toplevel declarations and those are treated as though they appear
in the current program.  Chialisp forms are allowed to appear in any order in
the source files they appear in.

Structurally, compilers tend to work in passes, even if on only part of the input.
In C, where the main abstraction is "functions" in the C sense, each function is
treated conceptually separately and code is generated for it.  In chialisp, the
situation is similar; each toplevel function is value that must exist as a quoted
CLVM value in the program's invariant environment (the word "invariant" is used
here because CLVM code generation generally encodes paths into this information
into multiple parts of the code and because of this the representation of this
information is consistent throughout the program's run).

The classic chialisp compiler does not actually work in usefully separable passes.
It produces a CLVM expression including a request to run a special primitive (where
"primitive" here means an operator installed in the CLVM run time system which is
implemented in "foreign" (non-CLVM) code) whose function is to produce one step of
chialisp compilation and wraps this in a CLVM primitive whose purpose is to perform
optimization and constant folding.  These are installed in stage_2 (more on this
later).  In classic chialisp, compilation is actually performed by constant
folding, as that is what causes a CLVM primitive ("com" here) to be evaluated to
a value when the argument is constant ("constant folding" refers to the compile
time process of taking expression involving constants and computing their results).
Because of this arrangement, there is only one "pass" (where "pass" refers to a
traversal of the complete program in some representation to transform it into
another representation, even when both are internal to the compilation process),
because the only time when a guarantee is given that the entire S-expression
("s-expression" refers to the kinds of values held in lisp and lisp-like languages)
held during compilation in the classic compiler is expected to be completely 
transformed into CLVM.

The modern compiler does have distinct phases of compilation.

- The first file is completely parsed, and all include files are also parsed and
  incorporated before any "frontend" work is done (a compiler "frontend" usually
  refers to the part of compilation that sits directly after the user's source code
  is parsed.  Usually, for ergonomic reasons, there are parts of the language the
  user writes that don't translate directly to data structures used to generate
  code, therefore "frontend" processing translate the user's text into some more
  convenient representation for further processing).  Preprocessing in the new
  compiler takes place in src/compiler/preprocess.rs, function preprocess.  This
  is performed currently at the beginning of the compiler "frontend" but it can
  (and probably should) be broken out into a pipeline at the same level as the
  frontend eventually.  You can find the entry into the compiler frontend in
  src/compiler/frontend.rs, function frontend.  It calls preprocess at line
  718 (currently) in frontend_start.
  
- The compiler frontend produces a kind of "intermediate representation" (where
  "intermediate representation" refers to a data structure built during compilation
  that is not intended to be used apart from that compiler and that contains some
  essential or extract meaning from the source program that relates to some aspect
  of the program's meaning, code generation or some other process the compiler
  must perform for some reason).  The main one used by the modern compiler is
  a CompileForm, which contains a complete representation of the input program
  expressed in a set of HelperForm objects and a BodyForm object.  CompileForm is
  declared at line 255 (currently) of src/compiler/comptypes.rs.  Its purpose to
  to provide a single object that encapsulates the entire meaning of a chialisp
  program.  It is the only thing that is passed on to code generation.
  HelperForm is declared slightly earlier (as logically CompileForm depends on it)
  and concerns representing the kinds of declarations ("declaration" here refers
  to a part of a user's program where some reusable part of the program is given
  a name.  In many languages, it is synonomous or at least co-located with a
  "definition", which in programming languages refers to the site where a named
  object in the universe of the user's program is given some kind of concrete
  form).  BodyForm is defined slightly earlier yet (as HelperForm depends on it),
  at line 122 (currently) in src/compiler/comptypes.rs and contains any kind of
  expression the chialisp language allows.  Because the frontend is still in the
  process of collecting information when it converts the user's code into these
  data structures, it does not neccessarily know (for example) what function is
  being referred to by a function call, so it does not make any claim that the
  program is self-consistent exiting the compiler frontend, although at the
  end of that pass it's possible to make additional checks (for example, the
  use checker (src/compiler/usecheck.rs) consumers a CompileForm yielded from
  the compiler frontend in order to check for unused mod arguments.
  
- The next phase of compilation and logical "pass" is "desugaring" (where
  "desugaring" refers to the process of translating high level intentions
  signalled by the user and translated into the frontend's intermediate
  representation of the program into a program with fewer or more targeted
  elements that relate more closely to a part of the compilation process that
  takes place chronologically later and typically is conceptually
  targeting a part of compilation that relates more closely to the final
  environment code is being generated for).  It's intended that desugaring
  steps will be moved out of codegen to be done fully first, but it counts
  as a full compiler pass because it is completed before any code generation
  activities take place.
  
- After desugaring, a frontend optimizer is intended to go in.  This is being
  worked on in a branch.  The frontend optimizer performs higher level
  transformations on the code that result is simpler or smaller code that has
  the same meaning as the original, but in a form that faclitates better code
  generation.  An example from chialisp is noting when a stack of (f and (r
  primitive invocations exist and translating them into a numeric path, saving
  space.
  
- The final pass is code generation proper.  The "left environment" shape is
  produced ("left environment" here refers to the invariant part of the CLVM
  environment which is given to the program when its main expression is run,
  providing a consistent pool of sibling functions and constant data that the
  program is able to draw from).
  The process and concerns of code generation are described in better detail
  below, but the result is SExp, the data structure that represents S-expressions
  in this compiler, defined at 33 (presently) of src/compiler/sexp.rs.  In both
  classic and modern compilers, the representation of parsed source code and
  emitted code are both their respective views of s-expressions.  The full
  power of the modern S-expression isn't required in the emitted CLVM code but
  it's convenient because it carries some useful information from the user's
  perspective; it's Srcloc (defined in src/compiler/srcloc.rs) contains a record
  of what part of the source code it was generated from.  These are collected
  into the "symbols" (where "symbols" generally refers to a parallel set of
  information about a program regarding user-understandable names and locations
  that can be associated with landmarks in the generated code).  In the case of
  chialisp, landmarks are identified by "sha256tree" (where "sha256tree" refers
  to a standard process by which CLVM values are given a fixed-length hash
  identifier based on their content that has a nearly 0 possibility of
  generating collisions).  Because of this, the symbols can refer to code by
  sha256tree hash and give names to sites in the generated code.
  
_Dive into the code from start to compiler_

The code here and the chialisp compiler has a history for its life so far.  It
started in python, was ported very faithfully to typescript and then the typed
version in typescript was used as a basis for the rust port.  The rust port
started after clvm_rs was started, because the wind seemed to be blowing toward
rust and because changes to the python code didn't seem as relevant.

The newer compiler has a different history; it was started in ocaml as a sketch
of how the structure of a chialisp compiler could improve some weaknesses of
the original chialisp compiler; reporting exact coordinates of errors in ways
that were more understandable to users, preserving more information from the
source text throughout compilation (the ability to do something like source
maps in javascript) and the ability to provide new forms with a reduced risk
of changing the form of code that already compiled.

In order to keep existing code working and ensure that existing processes didn't
break, a conservative approach was taken; the rust code here supports compiling
an advanced form of the original chialisp dialect with some things improved but
the shape and structure of most of the original code was maintained.  Since it
was in a python style to begin with, that may make some of it difficult to
navigate.

The python code started with entrypoints into two major functions, call\_tool and
launch\_tool of which launch_tool was the one that exposed code compilation and
is the more complex.  Since it both runs code and dispatches into 2 other fully
independent compiler implementations and does a few other things besides, here's a
guide to navigating it.

_The "main" program of the chialisp compiler_

The python code contained a 'cmds.py' in 'clvm\_tools', so the structure of the
rust code is similar; src/classic/clvm_tools/cmds.rs contains the code for all
the tools that historically existed; 'run', 'brun', 'opc', 'opd' and in similarly
named functions.  In particular, "launch\_tool".  This is exactly as in the python
code.

Similarly to the python code, the rust code starts by decoding arguments using an
analog of argparse.  The arguments are stored in a HashMap with type tags
determining what the stored data is.  In the case of PathOrCode conversion type,
it's been extended sligthly to remember the filename that was read in.  That's
useful so that the returned string can inform compilation of the filename it's
compiling so it can in turn inform the source locations what file name they belong
to.  A few other things are similar; since classic chialisp compilation mixes
code in the compiler's source language with expressions written in chialisp and
stores state in the CLVM runtime environment, a runtime environment is prepared
for it containing the search paths for include files (the classic compiler reads
include files via an imperative CLVM operator, \_read, installed at the time when
the interpreter is created.  This interpreter was called "stage\_2" in the python
code so a function, run\_program\_for\_search\_paths was included in
src/classic/stages/stage\_2/operators.rs for the purpose of creating that.  The
normal operation of a CLVM environment is usually immutable and this stance is
further encouraged by the lack of lifetime variables decorating the callbacks to
the clvm runner in clvmr.  Because of that, the code downstream uses a C++ like
approach to enriching the runtime environment with mutable state which will be
talked about later.  The actual call to run\_program\_for\_search\_paths is at
cmds.rs, line 798 currently.

At line 901 of cmds.rs (currently), the first decision is started regarding
compilation; an early function called detect_modern is called and returns
information regarding whether the user requested a specific chialisp dialect.
This is important because modern and classic chialisp compilers don't accept
precisely the same language and will continue to diverge.  Due to the demands
of long term storage of assets on the blockchain, it is necessary for code that
compiles in a specific form today retains that form forever, so the dialect also
tells the compiler which of the different forms compilation could take among
compatible alternatives.  This allows us to make better choices later on and
grow the ways in which the compiler can benefit users over time along with
the nature of HelperForm and BodyForm, which allow a clean separation between
generations of features.

The result of this choice is taken at line 940, where in the case of a modern
dialect, a short circuit path is taken that bypasses the more complicated
external interface of the classic compiler (below).  Since classic compilation
closely mirrors the form the python code takes (and should not be as new to
readers), I'll skip it for now and focus on modern compilation.

Due to a wrong choice I made early in the modern compiler's interface design,
cl21's basic optimization is enabled when called from python (the python
compilation interface used by chia-blockchain is fairly information poor so
assumes optimization on), but requires a traditional style -O flag on the command
line (line 358).  Called this way, python and command line compilation are the
same.  This will be fixed in the cl22 dialect, which will override the cl21
optimization flag entirely and also isn't the case in classic chialisp.

Mentioned later, CompilerOpts, which is a pure dynamic interface, implements
a set of toggles and settings that are global to the compilation process and
provide information during compilation.  One of these is required for compilation
in modern and the one used is generated at line 946 (currently).  A main entrypoint
to modern compilation that does all the needed work is in src/compiler/compiler.rs,
compile\_file.  It's called at line 952 (currently) and I'll talk about it in a
moment.

compile_file in src/compiler/compiler.rs is a simple interface to compilation,
given a collection of prerequisite objects, it first parses the source text given
using parse\_sexp in src/compiler/sexp.rs (line 817 currently) to produce SExp,
the data structure representing chialisp inputs and CLVM values, then gives the
resulting list of parsed forms to compile\_pre\_forms (line 170 of
src/compiler/compiler.rs currently).  compile\_pre\_forms first runs the frontend
(calling src/compiler/frontend.rs, function frontend at line 747) at line 177 of
src/compiler/compiler.rs yielding a CompileForm, which represents the full
semantics of the user's program.

It calls codegen (currently) at line 189 of src/compiler/compiler.rs, which
yields the compiled code in the way that has been outlined above and is detailed
below.

When called in this way, cmds.rs, launch_tool which runs the compiler, terminates
early, yielding the compiled program or error from the modern compiler process,
be it an error encountered while parsing, preprocessing, doing frontend
transformation, code generation or any other part of the modern compiler that
can produce an error.  Every error from the modern compiler has a Srcloc which
it tries to use to refer to something relevant in the source code.  These errors
have a relevant source of information through the process via the pervasive use
of Srcloc in the various compiler data structures.

Code
--

The modern compiler operates on just a few exposed types, and describes any
program using these (forming a rough hierarchy).

CompileForm   (src/compiler/comptypes.rs)
  HelperForm
    BodyForm

When things are referred to as "helpers" they are some kind of HelperForm.  These
are the definitions of things programs use as building blocks (as outlined below),
broadly the out of line constant, macro and function definitions that are used to
provide abstractions and parts of programs.

HelperForm and BodyForm are sum types which contain elements of the various kinds
of things the compiler understands as distinct forms the user can use.  At this
time, HelperForm is one of:

    Defconstant(DefconstData)
    Defmacro(DefmacData)
    Defun(bool, DefunData) // true = inline
    
Which spans the kinds of declarations that chialisp can contain.  Having a well
defined frontend type serves as a proof of sorts that the code has been fully
read and understood.  In the frontend form, we can perform transformations on the
code without worrying about breaking its more primitive representation.  Since
we've extracted the code's meaning we can more easily substitute runtime compatible
forms of the code from the perspective of the code's meaning.

The BodyForm is more diverse and things like Lambdas add alternatives.  Since
these are cleanly separated from a meaning perspective, it's possible to think
of groups of later alternatives as being layered on top of earlier ones without
affecting their meaning or how they're processed.

These are the current BodyForm alternatives:

    Let(LetFormKind, LetData) --
    
      Represents let forms and anything that's expressible through let forms,
      such as the 'assign' form here:
      
  [https://github.com/Chia-Network/chialisp/pull/103](assign form pr)
        
    Quoted(SExp) --
      
      Represents literal data in whatever form.  In particular, Quoted ensures
      that the semantic meaning of the value is always as a constant as opposed
      to being treated as a reference to something or a variable name.
      
    Value(SExp) --
    
      Value has a couple of meanings based on the content.  It can represent
      a self quoting value type such as a quoted string or integer if it contains
      a value in those domains or if it contains an atom, the atom is treated
      a reference to a constant or environment binding.
      
    Call(Srcloc, Vec<Rc<BodyForm>>) --
    
      Represents any kind of invocation in an expression position in the code,
      whether it's a macro invocation, invocation of a function or a primtive.
      The vector contains the top-level spine arguments of the notional cons
      form that will be given as arguments.  This language disallows tail
      improper call forms because, while identifiers work in tail position,
      there's no easy way to identify the user's intention to place a more
      complex form at the improper tail.  An 'apply' form as in lisp

   [http://clhs.lisp.se/Body/f_apply.htm](chialisp apply)
   
       honoring the convention that the final argument is properly bound according
       to the structure of the called functions arguments, allowing functions with
       tail-improper arguments to be called without having to manfacture a tail
       improper syntax for the call site.
    
    Mod(Srcloc, CompileForm) --
    
       Chialisp allows (mod () ...) as an expression yielding the
       compiled code in the form.  Naturally, its analog in the compiler
       contains a CompileForm, which allows it to be a full program of
       its own.

These are built from and often contain elements of SExp, which is also a sum type.
The sexp type is intended to reflect user intention, but keep the ability to treat
each value as plain clvm when required.  This places a greater burden on the
implementation when using these as values in a program, but allows all parts of
compilation (including code generation output) to remain associated with sites
and atoms in the source text when those associations make sense.  It contains:

    Nil(Srcloc) -- Represents a literal Nil in the source text.
    
      It may be useful for this to be distinct from 0 and "", at the very
      least to remember how the user spelled this particular value, but
      also (for example) it's possible to type Nil as a kind of list,
      but reject "" or 0 in the same scenario.
      
    Cons(Srcloc,Rc<SExp>,Rc<SExp>) -- The cons value.
    
    Integer(Srcloc,Number) -- An integer.
    
       Since the value system contains integers as a first class kind,
       it's possible to positively differentiate atoms from integers
       unless something disrupts them.  I am in the process of
       introducing a macro system that treats user input gently.
       
    QuotedString(Srcloc, u8, Vec<u8>) -- A quoted string.
    
       Since different kinds of quotes are possible, the QuotedString
       also remembers which quotation mark was used in the source text.
       Since its representation is distinct it can't be confused with
       an identifier.
       
    Atom(Srcloc,Vec<u8>) -- An atom or identifier.

Its job is to process programs so it doesn't implement compilation the same way
(by running a program that emits a mod form), but instead is purpose-built to
read and process a program.  As such it doesn't rely on populating a clvm
environment with special operators or on running anything necessarily in the VM,
although it does that for constant folding and a few other things.

Compilation can be done in a few ways.  There is a 

  CompilerOpts (src/compiler/compiler.rs)
  
Type which serves as connection between the consumer's settings for the compiler
and the compilation process.  Its method compile\_file takes a few objects it will
need in case it must run clvm code: an Allocator (clvmr) and a runner 
(TRunProgram from src/classic/stages/stage_0.rs).

It's used this way in 'run':

        let opts = Rc::new(DefaultCompilerOpts::new(&use_filename))
            .set_optimize(do_optimize)
            .set_search_paths(&search_paths)
            .set_frontend_opt(dialect > 21);
        let unopt_res = compile_file(
            &mut allocator,
            runner.clone(),
            opts.clone(),
            &input_program,
            &mut symbol_table,
        );
        let res = if do_optimize {
            unopt_res.and_then(|x| run_optimizer(&mut allocator, runner, Rc::new(x)))
        } else {
            unopt_res.map(Rc::new)
        };

This the the full lifecycle of chialisp compilation from an outside perspective.

If you want to parse the source file yourself, you can use parse_sexp from 
src/compiler/sexp, which yields a result of Vec&lt;Rc&lt;SExp&gt;&gt;, a vector
of refcounted pointers to s-expression objects.  These values are richer than clvm
values in that atoms, strings, hex values, integers and nils are distinguishable.
This is similar to the code in src/classic/clvm/type.rs but
since these value distinctions are first-class in SExp, all values processed
by the compiler, starting with the preprocessor (src/compiler/preprocessor.rs), 
going through the frontend (src/compile/frontend.rs) and to code generation
(src/compiler/codegen.rs) use these values.  Every expression read from the
input stream has a Srcloc (src/compiler/srcloc.rs) which indicates where it
was read from in the input, and as long as the srcloc is copied from form to
form, it allows the compiler to maintain known associations between input and
results.  For example, you can implement syntax coloring by simply using
parse\_sexp (src/compiler/sexp.rs) to parse a file, run it through the
preprocessor and frontend, and then decorate the locations present in the
HelperForms accessible from the CompilerForm that results.  If an error is
returned, it contains a srcloc that's relevant.

You can break down the basics of how chialisp compilation functions like this:

        let mut allocator = Allocator::new();
        let mut symbol_table = HashMap::new();
        let runner = Rc::new(DefaultProgramRunner::new());
        let parsed = parse_sexp(Srcloc::start(&use_filename), file_content.bytes())?;
        let compileform = frontend(opts.clone(), &parsed)?;
        let (hoisted_helpers, hoisted_body) = hoist_body_let_binding(compileform);
        let opts = Rc::new(DefaultCompilerOpts::new(&use_filename))
            .set_optimize(do_optimize)
            .set_search_paths(&search_paths)
            .set_frontend_opt(dialect > 21)
            .set_helpers(hoisted_helpers + default_helpers)
            .set_exp(hoisted_body);
        let generated_code = codegen(&mut allocator, runner, opts, &hoisted_body, &mut symbol_table)?;
        // generated_code.1 is an Rc<SExp> which contains the output code.

PrimaryCodegen is the object where the code generation stores and updates
information it needs and what gets collected during code generation.  It's defined
in src/compiler/comptypes.rs (line 281 at present).  Many of the functions in
src/compiler/codegen.rs take a PrimaryCodegen and most of those return a
PrimaryCodegen.  During this process, the compilation state is updated by each.

Everything before code generation is uninteresting, but I'll note at a high level
how functions on PrimaryCodegen work.

Next, let desugaring takes place in hoist_body_let_binding.

"Let" desugaring (let forms are used in lisp-like languages to bind additional
variables to new expressions, as in this example which uses a-greater-than-2
twice in difference sub-expressions.

    (mod (A B)
      (include *standard-cl-21*)
      (let ((a-greater-than-2 (> 2 A)))
        (c (i a-greater-than-2 B A) (i a-greater-than-2 (* 2 B) (* 2 A)))
        )
      )

inspects each defun (because they appear in the compiled code) and the program's
body for let forms and produces a list of new forms that must be re-inspected.
When no new forms are generated for any helper form, the full set of generated
helpers and the original are introduced to either the defun set or the inline set
in the PrimaryCodegen based on their type.  This will be a bit more flexible when
desugaring has its own definite pass as we'll have freedom to rearrange the
inlining without disturbing codegen itself.

Once all helpers are processed in this way, let desugaring takes place on the
main expression of the program in the same way.  The PrimaryCodegen has a special
field for the main expression.

codegen starts by running start_codegen to introduce each helper to the
PrimaryCodegen and based on their types, bin them into the appropriate parts
of the code generator to lookup later:

    Defconstants are turned into mods and run.
    The result is stored in the PrimaryCodegen's constant set.

    The defconst form is evaluated by putting all Helpers (as in objects of
    HelperForm type) into an Evaluator and asking it to shrink the constant's
    expression (resulting in constant folding).  The folded value must reduce
    to a constant, and if it does, it's stored in the constant set.

    Defmacros are converted to programs and compiled using the
    CompilerOpts' compile_program method.  These methods on CompilerOpts
    provide this kind of recursion with an eye to allowing the generation
    of program code to be used during compilation.  The resulting code
    is stored in the macro set.

Note that at this stage of compilation, we have helpers from the frontend, as well
as helpers introduced from let lifting. A HelperForm will have a member

        synthetic: bool = true;

if it was introduced by the compiler.

After start_codegen, the PrimaryCodegen is transformed by generating the dummy
environment via the dummy\_functions internal function.  For each non-inlined
defun and tabled constant, it extracts the name and generates an envrionment
shape from the names.  This is also where multiple definitions are detected.
As a result of this process, InlineFunction objects are generated for each inline
function and the PrimaryCodegen has its "parentfns" member popluated.

Each surviving helper is then passed through the codegen_ and given an opportunity
to transform the PrimaryCodegen.  The bodies of all functions are placed in the
PrimaryCodegen in the appropriate bin.  Defuns are turned into mods and code
generation is individually performed for them.  The representation placed in the
PrimaryCodegen is of compiled code.  During the process of generating code for
each live defun, the compiler is configured with the parent module's PrimaryCodegen
so that they observe the containing program's left environment and therefore can
request the code to make sibling function calls from it.

A few things about this are tricky; PrimaryCodegen uses a type called Callable to
lookup invocations and recognizes a few types:

    CallMacro
      
      Expands a macro and then treats the resulting output as the SExp
      representation of a BodyForm, so parses it and does expr code
      generation on it.
      
    CallInline
    
      Contains a literal body that is woven into the code via either the evaluator
      or src/compiler/inline.rs.
    
    CallDefun
    
      Contains recorded code indicating how to look up the function as
      well as the shape of its right env.
    
    CallPrim
    
      Contains a specification that a primitive is called, so outputs a primitive
      form directly after doing codegen on the arguments.
        
    RunCompiler
    
      This is exactly the "com" operator in classic chialisp.  When it is
      encountered, the expression evaluator creates a mod, prepares CompilerOpts
      to override the environment, provide this PrimaryCodegen for code generation
      and change other defaults and then uses it to compile the mod.  The result
      is code that "takes place" in the current context by sharing the environment
      shape and using the current PrimaryCodegen as a starting point.
        
    EnvPath
    
      As a compromise to allowing certain expressions to become environment lookups
      when that might not be expected, I provide a dedicated env-lookup form, 
      (@ n), where n is a constant integer only.  This desugars to a single 
      environment lookup in the generated code.

macro expansions require transformation of the user's code into clvm values and
back because the macro program is run as a clvm program (when this was written,
src/compiler/clvm wasn't fully mature and I hadn't written the evaluator yet).
A table of user supplied trees by treehash is made (src/compiler/debug.rs, 
build\_swap\_table\_mut), and the macro output is "rehydrated" by greedily
replacing each matching subtree with the one taken from the pre-expansion macro
callsite (src/compiler/debug.rs, relabel).  In this way, the code mostly preserved
distinctions between atoms and strings, etc in the source text through macro
invocation assuming things weren't too intrusive.

When running the "com" operator, which is used in macros to give the 

After all live defuns have been treated, the compiler uses final\_codegen to
generate the code for its main expression and then finalize\_env is called to
match each identifier stored in the left environment with a form for which it
has generated code recorded and build the env tree.

How CLVM code carries out programs
--

_The basics of CLVM from a compilation perspective_

One can think of CLVM as a calculator with 1 tricky operator, which is called 'a'
(apply).  The calculator's state can be thought of as a CLVM value like this:

  ((+ 2 5) . (99 101))
  
You can imagine the CLVM machine doing this on each evaluation step:

    Find the rightmost index (described well here: [https://github.com/Chia-Network/chialisp/blob/a660ce7ce07064a6a81bb361f169f6de195cba10/src/classic/clvm_tools/node_path.rs#L1](clvm paths) ) in the left part of the state that is not in the form of a quoted value
    (q . <something>) and:
   
     - if it's a number, enquote the correspondingly indexed value from
       the right part and replace the number with that.
       
     - otherwise, it must be an operator applicable to some constants so
       evaluate it.
       
    In this case,
    
      ((+ 2 3) . (99 . 101)) -> ((+ 2 (q . 101)) . (99 . 101))
      ((+ 2 (q . 101)) . (99 . 101)) -> ((+ (q . 99) (q . 101)) . (99 101))
      ((+ (q . 99) (q . 101)) . (99 . 101)) -> 200
      
    Of course, CLVM evaluators try to be more efficient than searching subtrees
    like this but at a theoretical level that's all one needs.
    
    The 'a' operator is actually not very exceptional in this, but you can think
    of its properties like this:
    
    1. It conceptually ignores one level of quoting from its first argument.
    2. It returns (q . X) when its first argument matches the pattern
       (q . (q . X)).
    3. When traversed during the search for the next rightmost element to
       transform, the right hand part of the machine state transforms into
       the dequoted form of its right hand argument:

            Nothing special happened yet, we just simplified a's second argument -.
                                                                                  v
      ((+ (q . 1) (a (q . (* (q . 3) 1)) 2)) . (10)) -> ((+ (q . 1) (a (q . (* (q . 3) 1) (q . 10)))) (10)))

                                      ,--- Here we traverse a and notice that we're entering the quoted code with a's env
                                      |                                                                  |
                                      v                                                                  v
      ((+ (q . 1) (a (q . [(* (q . 3) 1) . 10]) (q . 10)) . (10)) -> ((+ (q . 1) (a (q . (* (q . 3) (q . 10))) (q . 10))) . (10))

            The inner expression multiplied 3 by 10 and now matches (q . (q . X)) --.
                                                                                    v
      ((+ (q . 1) (a (q . (* (q . 3) (q . 10))))) . (10)) -> ((+ (q . 1) (a (q . (q . 30)) . (q . 10))) . (10))

        A disappears because we reached its end state of (q . (q . X)) --.
                                                                         v
      ((+ (q . 1) (a (q . (q . 30)) (q . 10))) . (10)) -> ((+ (q . 1) (q . 30)) . (10))

                                          Done

      ((+ (q . 1) (q . 30)) . (10)) -> 31

Thinking conceptually like this, we can construct any program we want by computing
a suitable environment (right side) and suitable code (left side), and letting
CLVM evaluate them until its done.  There are pragmatic things to think about in
how CLVM is evaluated:

 - If we have a program with functions that know about each other, their code has
   to be available to the program as quoted expressions to give to some 'a'
   operator.
   
 - If we have flow control, we have to model it by using some kind of computation
   and pass through code to an 'a' operator.
   
 - If we want to remember anything, we have to make an environment that contains it
   and then use it in some other code via an 'a' operator.
   
If one thinks of everything in these terms, then it isn't too difficult to generate
code from chialisp.

_The basic units of CLVM execution are env lookup and function application_

Because there is no memory other than the environment in CLVM, a basic building
block of CLVM code is an environment lookup.  If you consider the arguments to
a function to be some kind of mirror of the program's current environment:

    (defun F (X Y Z) (+ X Y Z))
    
    brun '(+ 2 5 11)' '(10 11 12)' -> 33
    
Then you can see that there's nothing keeping references to the function's
argument list from just becoming references.

You can do the same thing with functions in the environment:

    brun '(+ (a 2 (c 5 ())) (q . 3))' '((* (q . 2) 2) 9)' -> 21
    
But we don't want to make the user send in another copy of the program when
running the program, and even more, if it needs that function more than once,
we'll have to explicitly carry it around.

Richard's answer and the generallly accepted one is to let the environment usually
be a cons whose first part is the full set of constants and functions the program
can use and the rest part is the arguments to the current function.  Optimization
may in the future make this more complicated but thinking about it in this way
makes a lot of things pretty easy to reason about.

The program starts by putting its arguments with all its functions and applying
the main expression:

    (a main-expression (c (env ...) 1))

The main-expression expects the functions it has access to to be in the left
part of the environment:

                              .-- Reference to the left env.
                              |    .-- An argument
                              v    v
    (a (a some-even-number (c 2 (c 5 ()))) (c (env ...) 1))

And those functions expect the same thing, so if they generate code in the same
way, for example, in this program, the function could call itself recursively:

    (a (i 5 (q . (a some-even-number (c 2 (c (r 5) ())))) ()) 1)
    
some-even-number refers to the same subpart of the environment because we've
promised ourselves that the part of the environment containing ourselves is at
the same place in the environment 

Because the 'a' operator shares a lot of properties with a function call in a
purely functional language, chialisp compilers tend to rely heavily on "desugaring"
various language features into function calls so that they can use the code
generation tools they already have (hopefully) that implements functions.

This isn't uncommon in language compilers, but converting things to functions is
a higher percentage of what chialisp has to do.

The second thing chialisp compilers have to do is synthesize ("reify") the
environment to fit the shape that code it's calling is expected.  Consider this:

   (let ((X (+ Y 1))) (* X 2))
   
In chialisp, there are only a few choices you can make:

- Replace X with (+ Y 1) everywhere it appears in the body and return that.

    (* (+ Y 1) 2)
  
- Make some anonymous function for the binding like 
   
    (defun let_binding_11234 (Y) (+ Y 1))
    
  and replace every X with (let_binding_11234 Y)
   
- Make a function for the let body like:

    (defun let_body_11234 (Y and_X?) (* and_X? 2))
    
  And replace the original site with:
  
    (let_body_11234 Y (+ Y 1))
    
In each of these cases but the pure inline one, extra new arguments were fitted
into the argument list of some function and the function was called.  In this way,
we dreamed up an environment shape and then had to match it to call the function.

This takes on complication, but isn't really any worse, if let bindings can nest
in various ways.  The goal of the chialisp compiler and its optimization are to
make decent choices for these and improve the degree to which they produce the
smallest possible equivalent code over time.  Each of these can be the right choice
under some circumstances.

Along other lines, this example:

Imagine this program:

    (mod (X)
        (defun list-length (N L) (if L (list-length (+ N 1) (r L)) N))
        (defun-inline QQQ (A . B) (+ A (list-length 0 B)))
        (QQQ X 4 5 6)
        )
        
The inline here destructures its arguments.  The arguments aren't strictly aligned
to the elements that are used to call it; B captures (4 5 6).  Because QQQ isn't
really "called" with an apply atom, the shape of its environment is purely 
theoretical and yet the program *observes* that shape at runtime.  We must, when
required, aggregate or dis-aggregate arguments given by the user by figuring their
positions with respect to landmarks we know.  Best case, we can apply first, rest
or paths to values we have as arguments if arguments in standard positions are
destructured, worst case we have to cons values together as though the environment
was being built for an apply.
    
_CLVM Heap representation isn't dissimiar to other eager functional languages_

How CLVM compares to other functional languages in structure is a topic that is
talked about occasionally, and many of its decisions are fairly alike what's out
there:

    In CLVM, there are conses and atoms.  Atoms act as numbers and
    strings and conses act as boxes of size 2.
    
    (7 8 . 9) =
      
          ( 7 .  )
               \
              (8 . 9)

    
    In cadmium (the name some use for the ocaml bytecode vm), there are
    pointers and words.  Words act as numbers and pointers act as boxes
    of a size determined by their tag word.
        
    (7,(8,9)) =
    
      [size:2,7,*0x331110] 
                      \
                  [size:2,8,9]

    $ ocaml
    # let x = (7,(8,9));;
    val x : int * (int * int) = (7, (8, 9))
    # let a = Obj.repr x;;
    val a : Obj.t = <abstr>
    # Obj.size a;;
    - : int = 2
    # let b = Obj.field a 1;;
    val b : Obj.t = <abstr>
    # Obj.size b;;
    - : int = 2
