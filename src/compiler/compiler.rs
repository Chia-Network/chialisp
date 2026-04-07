use num_bigint::ToBigInt;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use clvm_rs::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::clvm::{sha256tree, NewStyleIntConversion};
use crate::compiler::codegen::{codegen, hoist_body_let_binding, process_helper_let_bindings};
use crate::compiler::comptypes::{CompileErr, CompileForm, CompilerOpts, PrimaryCodegen};
use crate::compiler::dialect::{AcceptedDialect, KNOWN_DIALECTS};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::prims;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::{BasicCompileContext, CompileContextWrapper};
use crate::util::Number;

pub const FUZZ_TEST_PRE_CSE_MERGE_FIX_FLAG: usize = 1;

lazy_static! {
    pub static ref STANDARD_MACROS: String = {
        indoc! {"(
            (defmacro if (A B C) (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @)))
            (defmacro list ARGS
                            (defun compile-list
                                   (args)
                                   (if args
                                       (qq (c (unquote (f args))
                                             (unquote (compile-list (r args)))))
                                       ()))
                            (compile-list ARGS)
                    )
            (defun-inline / (A B) (f (divmod A B)))
            )
            "}
        .to_string()
    };
    pub static ref ADVANCED_MACROS: String = {
        indoc! {"(
            (defmac __chia__primitive__if (A B C)
              (qq (a (i (unquote A) (com (unquote B)) (com (unquote C))) @))
              )

            (defun __chia__if (ARGS)
              (__chia__primitive__if (r (r (r ARGS)))
                (qq (a (i (unquote (f ARGS)) (com (unquote (f (r ARGS)))) (com (unquote (__chia__if (r (r ARGS)))))) @))
                (qq (a (i (unquote (f ARGS)) (com (unquote (f (r ARGS)))) (com (unquote (f (r (r ARGS)))))) @))
                )
              )

            (defmac if ARGS (__chia__if ARGS))

            (defun __chia__compile-list (args)
              (if args
                (c 4 (c (f args) (c (__chia__compile-list (r args)) ())))
                ()
                )
              )

            (defmac list ARGS (__chia__compile-list ARGS))

            (defun-inline / (A B) (f (divmod A B)))
            )
            "}
        .to_string()
    };
}

#[derive(Clone, Debug)]
pub struct DefaultCompilerOpts {
    pub include_dirs: Vec<String>,
    pub filename: String,
    pub code_generator: Option<PrimaryCodegen>,
    pub in_defun: bool,
    pub stdenv: bool,
    pub optimize: bool,
    pub frontend_opt: bool,
    pub frontend_check_live: bool,
    pub start_env: Option<Rc<SExp>>,
    pub disassembly_ver: Option<usize>,
    pub prim_map: Rc<HashMap<Vec<u8>, Rc<SExp>>>,
    pub diag_flags: Rc<HashSet<usize>>,
    pub dialect: AcceptedDialect,
}

pub fn create_prim_map() -> Rc<HashMap<Vec<u8>, Rc<SExp>>> {
    let mut prim_map: HashMap<Vec<u8>, Rc<SExp>> = HashMap::new();

    for p in prims::prims() {
        prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
    }

    Rc::new(prim_map)
}

fn do_desugar(program: &CompileForm) -> Result<CompileForm, CompileErr> {
    // Transform let bindings, merging nested let scopes with the top namespace
    let hoisted_bindings = hoist_body_let_binding(None, program.args.clone(), program.exp.clone())?;
    let mut new_helpers = hoisted_bindings.0;
    let expr = hoisted_bindings.1; // expr is the let-hoisted program

    // TODO: Distinguish the frontend_helpers and the hoisted_let helpers for later stages
    let mut combined_helpers = program.helpers.clone();
    combined_helpers.append(&mut new_helpers);
    let combined_helpers = process_helper_let_bindings(&combined_helpers)?;

    Ok(CompileForm {
        helpers: combined_helpers,
        exp: expr,
        ..program.clone()
    })
}

pub fn desugar_pre_forms(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    pre_forms: &[Rc<SExp>],
) -> Result<CompileForm, CompileErr> {
    let p0 = frontend(opts.clone(), pre_forms)?;

    let p1 = context.frontend_optimization(opts.clone(), p0)?;

    do_desugar(&p1)
}

pub fn compile_from_compileform(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    p2: CompileForm,
) -> Result<SExp, CompileErr> {
    let p3 = context.post_desugar_optimization(opts.clone(), p2)?;

    // generate code from AST, optionally with optimization
    let generated = codegen(context, opts.clone(), &p3)?;

    let g2 = context.post_codegen_output_optimize(opts, generated)?;

    Ok(g2)
}

pub fn compile_pre_forms(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    pre_forms: &[Rc<SExp>],
) -> Result<SExp, CompileErr> {
    // Resolve includes, convert program source to lexemes
    let p2 = desugar_pre_forms(context, opts.clone(), pre_forms)?;

    compile_from_compileform(context, opts, p2)
}

pub fn compile_file(
    allocator: &mut Allocator,
    runner: Rc<dyn TRunProgram>,
    opts: Rc<dyn CompilerOpts>,
    content: &str,
    symbol_table: &mut HashMap<String, String>,
) -> Result<SExp, CompileErr> {
    let _int_conversion_bug = NewStyleIntConversion::new(opts.dialect().int_fix);
    let srcloc = Srcloc::start(&opts.filename());
    let pre_forms = parse_sexp(srcloc.clone(), content.bytes())?;
    let mut context_wrapper = CompileContextWrapper::new(
        allocator,
        runner,
        symbol_table,
        get_optimizer(&srcloc, opts.clone())?,
    );
    compile_pre_forms(&mut context_wrapper.context, opts, &pre_forms)
}

impl CompilerOpts for DefaultCompilerOpts {
    fn filename(&self) -> String {
        self.filename.clone()
    }
    fn code_generator(&self) -> Option<PrimaryCodegen> {
        self.code_generator.clone()
    }
    fn dialect(&self) -> AcceptedDialect {
        self.dialect.clone()
    }
    fn in_defun(&self) -> bool {
        self.in_defun
    }
    fn stdenv(&self) -> bool {
        self.stdenv
    }
    fn optimize(&self) -> bool {
        self.optimize
    }
    fn frontend_opt(&self) -> bool {
        self.frontend_opt
    }
    fn frontend_check_live(&self) -> bool {
        self.frontend_check_live
    }
    fn start_env(&self) -> Option<Rc<SExp>> {
        self.start_env.clone()
    }
    fn prim_map(&self) -> Rc<HashMap<Vec<u8>, Rc<SExp>>> {
        self.prim_map.clone()
    }
    fn disassembly_ver(&self) -> Option<usize> {
        self.disassembly_ver
    }
    fn get_search_paths(&self) -> Vec<String> {
        self.include_dirs.clone()
    }
    fn diag_flags(&self) -> Rc<HashSet<usize>> {
        self.diag_flags.clone()
    }

    fn set_dialect(&self, dialect: AcceptedDialect) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.dialect = dialect;
        Rc::new(copy)
    }
    fn set_search_paths(&self, dirs: &[String]) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        dirs.clone_into(&mut copy.include_dirs);
        Rc::new(copy)
    }
    fn set_disassembly_ver(&self, ver: Option<usize>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.disassembly_ver = ver;
        Rc::new(copy)
    }
    fn set_in_defun(&self, new_in_defun: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.in_defun = new_in_defun;
        Rc::new(copy)
    }
    fn set_stdenv(&self, new_stdenv: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.stdenv = new_stdenv;
        Rc::new(copy)
    }
    fn set_optimize(&self, optimize: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.optimize = optimize;
        Rc::new(copy)
    }
    fn set_frontend_opt(&self, optimize: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.frontend_opt = optimize;
        Rc::new(copy)
    }
    fn set_frontend_check_live(&self, check: bool) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.frontend_check_live = check;
        Rc::new(copy)
    }
    fn set_code_generator(&self, new_code_generator: PrimaryCodegen) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.code_generator = Some(new_code_generator);
        Rc::new(copy)
    }
    fn set_start_env(&self, start_env: Option<Rc<SExp>>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.start_env = start_env;
        Rc::new(copy)
    }
    fn set_prim_map(&self, prims: Rc<HashMap<Vec<u8>, Rc<SExp>>>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.prim_map = prims;
        Rc::new(copy)
    }
    fn set_diag_flags(&self, flags: Rc<HashSet<usize>>) -> Rc<dyn CompilerOpts> {
        let mut copy = self.clone();
        copy.diag_flags = flags;
        Rc::new(copy)
    }

    fn read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr> {
        if filename == "*macros*" {
            if self.dialect().strict {
                return Ok((filename, ADVANCED_MACROS.bytes().collect()));
            } else {
                return Ok((filename, STANDARD_MACROS.bytes().collect()));
            }
        } else if let Some(dialect) = KNOWN_DIALECTS.get(&filename) {
            return Ok((filename, dialect.content.bytes().collect()));
        }

        for dir in self.include_dirs.iter() {
            let mut p = PathBuf::from(dir);
            p.push(filename.clone());
            match fs::read(p.clone()) {
                Err(_e) => {
                    continue;
                }
                Ok(content) => {
                    return Ok((
                        p.to_str().map(|x| x.to_owned()).unwrap_or_else(|| filename),
                        content,
                    ));
                }
            }
        }
        Err(CompileErr(
            Srcloc::start(&inc_from),
            format!("could not find {filename} to include"),
        ))
    }
    fn compile_program(
        &self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        sexp: Rc<SExp>,
        symbol_table: &mut HashMap<String, String>,
    ) -> Result<SExp, CompileErr> {
        let _int_conversion_bug = NewStyleIntConversion::new(self.dialect.int_fix);
        let me = Rc::new(self.clone());
        let optimizer = get_optimizer(&sexp.loc(), me.clone())?;
        let mut context_wrapper =
            CompileContextWrapper::new(allocator, runner, symbol_table, optimizer);
        compile_pre_forms(&mut context_wrapper.context, me, &[sexp])
    }
}

impl DefaultCompilerOpts {
    pub fn new(filename: &str) -> DefaultCompilerOpts {
        DefaultCompilerOpts {
            include_dirs: vec![".".to_string()],
            filename: filename.to_string(),
            code_generator: None,
            in_defun: false,
            stdenv: true,
            optimize: false,
            frontend_opt: false,
            frontend_check_live: true,
            start_env: None,
            dialect: AcceptedDialect::default(),
            prim_map: create_prim_map(),
            disassembly_ver: None,
            diag_flags: Rc::new(HashSet::default()),
        }
    }
}

fn path_to_function_inner(
    program: Rc<SExp>,
    hash: &[u8],
    path_mask: Number,
    current_path: Number,
) -> Option<Number> {
    let nextpath = path_mask.clone() * 2_i32.to_bigint().unwrap();
    match program.borrow() {
        SExp::Cons(_, a, b) => {
            path_to_function_inner(a.clone(), hash, nextpath.clone(), current_path.clone())
                .map(Some)
                .unwrap_or_else(|| {
                    path_to_function_inner(
                        b.clone(),
                        hash,
                        nextpath.clone(),
                        current_path.clone() + path_mask.clone(),
                    )
                    .map(Some)
                    .unwrap_or_else(|| {
                        let current_hash = sha256tree(program.clone());
                        if current_hash == hash {
                            Some(current_path + path_mask)
                        } else {
                            None
                        }
                    })
                })
        }
        _ => {
            let current_hash = sha256tree(program.clone());
            if current_hash == hash {
                Some(current_path + path_mask)
            } else {
                None
            }
        }
    }
}

pub fn path_to_function(program: Rc<SExp>, hash: &[u8]) -> Option<Number> {
    path_to_function_inner(program, hash, bi_one(), bi_zero())
}

fn op2(op: u32, code: Rc<SExp>, env: Rc<SExp>) -> Rc<SExp> {
    Rc::new(SExp::Cons(
        code.loc(),
        Rc::new(SExp::Integer(env.loc(), op.to_bigint().unwrap())),
        Rc::new(SExp::Cons(
            code.loc(),
            code.clone(),
            Rc::new(SExp::Cons(
                env.loc(),
                env.clone(),
                Rc::new(SExp::Nil(code.loc())),
            )),
        )),
    ))
}

fn quoted(env: Rc<SExp>) -> Rc<SExp> {
    Rc::new(SExp::Cons(
        env.loc(),
        Rc::new(SExp::Integer(env.loc(), bi_one())),
        env.clone(),
    ))
}

fn apply(code: Rc<SExp>, env: Rc<SExp>) -> Rc<SExp> {
    op2(2, code, env)
}

fn cons(f: Rc<SExp>, r: Rc<SExp>) -> Rc<SExp> {
    op2(4, f, r)
}

// compose (a (a path env) (c env 1))
pub fn rewrite_in_program(path: Number, env: Rc<SExp>) -> Rc<SExp> {
    apply(
        apply(
            // Env comes quoted, so divide by 2
            quoted(Rc::new(SExp::Integer(env.loc(), path / 2))),
            env.clone(),
        ),
        cons(env.clone(), Rc::new(SExp::Integer(env.loc(), bi_one()))),
    )
}

pub fn is_operator(op: u32, atom: &SExp) -> bool {
    match atom.to_bigint() {
        Some(n) => n == op.to_bigint().unwrap(),
        None => false,
    }
}

pub fn is_whole_env(atom: &SExp) -> bool {
    is_operator(1, atom)
}
pub fn is_apply(atom: &SExp) -> bool {
    is_operator(2, atom)
}
pub fn is_cons(atom: &SExp) -> bool {
    is_operator(4, atom)
}

// Extracts the environment from a clvm program that contains one.
// The usual form of a program to analyze is:
// (2 main (4 env 1))
pub fn extract_program_and_env(program: Rc<SExp>) -> Option<(Rc<SExp>, Rc<SExp>)> {
    // Most programs have apply as a toplevel form.  If we don't then it's
    // a form we don't understand.
    match program.proper_list() {
        Some(lst) => {
            if lst.len() != 3 {
                return None;
            }

            match (is_apply(&lst[0]), &lst[1], lst[2].proper_list()) {
                (true, real_program, Some(cexp)) => {
                    if cexp.len() != 3 || !is_cons(&cexp[0]) || !is_whole_env(&cexp[2]) {
                        None
                    } else {
                        Some((Rc::new(real_program.clone()), Rc::new(cexp[1].clone())))
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

pub fn is_at_capture(head: Rc<SExp>, rest: Rc<SExp>) -> Option<(Vec<u8>, Rc<SExp>)> {
    rest.proper_list().and_then(|l| {
        if l.len() != 2 {
            return None;
        }
        if let (SExp::Atom(_, a), SExp::Atom(_, cap)) = (head.borrow(), &l[0]) {
            if a == &vec![b'@'] {
                return Some((cap.clone(), Rc::new(l[1].clone())));
            }
        }

        None
    })
}
