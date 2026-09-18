use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;
use clvm_rs::reduction::Reduction;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm::sexp::{enlist, first, map_m, proper_list, rest};
use crate::classic::clvm::OPERATORS_LATEST_VERSION;
use crate::classic::clvm::{keyword_from_atom, keyword_to_atom};

use crate::classic::clvm_tools::binutils::assemble;
use crate::classic::clvm_tools::node_path::NodePath;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::classic::clvm_tools::stages::stage_2::abstraction::{
    ASExp, BufCarrier, ClError, ClassicAllocator,
};
use crate::classic::clvm_tools::stages::stage_2::defaults::default_macro_lookup;
use crate::classic::clvm_tools::stages::stage_2::helpers::{brun, evaluate, quote};
use crate::classic::clvm_tools::stages::stage_2::module::compile_mod;
use crate::compiler::srcloc::Srcloc;

const DIAG_OUTPUT: bool = false;

lazy_static! {
    static ref PASS_THROUGH_OPERATORS: HashSet<Vec<u8>> = {
        let mut result = HashSet::new();
        for key in keyword_to_atom(OPERATORS_LATEST_VERSION).keys() {
            result.insert(key.as_bytes().to_vec());
        }
        for key in keyword_from_atom(OPERATORS_LATEST_VERSION).keys() {
            result.insert(key.to_vec());
        }
        // added by optimize
        result.insert("com".as_bytes().to_vec());
        result.insert("opt".as_bytes().to_vec());
        result
    };
}

struct Closure<'a, A: ClassicAllocator> {
    name: String,
    #[allow(clippy::type_complexity)]
    to_run: &'a dyn Fn(
        &mut A,
        &A::NodePtr,
        &A::NodePtr,
        &A::NodePtr,
        Rc<dyn TRunProgram>,
        usize,
    ) -> Result<A::NodePtr, ClError>,
}

fn compile_bindings<'a, A: ClassicAllocator>() -> HashMap<Vec<u8>, Closure<'a, A>>
where
    A::NodePtr: Clone,
{
    let mut bindings = HashMap::new();
    let bindings_source = vec![
        Closure {
            name: "qq".to_string(),
            to_run: &compile_qq,
        },
        Closure {
            name: "macros".to_string(),
            to_run: &compile_macros,
        },
        Closure {
            name: "symbols".to_string(),
            to_run: &compile_symbols,
        },
        Closure {
            name: "lambda".to_string(),
            to_run: &compile_mod,
        },
        Closure {
            name: "mod".to_string(),
            to_run: &compile_mod,
        },
    ];

    for c in bindings_source {
        bindings.insert(c.name.as_bytes().to_vec(), c);
    }

    bindings
}

fn qq_atom() -> Vec<u8> {
    vec![b'q', b'q']
}
fn unquote_atom() -> Vec<u8> {
    "unquote".as_bytes().to_vec()
}

fn com_qq<A: ClassicAllocator>(
    allocator: &mut A,
    ident: String,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    runner: Rc<dyn TRunProgram>,
    sexp: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    if DIAG_OUTPUT {
        println!("com_qq {} {}", ident, allocator.disassemble(sexp, None));
    }
    do_com_prog(allocator, 110, sexp, macro_lookup, symbol_table, runner)
}

pub fn compile_qq<A: ClassicAllocator>(
    allocator: &mut A,
    args: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    runner: Rc<dyn TRunProgram>,
    level: usize,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    /*
     * (qq ATOM) => (q . ATOM)
     * (qq (unquote X)) => X
     * (qq (a . B)) => (c (qq a) (qq B))
     */

    let sexp = first(allocator, args)?;
    let loc = allocator.loc(&sexp);
    let nil_import = allocator.import(loc.clone(), NodePtr::NIL)?;

    match allocator.sexp(&sexp) {
        ASExp::Atom => {
            // (qq ATOM) => (q . ATOM)
            quote(allocator, &sexp)
        }
        ASExp::Pair(op, sexp_rest) => {
            if let ASExp::Atom = allocator.sexp(&op) {
                // opbuf => op
                let op_atom = allocator.atom(&op);
                let op_loc = allocator.loc(&op);
                if op_atom.as_ref() == qq_atom() {
                    let op_loc = allocator.loc(&op);
                    let cons_atom = allocator.new_atom(op_loc, &[4])?;
                    let subexp = compile_qq(
                        allocator,
                        &sexp_rest,
                        macro_lookup,
                        symbol_table,
                        runner.clone(),
                        level + 1,
                    )?;
                    let quoted_null = quote(allocator, &nil_import)?;
                    let consed = enlist(allocator, &[cons_atom.clone(), subexp, quoted_null])?;
                    let run_list = enlist(allocator, &[cons_atom, op, consed])?;
                    return com_qq(
                        allocator,
                        "qq sexp pair".to_string(),
                        macro_lookup,
                        symbol_table,
                        runner,
                        &run_list,
                    );
                } else if op_atom.as_ref() == unquote_atom() {
                    // opbuf
                    if level == 1 {
                        // (qq (unquote X)) => X
                        let sexp_rf = first(allocator, &sexp_rest)?;
                        return com_qq(
                            allocator,
                            "level 1".to_string(),
                            macro_lookup,
                            symbol_table,
                            runner,
                            &sexp_rf,
                        );
                    }

                    // (qq (a . B)) => (c (qq a) (qq B))
                    let cons_atom = allocator.new_atom(op_loc, &[4])?;
                    let subexp = compile_qq(
                        allocator,
                        &sexp_rest,
                        macro_lookup,
                        symbol_table,
                        runner.clone(),
                        level - 1,
                    )?;
                    let quoted_null = quote(allocator, &nil_import)?;
                    let consed_subexp =
                        enlist(allocator, &[cons_atom.clone(), subexp, quoted_null])?;
                    let run_list = enlist(allocator, &[cons_atom, op, consed_subexp])?;

                    return com_qq(
                        allocator,
                        "qq pair general".to_string(),
                        macro_lookup,
                        symbol_table,
                        runner,
                        &run_list,
                    );
                }
            }

            // (qq (a . B)) => (c (qq a) (qq B))
            let cons_atom = allocator.new_atom(loc.clone(), &[4])?;
            let qq = allocator.new_atom(loc, &qq_atom())?;
            let qq_l = enlist(allocator, &[qq.clone(), op])?;
            let qq_r = enlist(allocator, &[qq, sexp_rest])?;
            let compiled_l = com_qq(
                allocator,
                "A".to_string(),
                macro_lookup,
                symbol_table,
                runner.clone(),
                &qq_l,
            )?;
            let compiled_r = com_qq(
                allocator,
                "B".to_string(),
                macro_lookup,
                symbol_table,
                runner,
                &qq_r,
            )?;
            enlist(allocator, &[cons_atom, compiled_l, compiled_r])
        }
    }
}

pub fn compile_macros<A: ClassicAllocator>(
    allocator: &mut A,
    _args: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    _symbol_table: &A::NodePtr,
    _run_program: Rc<dyn TRunProgram>,
    _level: usize,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    quote(allocator, macro_lookup)
}

pub fn compile_symbols<A: ClassicAllocator>(
    allocator: &mut A,
    _args: &A::NodePtr,
    _macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    _run_program: Rc<dyn TRunProgram>,
    _level: usize,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    quote(allocator, symbol_table)
}

// # Transform "quote" to "q" everywhere. Note that quote will not be compiled if behind qq.
// # Overrides symbol table defns.
fn lower_quote_<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(prog);
    let exported = allocator.export(prog);

    if allocator.is_nil(prog) {
        return Ok(prog.clone());
    }

    if let Some(qlist) = proper_list(allocator, prog, true) {
        if qlist.is_empty() {
            return Ok(prog.clone());
        }

        // quote_node was Atom(q)
        let quote_node = &qlist[0];
        if let ASExp::Atom = allocator.sexp(quote_node) {
            let quote_atom = allocator.atom(quote_node);
            if quote_atom.as_ref() == b"quote" {
                if qlist.len() != 2 {
                    // quoted list should be 2: "(quote arg)"
                    return Err(
                        ClError(
                            loc,
                            EvalErr::InternalError(
                                exported,
                                format!(
                                    "Compilation error while compiling [{}]. quote takes exactly one argument.",
                                    allocator.disassemble(prog, None)
                                )
                            )
                        )
                    );
                }

                // Note: quote should have exactly one arg, so the length of
                let lowered = lower_quote(allocator, &qlist[1])?;
                return quote(allocator, &lowered);
            }
        }
    }

    // XXX Note that this recognizes potentially unintended
    // syntax, in that (sha256 3 quote ()) is valid in this
    // code.  It is corrected in the new compiler but left
    // here in case this bug is exploited.
    // Like a good neighbor, UB is there☺
    if let ASExp::Pair(f, r) = allocator.sexp(prog) {
        let first = lower_quote(allocator, &f)?;
        let rest = lower_quote(allocator, &r)?;
        return allocator.new_pair(loc, &first, &rest);
    }

    Ok(prog.clone())
}

pub fn lower_quote<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let res = lower_quote_(allocator, prog);
    if DIAG_OUTPUT {
        res.as_ref()
            .map(|x| {
                println!(
                    "LOWER_QUOTE {} TO {}",
                    allocator.disassemble(prog, None),
                    allocator.disassemble(x, None)
                );
            })
            .unwrap_or_else(|_| ())
    }
    res
}

fn try_expand_macro_for_atom_<A: ClassicAllocator>(
    allocator: &mut A,
    macro_code: &A::NodePtr,
    prog_rest: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(macro_code);
    let com_atom = allocator.new_atom(loc.clone(), "com".as_bytes())?;
    let exported_macro = allocator.export(macro_code);
    let exported_prog = allocator.export(prog_rest);
    let post_prog = brun(allocator.allocator(), exported_macro, exported_prog)?;
    let imported_post = allocator.import(loc.clone(), post_prog)?;
    let quoted_macros = quote(allocator, macro_lookup)?;
    let quoted_symbols = quote(allocator, symbol_table)?;
    let to_eval = enlist(
        allocator,
        &[com_atom, imported_post, quoted_macros, quoted_symbols],
    )?;
    let top_path = allocator.new_atom(loc, NodePath::new(None).as_path().data())?;
    evaluate(allocator, &to_eval, &top_path).inspect(|x| {
        if DIAG_OUTPUT {
            println!(
                "TRY_EXPAND_MACRO {} WITH {} GIVES {} MACROS {} SYMBOLS {}",
                allocator.disassemble(macro_code, None),
                allocator.disassemble(prog_rest, None),
                allocator.disassemble(x, None),
                allocator.disassemble(macro_lookup, None),
                allocator.disassemble(symbol_table, None)
            );
        }
    })
}

pub fn try_expand_macro_for_atom<A: ClassicAllocator>(
    allocator: &mut A,
    macro_code: &A::NodePtr,
    prog_rest: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    try_expand_macro_for_atom_(allocator, macro_code, prog_rest, macro_lookup, symbol_table)
}

fn get_macro_program<A: ClassicAllocator>(
    allocator: &mut A,
    operator: &[u8],
    macro_lookup: &A::NodePtr,
) -> Result<Option<A::NodePtr>, ClError>
where
    A::NodePtr: Clone,
{
    if let Some(mlist) = proper_list(allocator, macro_lookup, true) {
        for macro_pair in mlist {
            match proper_list(allocator, &macro_pair, true) {
                None => {}
                Some(mp_list) => {
                    if mp_list.is_empty() {
                        continue;
                    }
                    let value = if mp_list.len() > 1 {
                        mp_list[1].clone()
                    } else {
                        let loc = allocator.loc(macro_lookup);
                        allocator.import(loc, NodePtr::NIL)?
                    };

                    match allocator.sexp(&mp_list[0]) {
                        ASExp::Atom => {
                            // was macro_name, but it's singular and probably
                            // not useful to rename.
                            let atom = allocator.atom(&mp_list[0]);
                            if atom.as_ref() == operator {
                                return Ok(Some(value));
                            }
                        }
                        ASExp::Pair(_, _) => {
                            continue;
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

fn transform_program_atom<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
    a: &[u8],
    symbol_table: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let loc = allocator.loc(prog);
    if a == b"@" || a == b"@*env*" {
        return allocator.new_atom(loc, NodePath::new(None).as_path().data());
    }

    match proper_list(allocator, symbol_table, true) {
        None => {}
        Some(symlist) => {
            let nil_import = allocator.import(loc.clone(), NodePtr::NIL)?;
            for sym in symlist {
                match proper_list(allocator, &sym, true) {
                    None => {}
                    Some(v) => {
                        if v.is_empty() {
                            continue;
                        }

                        let value = if v.len() > 1 {
                            v[1].clone()
                        } else {
                            nil_import.clone()
                        };

                        match allocator.sexp(&v[0]) {
                            ASExp::Atom => {
                                // v[0] is close by, and probably not useful to
                                // rename here.
                                let atom = allocator.atom(&v[0]);
                                if atom.as_ref() == a {
                                    return Ok(value);
                                }
                            }
                            ASExp::Pair(_, _) => {}
                        }
                    }
                }
            }
        }
    }

    quote(allocator, prog)
}

fn compile_operator_atom<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
    avec: &[u8],
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Result<Option<A::NodePtr>, ClError>
where
    A::NodePtr: Clone,
{
    let compile_bindings = compile_bindings();

    if *avec == vec![1] {
        return Ok(Some(prog.clone()));
    }

    if let Some(f) = compile_bindings.get(avec) {
        let prog_rest = rest(allocator, prog)?;
        let post_prog = (f.to_run)(
            allocator,
            &prog_rest,
            macro_lookup,
            symbol_table,
            run_program.clone(),
            1,
        )?;
        let quoted_post_prog = quote(allocator, &post_prog)?;
        let loc = allocator.loc(prog);
        let top_atom = allocator.new_atom(loc, NodePath::new(None).as_path().data())?;
        if DIAG_OUTPUT {
            print!(
                "COMPILE_BINDINGS {}",
                allocator.disassemble(&quoted_post_prog, None)
            );
        };
        return evaluate(allocator, &quoted_post_prog, &top_atom).map(Some);
    }

    Ok(None)
}

enum SymbolResult<A: ClassicAllocator> {
    Direct(A::NodePtr),
    Matched(A::NodePtr, A::NodePtr),
}

fn find_symbol_match<A: ClassicAllocator>(
    allocator: &mut A,
    opname: &[u8],
    r: &A::NodePtr,
    symbol_table: &A::NodePtr,
) -> Result<Option<SymbolResult<A>>, ClError>
where
    A::NodePtr: Clone,
{
    if let Some(symlist) = proper_list(allocator, symbol_table, true) {
        for sym in symlist {
            if let Some(symdef) = proper_list(allocator, &sym, true) {
                if symdef.is_empty() {
                    continue;
                }

                match allocator.sexp(&symdef[0]) {
                    ASExp::Atom => {
                        let symbol = symdef[0].clone();
                        let value = if symdef.len() == 1 {
                            let loc = allocator.loc(r);
                            allocator.import(loc, NodePtr::NIL)?
                        } else {
                            symdef[1].clone()
                        };

                        let symbuf = allocator.atom(&symdef[0]);
                        if b"*" == symbuf.as_ref() {
                            return Ok(Some(SymbolResult::Direct(r.clone())));
                        } else if opname == symbuf.as_ref() {
                            return Ok(Some(SymbolResult::Matched(symbol, value)));
                        }
                    }

                    ASExp::Pair(_, _) => {}
                }
            }
        }
    }

    Ok(None)
}

pub type SplitRestResult<A> = Option<(
    Vec<<A as ClassicAllocator>::NodePtr>,
    Option<<A as ClassicAllocator>::NodePtr>,
)>;

fn split_rest_tail<A: ClassicAllocator>(
    allocator: &A,
    args: &A::NodePtr,
) -> Result<SplitRestResult<A>, ClError>
where
    A::NodePtr: Clone,
{
    let Some(mut args) = proper_list(allocator, args, true).map(|args| args.to_vec()) else {
        return Ok(None);
    };

    let rest_index = args.iter().position(|arg| match allocator.sexp(arg) {
        ASExp::Atom => allocator.atom(arg).as_ref() == b"&rest",
        ASExp::Pair(_, _) => false,
    });

    let Some(rest_index) = rest_index else {
        return Ok(Some((args, None)));
    };

    if rest_index + 2 != args.len() {
        return Err(ClError(
            allocator.loc(&args[rest_index]),
            EvalErr::InternalError(
                allocator.export(&args[rest_index]),
                "&rest must be followed by exactly one tail expression".to_string(),
            ),
        ));
    }

    let tail = args.pop();
    args.pop();
    Ok(Some((args, tail)))
}

fn enlist_with_tail<A: ClassicAllocator>(
    allocator: &mut A,
    args: &[A::NodePtr],
    tail: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let mut result = tail.clone();
    for arg in args.iter().rev() {
        result = allocator.new_pair(allocator.loc(arg), arg, &result)?;
    }
    Ok(result)
}

fn rest_argument_source<A: ClassicAllocator>(
    allocator: &mut A,
    args: &[A::NodePtr],
    tail: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let mut result = tail.clone();
    for arg in args.iter().rev() {
        let loc = allocator.loc(arg);
        let cons = allocator.new_atom(loc.clone(), b"c")?;
        result = enlist(allocator, &[cons, arg.clone(), result])?;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn compile_application<A: ClassicAllocator>(
    allocator: &mut A,
    prog: &A::NodePtr,
    operator: &A::NodePtr,
    opbuf: &[u8],
    rest: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    let mut compiled_args = vec![operator.clone()];

    let loc = allocator.loc(prog);
    let exported_prog = allocator.export(prog);
    let error_result = Err(ClError(
        loc,
        EvalErr::InternalError(
            exported_prog,
            format!(
                "can't compile {}, unknown operator",
                allocator.disassemble(prog, None)
            ),
        ),
    ));

    if *opbuf == vec![1] || *opbuf == vec![b'q'] {
        let rest_loc = allocator.loc(rest);
        return allocator.new_pair(rest_loc, operator, rest);
    }

    match split_rest_tail(allocator, rest)? {
        Some((prog_args, tail_arg)) => {
            let mut new_args = map_m(allocator, &mut prog_args.iter(), &|allocator, arg| {
                do_com_prog(
                    allocator,
                    544,
                    arg,
                    macro_lookup,
                    symbol_table,
                    run_program.clone(),
                )
            })?;

            let compiled_tail = match &tail_arg {
                Some(tail) => do_com_prog(
                    allocator,
                    544,
                    tail,
                    macro_lookup,
                    symbol_table,
                    run_program.clone(),
                )?,
                None => allocator.import(allocator.loc(rest), NodePtr::NIL)?,
            };
            compiled_args.append(&mut new_args);
            let r = enlist_with_tail(allocator, &compiled_args, &compiled_tail)?;

            if PASS_THROUGH_OPERATORS.contains(opbuf) || (!opbuf.is_empty() && opbuf[0] == b'_') {
                Ok(r)
            } else {
                find_symbol_match(allocator, opbuf, &r, symbol_table).and_then(|x| match x {
                    Some(SymbolResult::Direct(v)) => Ok(v),
                    Some(SymbolResult::Matched(_symbol, value)) => {
                        let loc = allocator.loc(&value);
                        let apply_atom = allocator.new_atom(loc.clone(), &[2])?;
                        let list_atom = allocator.new_atom(loc.clone(), "list".as_bytes())?;
                        let cons_atom = allocator.new_atom(loc.clone(), &[4])?;
                        let com_atom = allocator.new_atom(loc.clone(), "com".as_bytes())?;
                        let opt_atom = allocator.new_atom(loc.clone(), "opt".as_bytes())?;
                        let top_atom = allocator
                            .new_atom(loc.clone(), NodePath::new(None).as_path().data())?;
                        let left_atom = allocator
                            .new_atom(loc.clone(), NodePath::new(None).first().as_path().data())?;
                        let argument_source = match &tail_arg {
                            Some(tail) => rest_argument_source(allocator, &prog_args, tail)?,
                            None => {
                                let enlisted = enlist(allocator, &prog_args)?;
                                allocator.new_pair(loc, &list_atom, &enlisted)?
                            }
                        };
                        let quoted_list = quote(allocator, &argument_source)?;
                        let quoted_macros = quote(allocator, macro_lookup)?;
                        let quoted_symbols = quote(allocator, symbol_table)?;
                        let compiled = enlist(
                            allocator,
                            &[com_atom, quoted_list, quoted_macros, quoted_symbols],
                        )?;
                        let to_run = enlist(allocator, &[opt_atom, compiled])?;
                        let new_args = evaluate(allocator, &to_run, &top_atom)?;
                        let cons_enlisted = enlist(allocator, &[cons_atom, left_atom, new_args])?;
                        enlist(allocator, &[apply_atom, value, cons_enlisted])
                    }
                    None => error_result,
                })
            }
        }
        None => error_result,
    }
}

pub fn do_com_prog<A: ClassicAllocator>(
    allocator: &mut A,
    from: usize,
    prog: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    if DIAG_OUTPUT {
        println!(
            "START COMPILE {}: {}\nMACRO {}\nSYMBOLS {}",
            from,
            allocator.disassemble(prog, None),
            allocator.disassemble(macro_lookup, None),
            allocator.disassemble(symbol_table, None),
        );
    }
    do_com_prog_(allocator, prog, macro_lookup, symbol_table, run_program).inspect(|x| {
        if DIAG_OUTPUT {
            println!(
                "DO_COM_PROG {}: {}\nMACRO {}\nSYMBOLS {}\nRESULT {}",
                from,
                allocator.disassemble(prog, None),
                allocator.disassemble(macro_lookup, None),
                allocator.disassemble(symbol_table, None),
                allocator.disassemble(x, None)
            );
        }
    })
}

fn do_com_prog_<A: ClassicAllocator>(
    allocator: &mut A,
    prog_: &A::NodePtr,
    macro_lookup: &A::NodePtr,
    symbol_table: &A::NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    /*
     * Turn the given program `prog` into a clvm program using
     * the macros to do transformation.
     * prog is an uncompiled s-expression.
     * Return a new expanded s-expression PROG_EXP that is equivalent by rewriting
     * based upon the operator, where "equivalent" means
     * (a (com (q PROG) (MACROS)) ARGS) == (a (q PROG_EXP) ARGS)
     * for all ARGS.
     * Also, (opt (com (q PROG) (MACROS))) == (opt (com (q PROG_EXP) (MACROS)))
     */

    // lower "quote" to "q"
    m! {
        prog <- lower_quote(allocator, prog_);

        // quote atoms
        match allocator.sexp(&prog) {
            ASExp::Atom => {
                // Note: can't co-borrow with allocator below.
                let prog_atom = allocator.atom(&prog);
                transform_program_atom(
                    allocator,
                    &prog,
                    // This is a false positive due to Allocator lifetime.
                    #[allow(clippy::unnecessary_to_owned)]
                    &prog_atom.as_ref().to_vec(),
                    symbol_table
                )
            },
            ASExp::Pair(operator,prog_rest) => {
                match allocator.sexp(&operator) {
                    ASExp::Atom => {
                        // Note: can't co-borrow with allocator below.
                        let op_atom = allocator.atom(&operator);
                        let op_buf = op_atom.as_ref().to_vec();
                        get_macro_program(allocator, &op_buf, macro_lookup).
                            and_then(|x| match x {
                                Some(value) => {
                                    try_expand_macro_for_atom(
                                        allocator,
                                        &value,
                                        &prog_rest,
                                        macro_lookup,
                                        symbol_table
                                    )
                                },
                                None => {
                                    compile_operator_atom(
                                        allocator,
                                        &prog,
                                        &op_buf,
                                        macro_lookup,
                                        symbol_table,
                                        run_program.clone()
                                    ).and_then(|x| x.map(Ok).unwrap_or_else(|| m! {
                                        compile_application(
                                            allocator,
                                            &prog,
                                            &operator,
                                            &op_buf,
                                            &prog_rest,
                                            macro_lookup,
                                            symbol_table,
                                            run_program.clone()
                                        )
                                    }))
                                }
                            })
                    },
                    _ => {
                        // (com ((OP) . RIGHT)) => (a (com (q OP)) 1)
                        let loc = allocator.loc(&operator);
                        let com_atom = allocator.new_atom(loc.clone(), "com".as_bytes())?;
                        let quoted_op = quote(allocator, &operator)?;
                        let quoted_macro_lookup = quote(allocator, macro_lookup)?;
                        let quoted_symbol_table = quote(allocator, symbol_table)?;
                        let top_atom = allocator.new_atom(loc, NodePath::new(None).as_path().data())?;
                        let eval_list = enlist(allocator, &[
                            com_atom,
                            quoted_op,
                            quoted_macro_lookup,
                            quoted_symbol_table
                        ])?;
                        evaluate(
                            allocator, &eval_list, &top_atom
                        ).and_then(|x| enlist(allocator, &[x]))
                    }
                }
            }
        }
    }
}

pub fn do_com_prog_for_dialect<A: ClassicAllocator>(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut A,
    sexp: &A::NodePtr,
) -> Result<A::NodePtr, ClError>
where
    A::NodePtr: Clone,
{
    match allocator.sexp(sexp) {
        ASExp::Pair(prog, extras) => {
            let loc = allocator.loc(sexp);
            let imported_nil = allocator.import(loc, NodePtr::NIL)?;
            let mut symbol_table = imported_nil;
            let macro_lookup;

            let mut elist = Vec::new();
            if let Some(elist_vec) = proper_list(allocator, &extras, true) {
                elist = elist_vec.to_vec();
            }

            if elist.is_empty() {
                macro_lookup = default_macro_lookup(allocator, runner.clone());
            } else {
                macro_lookup = elist[0].clone();
                if elist.len() > 1 {
                    symbol_table = elist[1].clone();
                }
            }

            // XXX enable extra info in sym file.
            // let dequoted = dequote(allocator, prog);
            // let sexp_dis = disassemble(allocator, dequoted);

            do_com_prog(
                allocator,
                773,
                &prog,
                &macro_lookup,
                &symbol_table,
                runner.clone(),
            )
            //.map(|x| {
            // XXX Enable extra info in sym file.
            // self.compile_outcomes.replace_with(|co| {
            //     let key = sha256tree(allocator, x.1).hex();
            //     co.insert(key, sexp_dis);
            //     co.clone()
            // });
            // - or -
            // x
            //})
        }
        _ => {
            let exported = allocator.export(sexp);
            Err(ClError(
                allocator.loc(sexp),
                EvalErr::InternalError(
                    exported,
                    "Program is not a pair in do_com_prog".to_string(),
                ),
            ))
        }
    }
}

pub fn get_compile_filename(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut Allocator,
) -> Result<Option<String>, EvalErr> {
    let cvt_prog = assemble(allocator, "(_get_compile_filename)")?;

    let Reduction(_, cvt_prog_result) =
        runner.run_program(allocator, cvt_prog, NodePtr::NIL, None)?;

    if cvt_prog_result == NodePtr::NIL {
        return Ok(None);
    }

    if let SExp::Atom = allocator.sexp(cvt_prog_result) {
        // only cvt_prog_result in scope.
        let atom = allocator.atom(cvt_prog_result);
        return Ok(Some(
            Bytes::new(Some(BytesFromType::Raw(atom.as_ref().to_vec()))).decode(),
        ));
    }

    Err(EvalErr::InternalError(
        NodePtr::NIL,
        "Couldn't decode result filename".to_string(),
    ))
}

pub fn get_search_paths<A: ClassicAllocator>(
    runner: Rc<dyn TRunProgram>,
    loc: Srcloc,
    allocator: &mut A,
) -> Result<Vec<String>, ClError>
where
    A::NodePtr: Clone,
{
    let search_paths_result = ((|| {
        let search_paths_prog = assemble(allocator.allocator(), "(_get_include_paths)")?;
        runner.run_program(allocator.allocator(), search_paths_prog, NodePtr::NIL, None)
    })())
    .map_err(|e| ClError(loc.clone(), e))?;
    let mut res = Vec::new();
    let search_paths_result_import = allocator.import(loc, search_paths_result.1)?;
    if let Some(l) = proper_list(allocator, &search_paths_result_import, true) {
        for elt in l.iter() {
            if let ASExp::Atom = allocator.sexp(elt) {
                // Only elt in scope.
                let atom = allocator.atom(elt);
                res.push(Bytes::new(Some(BytesFromType::Raw(atom.as_ref().to_vec()))).decode());
            }
        }
    }

    Ok(res)
}

pub fn get_last_path_component(name: &str) -> String {
    let mut skip_start = None;
    let fnbytes = name.as_bytes();

    for (i, ch) in fnbytes.iter().enumerate() {
        if *ch == b'/' || *ch == b'\\' {
            skip_start = Some(i + 1);
        }
    }

    if let Some(skip) = skip_start {
        let namevec = fnbytes.iter().skip(skip).copied().collect();
        Bytes::new(Some(BytesFromType::Raw(namevec))).decode()
    } else {
        name.to_owned()
    }
}

pub fn make_symbols_name(current_filename: &str, name: &str) -> String {
    // Grab the final path component if these strings are composed
    // that way.
    let take_start = get_last_path_component(current_filename);
    let take_end = get_last_path_component(name);

    format!("{take_start}_{take_end}.sym")
}
