use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;
use clvm_rs::reduction::{Reduction, Response};

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm::sexp::{enlist, first, map_m, non_nil, proper_list, rest};
use crate::classic::clvm::OPERATORS_LATEST_VERSION;
use crate::classic::clvm::{keyword_from_atom, keyword_to_atom};

use crate::classic::clvm_tools::binutils::{assemble, disassemble};
use crate::classic::clvm_tools::node_path::NodePath;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::classic::clvm_tools::stages::stage_2::defaults::default_macro_lookup;
use crate::classic::clvm_tools::stages::stage_2::helpers::{brun, evaluate, quote};
use crate::classic::clvm_tools::stages::stage_2::module::compile_mod;

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

struct Closure<'a> {
    name: String,
    #[allow(clippy::type_complexity)]
    to_run: &'a dyn Fn(
        &mut Allocator,
        NodePtr,
        NodePtr,
        NodePtr,
        Rc<dyn TRunProgram>,
        usize,
    ) -> Result<NodePtr, EvalErr>,
}

fn compile_bindings<'a>() -> HashMap<Vec<u8>, Closure<'a>> {
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

fn com_qq(
    allocator: &mut Allocator,
    ident: String,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
    runner: Rc<dyn TRunProgram>,
    sexp: NodePtr,
) -> Result<NodePtr, EvalErr> {
    if DIAG_OUTPUT {
        println!("com_qq {} {}", ident, disassemble(allocator, sexp, None));
    }
    do_com_prog(allocator, 110, sexp, macro_lookup, symbol_table, runner).map(|x| x.1)
}

pub fn compile_qq(
    allocator: &mut Allocator,
    args: NodePtr,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
    runner: Rc<dyn TRunProgram>,
    level: usize,
) -> Result<NodePtr, EvalErr> {
    /*
     * (qq ATOM) => (q . ATOM)
     * (qq (unquote X)) => X
     * (qq (a . B)) => (c (qq a) (qq B))
     */

    let sexp = match first(allocator, args) {
        Err(e) => {
            return Err(e);
        }
        Ok(x) => x,
    };

    match allocator.sexp(sexp) {
        SExp::Atom => {
            // (qq ATOM) => (q . ATOM)
            quote(allocator, sexp)
        }
        SExp::Pair(op, sexp_rest) => {
            if let SExp::Atom = allocator.sexp(op) {
                // opbuf => op
                let op_atom = allocator.atom(op);
                if op_atom.as_ref() == qq_atom() {
                    return m! {
                        cons_atom <- allocator.new_atom(&[4]);
                        subexp <-
                            compile_qq(allocator, sexp_rest, macro_lookup, symbol_table, runner.clone(), level+1);
                        quoted_null <- quote(allocator, NodePtr::NIL);
                        consed <- enlist(allocator, &[cons_atom, subexp, quoted_null]);
                        run_list <- enlist(allocator, &[cons_atom, op, consed]);
                        com_qq(allocator, "qq sexp pair".to_string(), macro_lookup, symbol_table, runner, run_list)
                    };
                } else if op_atom.as_ref() == unquote_atom() {
                    // opbuf
                    if level == 1 {
                        // (qq (unquote X)) => X
                        return m! {
                            sexp_rf <- first(allocator, sexp_rest);
                            com_qq(allocator, "level 1".to_string(), macro_lookup, symbol_table, runner, sexp_rf)
                        };
                    }
                    return m! {
                        // (qq (a . B)) => (c (qq a) (qq B))
                        cons_atom <- allocator.new_atom(&[4]);
                        subexp <-
                            compile_qq(allocator, sexp_rest, macro_lookup, symbol_table, runner.clone(), level-1);
                        quoted_null <- quote(allocator, NodePtr::NIL);
                        consed_subexp <- enlist(allocator, &[cons_atom, subexp, quoted_null]);
                        run_list <- enlist(allocator, &[cons_atom, op, consed_subexp]);
                        com_qq(allocator, "qq pair general".to_string(), macro_lookup, symbol_table, runner, run_list)
                    };
                }
            }

            // (qq (a . B)) => (c (qq a) (qq B))
            m! {
                cons_atom <- allocator.new_atom(&[4]);
                qq <- allocator.new_atom(&qq_atom());
                qq_l <- enlist(allocator, &[qq, op]);
                qq_r <- enlist(allocator, &[qq, sexp_rest]);
                compiled_l <- com_qq(allocator, "A".to_string(), macro_lookup, symbol_table, runner.clone(), qq_l);
                compiled_r <- com_qq(allocator, "B".to_string(), macro_lookup, symbol_table, runner, qq_r);
                enlist(allocator, &[cons_atom, compiled_l, compiled_r])
            }
        }
    }
}

pub fn compile_macros(
    allocator: &mut Allocator,
    _args: NodePtr,
    macro_lookup: NodePtr,
    _symbol_table: NodePtr,
    _run_program: Rc<dyn TRunProgram>,
    _level: usize,
) -> Result<NodePtr, EvalErr> {
    quote(allocator, macro_lookup)
}

pub fn compile_symbols(
    allocator: &mut Allocator,
    _args: NodePtr,
    _macro_lookup: NodePtr,
    symbol_table: NodePtr,
    _run_program: Rc<dyn TRunProgram>,
    _level: usize,
) -> Result<NodePtr, EvalErr> {
    quote(allocator, symbol_table)
}

// # Transform "quote" to "q" everywhere. Note that quote will not be compiled if behind qq.
// # Overrides symbol table defns.
fn lower_quote_(allocator: &mut Allocator, prog: NodePtr) -> Result<NodePtr, EvalErr> {
    if !non_nil(allocator, prog) {
        return Ok(prog);
    }

    if let Some(qlist) = proper_list(allocator, prog, true) {
        if qlist.is_empty() {
            return Ok(prog);
        }

        // quote_node was Atom(q)
        let quote_node = qlist[0];
        if let SExp::Atom = allocator.sexp(quote_node) {
            let quote_atom = allocator.atom(quote_node);
            if quote_atom.as_ref() == b"quote" {
                if qlist.len() != 2 {
                    // quoted list should be 2: "(quote arg)"
                    return Err(EvalErr::InternalError(prog, format!("Compilation error while compiling [{}]. quote takes exactly one argument.", disassemble(allocator, prog, None))));
                }

                // Note: quote should have exactly one arg, so the length of
                return m! {
                    lowered <- lower_quote(allocator, qlist[1]);
                    quote(allocator, lowered)
                };
            }
        }
    }

    // XXX Note that this recognizes potentially unintended
    // syntax, in that (sha256 3 quote ()) is valid in this
    // code.  It is corrected in the new compiler but left
    // here in case this bug is exploited.
    // Like a good neighbor, UB is there☺
    if let SExp::Pair(f, r) = allocator.sexp(prog) {
        return m! {
            first <- lower_quote(allocator, f);
            rest <- lower_quote(allocator, r);
            allocator.new_pair(first, rest)
        };
    }

    Ok(prog)
}

pub fn lower_quote(allocator: &mut Allocator, prog: NodePtr) -> Result<NodePtr, EvalErr> {
    let res = lower_quote_(allocator, prog);
    if DIAG_OUTPUT {
        res.as_ref()
            .map(|x| {
                println!(
                    "LOWER_QUOTE {} TO {}",
                    disassemble(allocator, prog, None),
                    disassemble(allocator, *x, None)
                );
            })
            .unwrap_or_else(|_| ())
    }
    res
}

fn try_expand_macro_for_atom_(
    allocator: &mut Allocator,
    macro_code: NodePtr,
    prog_rest: NodePtr,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
) -> Response {
    m! {
        com_atom <- allocator.new_atom("com".as_bytes());
        post_prog <- brun(allocator, macro_code, prog_rest);

        quoted_macros <- quote(allocator, macro_lookup);
        quoted_symbols <- quote(allocator, symbol_table);
        to_eval <- enlist(
            allocator,
            &[
                com_atom,
                post_prog,
                quoted_macros,
                quoted_symbols
            ]
        );
        top_path <- allocator.new_atom(NodePath::new(None).as_path().data());
        evaluate(
            allocator,
            to_eval,
            top_path
        ).map(|x| {
            if DIAG_OUTPUT {
                print!(
                    "TRY_EXPAND_MACRO {} WITH {} GIVES {} MACROS {} SYMBOLS {}",
                    disassemble(allocator, macro_code, None),
                    disassemble(allocator, prog_rest, None),
                    disassemble(allocator, x, None),
                    disassemble(allocator, macro_lookup, None),
                    disassemble(allocator, symbol_table, None)
                );
            }
            Reduction(1, x)
        })
    }
}

pub fn try_expand_macro_for_atom(
    allocator: &mut Allocator,
    macro_code: NodePtr,
    prog_rest: NodePtr,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
) -> Response {
    m! {
        res <- try_expand_macro_for_atom_(
            allocator,
            macro_code,
            prog_rest,
            macro_lookup,
            symbol_table
        );
        Ok(res)
    }
}

fn get_macro_program(
    allocator: &Allocator,
    operator: &[u8],
    macro_lookup: NodePtr,
) -> Result<Option<NodePtr>, EvalErr> {
    if let Some(mlist) = proper_list(allocator, macro_lookup, true) {
        for macro_pair in mlist {
            match proper_list(allocator, macro_pair, true) {
                None => {}
                Some(mp_list) => {
                    if mp_list.is_empty() {
                        continue;
                    }
                    let value = if mp_list.len() > 1 {
                        mp_list[1]
                    } else {
                        NodePtr::NIL
                    };

                    match allocator.sexp(mp_list[0]) {
                        SExp::Atom => {
                            // was macro_name, but it's singular and probably
                            // not useful to rename.
                            let atom = allocator.atom(mp_list[0]);
                            if atom.as_ref() == operator {
                                return Ok(Some(value));
                            }
                        }
                        SExp::Pair(_, _) => {
                            continue;
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

fn transform_program_atom(
    allocator: &mut Allocator,
    prog: NodePtr,
    a: &[u8],
    symbol_table: NodePtr,
) -> Response {
    if a == b"@" {
        return allocator
            .new_atom(NodePath::new(None).as_path().data())
            .map(|x| Reduction(1, x));
    }
    match proper_list(allocator, symbol_table, true) {
        None => {}
        Some(symlist) => {
            for sym in symlist {
                match proper_list(allocator, sym, true) {
                    None => {}
                    Some(v) => {
                        if v.is_empty() {
                            continue;
                        }

                        let value = if v.len() > 1 { v[1] } else { NodePtr::NIL };

                        match allocator.sexp(v[0]) {
                            SExp::Atom => {
                                // v[0] is close by, and probably not useful to
                                // rename here.
                                let atom = allocator.atom(v[0]);
                                if atom.as_ref() == a {
                                    return Ok(Reduction(1, value));
                                }
                            }
                            SExp::Pair(_, _) => {}
                        }
                    }
                }
            }
        }
    }

    quote(allocator, prog).map(|x| Reduction(1, x))
}

fn compile_operator_atom(
    allocator: &mut Allocator,
    prog: NodePtr,
    avec: &[u8],
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Result<Option<NodePtr>, EvalErr> {
    let compile_bindings = compile_bindings();

    if *avec == vec![1] {
        return Ok(Some(prog));
    }

    if let Some(f) = compile_bindings.get(avec) {
        return m! {
            prog_rest <- rest(allocator, prog);
            post_prog <-
                (f.to_run)(
                    allocator,
                    prog_rest,
                    macro_lookup,
                    symbol_table,
                    run_program.clone(),
                    1
                );
            quoted_post_prog <- quote(allocator, post_prog);
            top_atom <-
                allocator.new_atom(NodePath::new(None).as_path().data());

            let _ = if DIAG_OUTPUT {
                print!("COMPILE_BINDINGS {}", disassemble(allocator, quoted_post_prog, None));
            };
            evaluate(allocator, quoted_post_prog, top_atom).map(Some)
        };
    }

    Ok(None)
}

enum SymbolResult {
    Direct(NodePtr),
    Matched(NodePtr, NodePtr),
}

fn find_symbol_match(
    allocator: &Allocator,
    opname: &[u8],
    r: NodePtr,
    symbol_table: NodePtr,
) -> Result<Option<SymbolResult>, EvalErr> {
    if let Some(symlist) = proper_list(allocator, symbol_table, true) {
        for sym in symlist {
            if let Some(symdef) = proper_list(allocator, sym, true) {
                if symdef.is_empty() {
                    continue;
                }

                match allocator.sexp(symdef[0]) {
                    SExp::Atom => {
                        let symbol = symdef[0];
                        let value = if symdef.len() == 1 {
                            NodePtr::NIL
                        } else {
                            symdef[1]
                        };

                        let symbuf = allocator.atom(symdef[0]);
                        if b"*" == symbuf.as_ref() {
                            return Ok(Some(SymbolResult::Direct(r)));
                        } else if opname == symbuf.as_ref() {
                            return Ok(Some(SymbolResult::Matched(symbol, value)));
                        }
                    }

                    SExp::Pair(_, _) => {}
                }
            }
        }
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn compile_application(
    allocator: &mut Allocator,
    prog: NodePtr,
    operator: NodePtr,
    opbuf: &[u8],
    rest: NodePtr,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    let mut compiled_args = vec![operator];

    let error_result = Err(EvalErr::InternalError(
        prog,
        format!(
            "can't compile {}, unknown operator",
            disassemble(allocator, prog, None)
        ),
    ));

    if *opbuf == vec![1] || *opbuf == vec![b'q'] {
        return allocator.new_pair(operator, rest);
    }

    match proper_list(allocator, rest, true) {
        Some(prog_args) => {
            let mut new_args = map_m(allocator, &mut prog_args.iter(), &|allocator, arg| {
                do_com_prog(
                    allocator,
                    544,
                    *arg,
                    macro_lookup,
                    symbol_table,
                    run_program.clone(),
                )
                .map(|x| x.1)
            })?;

            compiled_args.append(&mut new_args);
            let r = enlist(allocator, &compiled_args)?;

            if PASS_THROUGH_OPERATORS.contains(opbuf) || (!opbuf.is_empty() && opbuf[0] == b'_') {
                Ok(r)
            } else {
                find_symbol_match(
                    allocator,
                    opbuf,
                    r,
                    symbol_table
                ).and_then(|x| match x {
                    Some(SymbolResult::Direct(v)) => { Ok(v) },
                    Some(SymbolResult::Matched(_symbol,value)) => {
                        match proper_list(allocator, rest, true) {
                            Some(proglist) => {
                                m! {
                                    apply_atom <- allocator.new_atom(&[2]);
                                    list_atom <- allocator.new_atom("list".as_bytes());
                                    cons_atom <- allocator.new_atom(&[4]);
                                    com_atom <- allocator.new_atom("com".as_bytes());
                                    opt_atom <- allocator.new_atom("opt".as_bytes());
                                    top_atom <- allocator.new_atom(NodePath::new(None).as_path().data());
                                    left_atom <- allocator.new_atom(NodePath::new(None).first().as_path().data());

                                    enlisted <- enlist(allocator, &proglist);
                                    list_application <- allocator.new_pair(list_atom, enlisted);

                                    quoted_list <- quote(allocator, list_application);
                                    quoted_macros <- quote(allocator, macro_lookup);
                                    quoted_symbols <- quote(allocator, symbol_table);
                                    compiled <- enlist(allocator, &[com_atom, quoted_list, quoted_macros, quoted_symbols]);
                                    to_run <- enlist(allocator, &[opt_atom, compiled]);
                                    new_args <- evaluate(allocator, to_run, top_atom);

                                    cons_enlisted <- enlist(allocator, &[cons_atom, left_atom, new_args]);

                                    result <- enlist(
                                        allocator,
                                        &[apply_atom, value, cons_enlisted]
                                    );

                                    Ok(result)
                                }
                            },
                            None => { error_result }
                        }
                    },
                    None => { error_result }
                })
            }
        }
        None => error_result,
    }
}

pub fn do_com_prog(
    allocator: &mut Allocator,
    from: usize,
    prog: NodePtr,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Response {
    if DIAG_OUTPUT {
        println!(
            "START COMPILE {}: {} MACRO {} SYMBOLS {}",
            from,
            disassemble(allocator, prog, None),
            disassemble(allocator, macro_lookup, None),
            disassemble(allocator, symbol_table, None),
        );
    }
    do_com_prog_(allocator, prog, macro_lookup, symbol_table, run_program).inspect(|x| {
        if DIAG_OUTPUT {
            println!(
                "DO_COM_PROG {}: {} MACRO {} SYMBOLS {} RESULT {}",
                from,
                disassemble(allocator, prog, None),
                disassemble(allocator, macro_lookup, None),
                disassemble(allocator, symbol_table, None),
                disassemble(allocator, x.1, None)
            );
        }
    })
}

fn do_com_prog_(
    allocator: &mut Allocator,
    prog_: NodePtr,
    macro_lookup: NodePtr,
    symbol_table: NodePtr,
    run_program: Rc<dyn TRunProgram>,
) -> Response {
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
        match allocator.sexp(prog) {
            SExp::Atom => {
                // Note: can't co-borrow with allocator below.
                let prog_atom = allocator.atom(prog);
                transform_program_atom(
                    allocator,
                    prog,
                    // This is a false positive due to Allocator lifetime.
                    #[allow(clippy::unnecessary_to_owned)]
                    &prog_atom.as_ref().to_vec(),
                    symbol_table
                )
            },
            SExp::Pair(operator,prog_rest) => {
                match allocator.sexp(operator) {
                    SExp::Atom => {
                        // Note: can't co-borrow with allocator below.
                        let op_atom = allocator.atom(operator);
                        let op_buf = op_atom.as_ref().to_vec();
                        get_macro_program(allocator, op_atom.as_ref(), macro_lookup).
                            and_then(|x| match x {
                                Some(value) => {
                                    try_expand_macro_for_atom(
                                        allocator,
                                        value,
                                        prog_rest,
                                        macro_lookup,
                                        symbol_table
                                    )
                                },
                                None => {
                                    compile_operator_atom(
                                        allocator,
                                        prog,
                                        &op_buf,
                                        macro_lookup,
                                        symbol_table,
                                        run_program.clone()
                                    ).and_then(|x| x.map(Ok).unwrap_or_else(|| m! {
                                        compile_application(
                                            allocator,
                                            prog,
                                            operator,
                                            &op_buf,
                                            prog_rest,
                                            macro_lookup,
                                            symbol_table,
                                            run_program.clone()
                                        )
                                    })).map(|x| Reduction(1, x))
                                }
                            })
                    },
                    _ => {
                        // (com ((OP) . RIGHT)) => (a (com (q OP)) 1)
                        m! {
                            com_atom <- allocator.new_atom("com".as_bytes());
                            quoted_op <- quote(allocator, operator);
                            quoted_macro_lookup <-
                                quote(allocator, macro_lookup);
                            quoted_symbol_table <-
                                quote(allocator, symbol_table);
                            top_atom <- allocator.new_atom(NodePath::new(None).as_path().data());
                            eval_list <- enlist(allocator, &[
                                com_atom,
                                quoted_op,
                                quoted_macro_lookup,
                                quoted_symbol_table
                            ]);

                            evaluate(
                                allocator, eval_list, top_atom
                            ).and_then(|x| enlist(allocator, &[x])).
                                map(|x| Reduction(1, x))
                        }
                    }
                }
            }
        }
    }
}

pub fn do_com_prog_for_dialect(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut Allocator,
    sexp: NodePtr,
) -> Response {
    match allocator.sexp(sexp) {
        SExp::Pair(prog, extras) => {
            let mut symbol_table = NodePtr::NIL;
            let macro_lookup;

            let mut elist = Vec::new();
            if let Some(elist_vec) = proper_list(allocator, extras, true) {
                elist = elist_vec.to_vec();
            }

            if elist.is_empty() {
                macro_lookup = default_macro_lookup(allocator, runner.clone());
            } else {
                macro_lookup = elist[0];
                if elist.len() > 1 {
                    symbol_table = elist[1];
                }
            }

            // XXX enable extra info in sym file.
            // let dequoted = dequote(allocator, prog);
            // let sexp_dis = disassemble(allocator, dequoted);

            do_com_prog(
                allocator,
                773,
                prog,
                macro_lookup,
                symbol_table,
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
        _ => Err(EvalErr::InternalError(
            sexp,
            "Program is not a pair in do_com_prog".to_string(),
        )),
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

pub fn get_search_paths(
    runner: Rc<dyn TRunProgram>,
    allocator: &mut Allocator,
) -> Result<Vec<String>, EvalErr> {
    let search_paths_prog = assemble(allocator, "(_get_include_paths)")?;
    let search_path_result =
        runner.run_program(allocator, search_paths_prog, NodePtr::NIL, None)?;

    let mut res = Vec::new();
    if let Some(l) = proper_list(allocator, search_path_result.1, true) {
        for elt in l.iter().copied() {
            if let SExp::Atom = allocator.sexp(elt) {
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
