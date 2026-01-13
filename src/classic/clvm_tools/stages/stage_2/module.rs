use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::error::EvalErr;

use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::classic::clvm::sexp::{
    enlist, first, flatten, fold_m, map_m, non_nil, nonempty_last, proper_list, rest, First,
    NodeSel, Rest, SelectNode, ThisNode,
};
use crate::classic::clvm_tools::binutils::disassemble;
use crate::classic::clvm_tools::debug::{build_symbol_dump, FunctionExtraInfo};
use crate::classic::clvm_tools::node_path::NodePath;
use crate::classic::clvm_tools::stages::assemble;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::classic::clvm_tools::stages::stage_2::helpers::{evaluate, quote};
use crate::classic::clvm_tools::stages::stage_2::inline::{
    formulate_path_selections_for_destructuring, is_at_capture, is_inline_destructure,
};
use crate::classic::clvm_tools::stages::stage_2::optimize::optimize_sexp;
use crate::classic::clvm_tools::stages::stage_2::reader::process_embed_file;

lazy_static! {
    pub static ref MAIN_NAME: String = "".to_string();
}

struct CollectionResult {
    pub functions: HashMap<Vec<u8>, NodePtr>,
    pub constants: HashMap<Vec<u8>, NodePtr>,
    pub macros: Vec<(Vec<u8>, NodePtr)>,
}

#[derive(Default)]
struct CompileOutput {
    pub functions: HashMap<Vec<u8>, NodePtr>,
    pub symbols_extra_info: HashMap<Vec<u8>, FunctionExtraInfo>,
}

impl CompileOutput {
    pub fn add_definitions(&mut self, other: &CompileOutput) {
        for (n, v) in other.functions.iter() {
            self.functions.insert(n.to_vec(), *v);
        }
        for (n, v) in other.symbols_extra_info.iter() {
            self.symbols_extra_info.insert(n.to_vec(), v.clone());
        }
    }
}

// export type TBuildTree = Bytes | Tuple<TBuildTree, TBuildTree> | [];
fn build_tree(allocator: &mut Allocator, items: &[Vec<u8>]) -> Result<NodePtr, EvalErr> {
    if items.is_empty() {
        Ok(NodePtr::NIL)
    } else if items.len() == 1 {
        allocator.new_atom(&items[0])
    } else {
        m! {
            let half_size = items.len() >> 1;
            left <- build_tree(allocator, &items[..half_size]);
            right <- build_tree(allocator, &items[half_size..]);
            allocator.new_pair(left, right)
        }
    }
}

// export type TBuildTreeProgram = SExp | [Bytes, TBuildTree, TBuildTree] | [Tuple<Bytes, SExp>];
fn build_tree_program(allocator: &mut Allocator, items: &[NodePtr]) -> Result<NodePtr, EvalErr> {
    // This function takes a Python list of items and turns it into a program that
    //  a binary tree of the items, suitable for casting to an s-expression.
    let size = items.len();
    if size == 0 {
        m! {
            list_of_nil <- enlist(allocator, &[NodePtr::NIL]);
            quote(allocator, list_of_nil)
        }
    } else if size == 1 {
        Ok(items[0])
    } else {
        m! {
            let half_size = items.len() >> 1;
            left <-
                build_tree_program(allocator, &items[..half_size]);
            right <-
                build_tree_program(allocator, &items[half_size..]);

            cons_atom <- allocator.new_atom(&[4_u8]);
            enlist(allocator, &[cons_atom, left, right])
        }
    }
}

/**
 * @return Used constants name array in `hex string` format.
 */
fn build_used_constants_names(
    allocator: &mut Allocator,
    functions: &HashMap<Vec<u8>, NodePtr>,
    constants: &HashMap<Vec<u8>, NodePtr>,
    macros: &[(Vec<u8>, NodePtr)],
) -> Result<Vec<Vec<u8>>, EvalErr> {
    /*
    Do a naïve pruning of unused symbols. It may be too big, but it shouldn't
    be too small. Return a list of all atoms used that are also the names of
    functions or constants, starting with the MAIN_NAME function.
     */
    let mut macro_as_dict = HashMap::new();

    for nv in macros {
        let (name, value) = nv;
        macro_as_dict.insert(name.to_vec(), *value);
    }

    let mut possible_symbols = HashSet::new();
    for key in functions.keys() {
        possible_symbols.insert(key);
    }

    for key in constants.keys() {
        possible_symbols.insert(key);
    }

    let mut new_names: HashSet<Vec<u8>> = HashSet::new();
    new_names.insert(MAIN_NAME.as_bytes().to_vec());
    let mut used_names = new_names.clone();

    while !new_names.is_empty() {
        let iterate_names = new_names.clone();
        new_names = HashSet::new();

        for name in iterate_names {
            let functions_and_macros = [functions.get(&name), macro_as_dict.get(&name)];

            let matching_names_1 = functions_and_macros
                .iter()
                .flat_map(|v| {
                    v.map(|v| {
                        let mut res = Vec::new();
                        flatten(allocator, *v, &mut res);
                        res
                    })
                    .unwrap_or_default()
                })
                .collect::<Vec<NodePtr>>();

            let matching_names = matching_names_1.iter().filter_map(|v| {
                // Only v usefully in scope.
                if let SExp::Atom = allocator.sexp(*v) {
                    let atom = allocator.atom(*v);
                    Some(atom.as_ref().to_vec())
                } else {
                    None
                }
            });

            for name in matching_names {
                if !used_names.contains(&name) {
                    used_names.insert(name.to_vec());
                    new_names.insert(name);
                }
            }
        }
    }

    // used_names.intersection_update(possible_symbols)
    let mut used_name_list: Vec<Vec<u8>> = Vec::new();
    for name in used_names.iter() {
        if possible_symbols.contains(name) && *name != MAIN_NAME.as_bytes() {
            used_name_list.push(name.to_vec());
        }
    }

    used_name_list.sort();
    Ok(used_name_list)
}

#[allow(clippy::too_many_arguments)]
fn parse_include(
    allocator: &mut Allocator,
    name: NodePtr,
    namespace: &mut HashSet<Vec<u8>>,
    functions: &mut HashMap<Vec<u8>, NodePtr>,
    constants: &mut HashMap<Vec<u8>, NodePtr>,
    delayed_constants: &mut HashMap<Vec<u8>, NodePtr>,
    macros: &mut Vec<(Vec<u8>, NodePtr)>,
    run_program: Rc<dyn TRunProgram>,
) -> Result<(), EvalErr> {
    m! {
        prog <- assemble(
            allocator,
            "(_read (_full_path_for_name 1))"
        );
        assembled_sexp <- run_program.run_program(
            allocator,
            prog,
            name,
            None
        );
        match proper_list(allocator, assembled_sexp.1, true) {
            None => { Err(EvalErr::InternalError(name, "include returned malformed result".to_string())) },
            Some(assembled) => {
                for sexp in assembled {
                    parse_mod_sexp(
                        allocator,
                        sexp,
                        namespace,
                        functions,
                        constants,
                        delayed_constants,
                        macros,
                        run_program.clone()
                    )?;
                };
                Ok(())
            }
        }
    }
}

fn unquote_args(
    allocator: &mut Allocator,
    code: NodePtr,
    args: &[Vec<u8>],
    matches: &HashMap<Vec<u8>, NodePtr>,
) -> Result<NodePtr, EvalErr> {
    match allocator.sexp(code) {
        SExp::Atom => {
            // Only code in scope.
            let code_atom = allocator.atom(code);
            let matching_args = args
                .iter()
                .filter(|arg| *arg == code_atom.as_ref())
                .cloned()
                .collect::<Vec<Vec<u8>>>();
            if !matching_args.is_empty() {
                if let Some(argval) = matches.get(&matching_args[0]) {
                    // New case: if we've been given an alternate way of computing
                    // the argument, use it here.
                    return Ok(*argval);
                }

                let unquote_atom = allocator.new_atom("unquote".as_bytes())?;
                return enlist(allocator, &[unquote_atom, code]);
            }

            Ok(code)
        }
        SExp::Pair(c1, c2) => {
            m! {
                unquoted_c2 <- unquote_args(allocator, c2, args, matches);
                unquoted_c1 <- unquote_args(allocator, c1, args, matches);
                allocator.new_pair(unquoted_c1, unquoted_c2)
            }
        }
    }
}

fn defun_inline_to_macro(
    allocator: &mut Allocator,
    declaration_sexp: NodePtr,
) -> Result<NodePtr, EvalErr> {
    let Rest::Here(NodeSel::Cons(name, NodeSel::Cons(arg_spec, First::Here(code)))) =
        Rest::Here(NodeSel::Cons(
            ThisNode::Here,
            NodeSel::Cons(ThisNode::Here, First::Here(ThisNode::Here)),
        ))
        .select_nodes(allocator, declaration_sexp)?;
    let defmacro_atom = allocator.new_atom("defmacro".as_bytes())?;

    let mut destructure_matches = HashMap::new();
    let use_args = if is_inline_destructure(allocator, arg_spec) {
        // Given an attempt to destructure via the argument list, we need
        // to ensure that the inline function receives arguments that are
        // relative to the _values_ given rather than the code given to
        // generate the arguments.  These overlap when the argument list is
        // a single level proper list, but not otherwise.
        formulate_path_selections_for_destructuring(allocator, arg_spec, &mut destructure_matches)?
    } else {
        arg_spec
    };

    let mut r_vec = vec![defmacro_atom, name, use_args];

    let mut arg_atom_list = Vec::new();
    flatten(allocator, use_args, &mut arg_atom_list);
    let arg_name_list = arg_atom_list
        .iter()
        .filter_map(|x| {
            if let SExp::Atom = allocator.sexp(*x) {
                // only x usefully in scope.
                Some(allocator.atom(*x))
            } else {
                None
            }
        })
        .filter(|x| !x.as_ref().is_empty())
        .map(|v| v.as_ref().to_vec())
        .collect::<Vec<Vec<u8>>>();

    let unquoted_code = unquote_args(allocator, code, &arg_name_list, &destructure_matches)?;

    let qq_atom = allocator.new_atom("qq".as_bytes())?;
    let qq_list = enlist(allocator, &[qq_atom, unquoted_code])?;
    r_vec.push(qq_list);
    let res = enlist(allocator, &r_vec)?;
    Ok(res)
}

#[allow(clippy::too_many_arguments)]
fn parse_mod_sexp(
    allocator: &mut Allocator,
    declaration_sexp: NodePtr,
    namespace: &mut HashSet<Vec<u8>>,
    functions: &mut HashMap<Vec<u8>, NodePtr>,
    constants: &mut HashMap<Vec<u8>, NodePtr>,
    // Delayed constants are new: they represent constant values
    // but we need the whole module to evaluate them since they
    // may call local functions (such as sha256tree).
    delayed_constants: &mut HashMap<Vec<u8>, NodePtr>,
    macros: &mut Vec<(Vec<u8>, NodePtr)>,
    run_program: Rc<dyn TRunProgram>,
) -> Result<(), EvalErr> {
    let NodeSel::Cons(op_node, First::Here(name_node)) =
        NodeSel::Cons(ThisNode::Here, First::Here(ThisNode::Here))
            .select_nodes(allocator, declaration_sexp)?;

    let op = match allocator.sexp(op_node) {
        // op_node in use.
        SExp::Atom => {
            let atom = allocator.atom(op_node);
            atom.as_ref().to_vec()
        }
        _ => Vec::new(),
    };
    let name = match allocator.sexp(name_node) {
        // name_node in use.
        SExp::Atom => {
            let atom = allocator.atom(name_node);
            atom.as_ref().to_vec()
        }
        _ => Vec::new(),
    };

    if op == "include".as_bytes() {
        parse_include(
            allocator,
            name_node,
            namespace,
            functions,
            constants,
            delayed_constants,
            macros,
            run_program.clone(),
        )
    } else if op == "embed-file".as_bytes() {
        let (name, constant) =
            process_embed_file(allocator, run_program.clone(), declaration_sexp)?;
        constants.insert(name, constant);
        Ok(())
    } else if namespace.contains(&name) {
        Err(EvalErr::InternalError(
            declaration_sexp,
            format!(
                "symbol \"{}\" redefined",
                Bytes::new(Some(BytesFromType::Raw(name))).decode()
            ),
        ))
    } else {
        namespace.insert(name.to_vec());

        if op == "defmacro".as_bytes() {
            macros.push((name.to_vec(), declaration_sexp));
            Ok(())
        } else if op == "defun".as_bytes() {
            let Rest::Here(Rest::Here(declaration_sexp_rr)) =
                Rest::Here(Rest::Here(ThisNode::Here)).select_nodes(allocator, declaration_sexp)?;
            functions.insert(name, declaration_sexp_rr);
            Ok(())
        } else if op == "defun-inline".as_bytes() {
            let defined_macro = defun_inline_to_macro(allocator, declaration_sexp)?;
            macros.push((name, defined_macro));
            Ok(())
        } else if op == "defconstant".as_bytes() {
            let Rest::Here(Rest::Here(First::Here(frr_of_declaration))) =
                Rest::Here(Rest::Here(First::Here(ThisNode::Here)))
                    .select_nodes(allocator, declaration_sexp)?;
            let quoted_decl = quote(allocator, frr_of_declaration)?;
            constants.insert(name, quoted_decl);
            Ok(())
        } else if op == "defconst".as_bytes() {
            // Use a new type-based match language.
            let Rest::Here(Rest::Here(First::Here(definition))) =
                Rest::Here(Rest::Here(First::Here(ThisNode::Here)))
                    .select_nodes(allocator, declaration_sexp)?;
            delayed_constants.insert(name, definition);
            Ok(())
        } else {
            Err(EvalErr::InternalError(
                declaration_sexp,
                "expected defun, defmacro, defconst, compile-file or defconstant".to_string(),
            ))
        }
    }
}

fn compile_mod_stage_1(
    allocator: &mut Allocator,
    args: NodePtr,
    macro_lookup: NodePtr,
    run_program: Rc<dyn TRunProgram>,
    produce_extra_info: bool,
) -> Result<CollectionResult, EvalErr> {
    // stage 1: collect up names of globals (functions, constants, macros)
    m! {
        let mut functions = HashMap::new();
        let mut constants = HashMap::new();
        let mut delayed_constants = HashMap::new();
        let mut macros = Vec::new();
        let mut namespace = HashSet::new();

        // eslint-disable-next-line no-constant-condition
        match proper_list(allocator, args, true) {
            None => { Err(EvalErr::InternalError(args, "miscompiled mod is not a proper list\n".to_string())) },
            Some(alist) => {
                if alist.is_empty() {
                    return Err(EvalErr::InternalError(args, "miscompiled mod is 0 size\n".to_string()));
                }

                let main_local_arguments = alist[0];

                for arg in alist.iter().take(alist.len()-1).skip(1) {
                    parse_mod_sexp(
                        allocator,
                        *arg,
                        &mut namespace,
                        &mut functions,
                        &mut constants,
                        &mut delayed_constants,
                        &mut macros,
                        run_program.clone()
                    )?;
                }

                // For each delayed constant, drain it into the
                // main constant pool.
                let mut result_collection = CollectionResult {
                    functions,
                    constants,
                    macros
                };

                let main_name_vec = MAIN_NAME.as_bytes().to_vec();

                // Process delayed constants until we either can't advance or
                // they're all done.

                loop {
                    if delayed_constants.is_empty() {
                        break;
                    }

                    let mut processed = false;
                    // copy so we can modify delayed_constants.
                    let delayed_constant_defs: Vec<(Vec<u8>, NodePtr)> =
                        delayed_constants.iter().map(|(k,v)| (k.clone(), *v)).collect();

                    for (name, delayed_body) in delayed_constant_defs.iter() {
                        let main_list =
                            enlist(
                                allocator,
                                &[NodePtr::NIL, *delayed_body]
                            )?;

                        result_collection.functions.insert(
                            main_name_vec.clone(), main_list
                        );

                        let used_in_this_constant = build_used_constants_names(
                            allocator,
                            &result_collection.functions,
                            &delayed_constants,
                            &result_collection.macros
                        )?;

                        let uses_other_constants = used_in_this_constant.iter().
                            any(|u| delayed_constants.contains_key(u));

                        if uses_other_constants {
                            continue;
                        }

                        processed = true;

                        let compiled =
                            finish_compile_from_collection(
                                allocator,
                                args,
                                macro_lookup,
                                run_program.clone(),
                                &result_collection,
                                produce_extra_info
                            )?;

                        let compilation_result =
                            run_program.run_program(
                                allocator,
                                compiled,
                                NodePtr::NIL,
                                None
                            )?;

                        let result =
                            run_program.run_program(
                                allocator,
                                compilation_result.1,
                                NodePtr::NIL,
                                None
                            )?;

                        delayed_constants.remove(name);
                        result_collection.constants.insert(
                            name.to_vec(), quote(allocator, result.1)?
                        );
                    }

                    if !processed {
                        return Err(EvalErr::InternalError(NodePtr::NIL, "got stuck untangling defconst dependencies".to_string()));
                    }
                }

                let uncompiled_main = nonempty_last(NodePtr::NIL, &alist)?;
                let main_list =
                    enlist(
                        allocator,
                        &[main_local_arguments, uncompiled_main]
                    )?;

                result_collection.functions.insert(
                    MAIN_NAME.as_bytes().to_vec(), main_list
                );

                Ok(result_collection)
            }
        }
    }
}

// export type TSymbolTable = Array<[SExp, Bytes]>;

fn symbol_table_for_tree(
    allocator: &mut Allocator,
    tree: NodePtr,
    root_node: &NodePath,
) -> Result<Vec<(NodePtr, Vec<u8>)>, EvalErr> {
    if !non_nil(allocator, tree) {
        return Ok(Vec::new());
    }

    match allocator.sexp(tree) {
        SExp::Atom => Ok(vec![(tree, root_node.as_path().data().to_vec())]),
        SExp::Pair(_, _) => {
            let left_bytes = NodePath::new(None).first();
            let right_bytes = NodePath::new(None).rest();

            let NodeSel::Cons(tree_first, tree_rest) =
                NodeSel::Cons(ThisNode::Here, ThisNode::Here).select_nodes(allocator, tree)?;

            // Allow haskell-like @ capture for destructuring.
            // If we encounter a form like (@ name substructure) then
            // treat it as though name captures the current path but
            // we also continue evaluating at the current position.
            let mut result_fin = Vec::new();
            if let Some((capture, destructure)) = is_at_capture(allocator, tree_first, tree_rest) {
                // Push the given name here.
                result_fin.push((capture, root_node.as_path().data().to_vec()));

                let mut substructure = symbol_table_for_tree(allocator, destructure, root_node)?;

                result_fin.append(&mut substructure);
            } else {
                let mut left =
                    symbol_table_for_tree(allocator, tree_first, &root_node.add(left_bytes))?;
                let mut right =
                    symbol_table_for_tree(allocator, tree_rest, &root_node.add(right_bytes))?;

                result_fin.append(&mut left);
                result_fin.append(&mut right);
            }

            Ok(result_fin)
        }
    }
}

fn build_macro_lookup_program(
    allocator: &mut Allocator,
    macro_lookup: NodePtr,
    macros: &[(Vec<u8>, NodePtr)],
    run_program: Rc<dyn TRunProgram>,
) -> Result<NodePtr, EvalErr> {
    m! {
        com_atom <- allocator.new_atom("com".as_bytes());
        cons_atom <- allocator.new_atom(&[4]);
        opt_atom <- allocator.new_atom("opt".as_bytes());

        let runner = || run_program.clone();
        macro_lookup_program <- quote(allocator, macro_lookup);
        result_program <- fold_m(
            allocator,
            &|allocator, macro_lookup_program, macro_def: &(Vec<u8>, NodePtr)| m! {
                cons_list <-
                    enlist(
                        allocator,
                        &[cons_atom, macro_def.1, macro_lookup_program]
                    );
                quoted_to_compile <- quote(allocator, cons_list);
                compile_form <-
                    enlist(
                        allocator,
                        &[com_atom, quoted_to_compile, macro_lookup_program]
                    );
                opt_form <- enlist(allocator, &[opt_atom, compile_form]);
                top_atom <- allocator.new_atom(NodePath::new(None).as_path().data());
                macro_evaluated <- evaluate(allocator, opt_form, top_atom);
                optimize_sexp(allocator, macro_evaluated, runner())
            },
            macro_lookup_program,
            &mut macros.iter()
        );
        Ok(result_program)
    }
}

#[allow(clippy::too_many_arguments)]
fn add_one_function(
    allocator: &mut Allocator,
    args_root_node: &NodePath,
    macro_lookup_program: NodePtr,
    constants_symbol_table: &[(NodePtr, Vec<u8>)],
    name: &[u8],
    lambda_expression: NodePtr,
    has_constants_tree: bool,
) -> Result<CompileOutput, EvalErr> {
    let mut compile: CompileOutput = Default::default();
    let com_atom = allocator.new_atom("com".as_bytes())?;
    let opt_atom = allocator.new_atom("opt".as_bytes())?;

    let function_args = first(allocator, lambda_expression)?;
    let local_symbol_table = symbol_table_for_tree(allocator, function_args, args_root_node)?;
    let mut all_symbols = local_symbol_table;
    all_symbols.append(&mut constants_symbol_table.to_owned());
    let lambda_form_content = rest(allocator, lambda_expression)?;
    let lambda_body = first(allocator, lambda_form_content)?;
    let quoted_lambda_expr = quote(allocator, lambda_body)?;
    let all_symbols_list_sexp = map_m(allocator, &mut all_symbols.iter(), &|allocator, pair| {
        let path_atom = allocator.new_atom(&pair.1)?;
        enlist(allocator, &[pair.0, path_atom])
    })?;

    let all_symbols_list = enlist(allocator, &all_symbols_list_sexp)?;

    let quoted_symbols = quote(allocator, all_symbols_list)?;
    let com_list = enlist(
        allocator,
        &[
            com_atom,
            quoted_lambda_expr,
            macro_lookup_program,
            quoted_symbols,
        ],
    )?;

    let opt_list = enlist(allocator, &[opt_atom, com_list])?;
    compile.functions.insert(name.to_vec(), opt_list);
    compile.symbols_extra_info.insert(
        name.to_vec(),
        FunctionExtraInfo {
            args: function_args,
            has_constants_tree,
        },
    );

    Ok(compile)
}

fn compile_functions(
    allocator: &mut Allocator,
    functions: &HashMap<Vec<u8>, NodePtr>,
    macro_lookup_program: NodePtr,
    constants_symbol_table: &[(NodePtr, Vec<u8>)],
    args_root_node: &NodePath,
    has_constants_tree: bool,
) -> Result<CompileOutput, EvalErr> {
    let mut compiled: CompileOutput = Default::default();

    for (name, exp) in functions.iter() {
        compiled.add_definitions(&add_one_function(
            allocator,
            args_root_node,
            macro_lookup_program,
            constants_symbol_table,
            name,
            *exp,
            has_constants_tree,
        )?);
    }

    Ok(compiled)
}

// Add an entry for main's arguments, named __chia__main_arguments in the
// symbols, to the symbol list, placing it at the front for simplicity.
fn add_main_args(
    allocator: &mut Allocator,
    args: NodePtr,
    symbols: NodePtr,
) -> Result<NodePtr, EvalErr> {
    let entry_name = allocator.new_atom("__chia__main_arguments".as_bytes())?;
    let entry_value_string = disassemble(allocator, args, None);
    let entry_value = allocator.new_atom(entry_value_string.as_bytes())?;
    let entry_cons = allocator.new_pair(entry_name, entry_value)?;
    allocator.new_pair(entry_cons, symbols)
}

fn finish_compile_from_collection(
    allocator: &mut Allocator,
    args: NodePtr,
    macro_lookup: NodePtr,
    run_program: Rc<dyn TRunProgram>,
    cr: &CollectionResult,
    produce_extra_info: bool,
) -> Result<NodePtr, EvalErr> {
    let a_atom = allocator.new_atom(&[2])?;
    let cons_atom = allocator.new_atom(&[4])?;
    let opt_atom = allocator.new_atom("opt".as_bytes())?;

    // move macros into the macro lookup
    let macro_lookup_program =
        build_macro_lookup_program(allocator, macro_lookup, &cr.macros, run_program.clone())?;

    // get a list of all symbols that are possibly used
    let all_constants_names =
        build_used_constants_names(allocator, &cr.functions, &cr.constants, &cr.macros)?;

    let has_constants_tree = !all_constants_names.is_empty();
    // build defuns table, with function names as keys

    let constants_tree = build_tree(allocator, &all_constants_names)?;

    let constants_root_node = NodePath::new(None).first();
    let args_root_node = if has_constants_tree {
        NodePath::new(None).rest()
    } else {
        NodePath::new(None)
    };

    let constants_symbol_table =
        symbol_table_for_tree(allocator, constants_tree, &constants_root_node)?;

    let compiled = compile_functions(
        allocator,
        &cr.functions,
        macro_lookup_program,
        &constants_symbol_table,
        &args_root_node,
        has_constants_tree,
    )?;

    let main_path = compiled.functions[MAIN_NAME.as_bytes()];

    if has_constants_tree {
        let mut all_constants_lookup = HashMap::new();
        for (k, v) in compiled.functions {
            if all_constants_names.contains(&k) {
                all_constants_lookup.insert(k, v);
            }
        }

        for (k, v) in cr.constants.iter() {
            all_constants_lookup.insert(k.to_vec(), *v);
        }

        let all_constants_list = all_constants_names
            .iter()
            .filter_map(|name| all_constants_lookup.get(name))
            .copied()
            .collect::<Vec<NodePtr>>();

        let all_constants_tree_program = build_tree_program(allocator, &all_constants_list)?;
        let top_atom = allocator.new_atom(NodePath::new(None).as_path().data())?;
        let arg_tree = enlist(
            allocator,
            &[cons_atom, all_constants_tree_program, top_atom],
        )?;

        let apply_list = enlist(allocator, &[a_atom, main_path, arg_tree])?;

        let quoted_apply_list = quote(allocator, apply_list)?;
        let opt_list = enlist(allocator, &[opt_atom, quoted_apply_list])?;
        let symbols_no_main = build_symbol_dump(
            allocator,
            &all_constants_lookup,
            &compiled.symbols_extra_info,
            run_program.clone(),
            produce_extra_info,
        )?;
        let first_of_args = first(allocator, args)?;
        let symbols = if produce_extra_info {
            add_main_args(allocator, first_of_args, symbols_no_main)?
        } else {
            symbols_no_main
        };

        let to_run = assemble(
            allocator,
            if produce_extra_info {
                "(_set_symbol_table (c (c (q . \"source_file\") (_get_source_file)) 1))"
            } else {
                "(_set_symbol_table 1)"
            },
        )?;

        run_program.run_program(allocator, to_run, symbols, None)?;

        Ok(opt_list)
    } else {
        let top_atom = allocator.new_atom(NodePath::new(None).as_path().data())?;
        let apply_list = enlist(allocator, &[a_atom, main_path, top_atom])?;
        let quoted_apply_list = quote(allocator, apply_list)?;
        enlist(allocator, &[opt_atom, quoted_apply_list])
    }
}

pub fn compile_mod(
    allocator: &mut Allocator,
    args: NodePtr,
    macro_lookup: NodePtr,
    _symbol_table: NodePtr,
    run_program: Rc<dyn TRunProgram>,
    _level: usize,
) -> Result<NodePtr, EvalErr> {
    // Deal with the "mod" keyword.
    let produce_extra_info_prog = assemble(allocator, "(_symbols_extra_info)")?;
    let produce_extra_info_null = NodePtr::NIL;
    let extra_info_res = run_program.run_program(
        allocator,
        produce_extra_info_prog,
        produce_extra_info_null,
        None,
    )?;
    let produce_extra_info = non_nil(allocator, extra_info_res.1);

    let cr = compile_mod_stage_1(
        allocator,
        args,
        macro_lookup,
        run_program.clone(),
        produce_extra_info,
    )?;
    finish_compile_from_collection(
        allocator,
        args,
        macro_lookup,
        run_program,
        &cr,
        produce_extra_info,
    )
}
