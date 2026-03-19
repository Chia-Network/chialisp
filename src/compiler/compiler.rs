use num_bigint::ToBigInt;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use clvm_rs::allocator::Allocator;
use indexmap::IndexMap;
use rue_diagnostic::Name;
use rue_hir::{
    BinaryOp, Database, FunctionCall, FunctionKind, FunctionSymbol, Hir, HirId, ParameterSymbol,
    Scope, ScopeId, Symbol, SymbolId, UnaryOp,
};
use rue_types::{Type, TypeId};

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

use crate::compiler::clvm::{sha256tree, NewStyleIntConversion};
use crate::compiler::codegen::{codegen, hoist_body_let_binding, process_helper_let_bindings};
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompilerOpts, HelperForm, PrimaryCodegen,
};
use crate::compiler::dialect::{AcceptedDialect, KNOWN_DIALECTS};
use crate::compiler::frontend::frontend;
use crate::compiler::optimize::get_optimizer;
use crate::compiler::prims;
use crate::compiler::sexp::{decode_string, parse_sexp, printable, SExp};
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

fn rue_err(loc: Srcloc, msg: impl Into<String>) -> CompileErr {
    CompileErr(loc, format!("rue translation: {}", msg.into()))
}

fn lookup_symbol_in_scope(db: &Database, scope: ScopeId, name: &str) -> Option<SymbolId> {
    let mut current = Some(scope);
    while let Some(scope_id) = current {
        let this_scope = db.scope(scope_id);
        if let Some(symbol) = this_scope.symbol(name) {
            return Some(symbol);
        }
        current = this_scope.parent();
    }
    None
}

fn intern_sexp_hir(db: &mut Database, s: &SExp) -> HirId {
    match s {
        SExp::Nil(_) => db.alloc_hir(Hir::Nil),
        SExp::Integer(_, i) => db.alloc_hir(Hir::Int(i.clone())),
        SExp::QuotedString(_, _, bytes) => db.alloc_hir(Hir::Bytes(bytes.clone())),
        SExp::Atom(_, bytes) => db.alloc_hir(Hir::Bytes(bytes.clone())),
        SExp::Cons(_, a, b) => {
            let first = intern_sexp_hir(db, a);
            let rest = intern_sexp_hir(db, b);
            db.alloc_hir(Hir::Pair(first, rest))
        }
    }
}

fn intern_expr_hir(db: &mut Database, scope: ScopeId, e: &BodyForm) -> Result<HirId, CompileErr> {
    match e {
        BodyForm::Quoted(s) => Ok(intern_sexp_hir(db, s)),
        BodyForm::Value(SExp::Nil(_)) => Ok(db.alloc_hir(Hir::Nil)),
        BodyForm::Value(SExp::Integer(_, i)) => Ok(db.alloc_hir(Hir::Int(i.clone()))),
        BodyForm::Value(SExp::QuotedString(_, _, bytes)) => {
            Ok(db.alloc_hir(Hir::Bytes(bytes.clone())))
        }
        BodyForm::Value(SExp::Atom(l, atom)) => {
            let symbol_name = decode_string(atom);
            if let Some(symbol) = lookup_symbol_in_scope(db, scope, &symbol_name) {
                let call_accessor = matches!(
                    db.symbol(symbol),
                    Symbol::Function(function)
                        if function.kind == FunctionKind::Inline && function.parameters.is_empty()
                );
                if call_accessor {
                    let function = db.alloc_hir(Hir::Reference(symbol));
                    Ok(db.alloc_hir(Hir::FunctionCall(FunctionCall {
                        function,
                        args: Vec::new(),
                        nil_terminated: true,
                    })))
                } else {
                    Ok(db.alloc_hir(Hir::Reference(symbol)))
                }
            } else {
                Err(rue_err(
                    l.clone(),
                    format!("unresolved symbol `{symbol_name}` in `{}`", e.to_sexp()),
                ))
            }
        }
        BodyForm::Value(v) => Ok(intern_sexp_hir(db, v)),
        BodyForm::Call(_, forms, tail) => {
            if forms.is_empty() {
                return Err(rue_err(e.loc(), "empty call expression"));
            }
            if tail.is_some() {
                return Err(rue_err(
                    e.loc(),
                    format!("rest-tail call form not yet supported: {}", e.to_sexp()),
                ));
            }

            let op_atom = if let BodyForm::Value(SExp::Atom(_, atom)) = &*forms[0] {
                Some(atom.as_slice())
            } else {
                None
            };
            if let Some(op_name) = op_atom {
                if (op_name == b"+" || op_name == b"*" || op_name == b"-") && forms.len() > 1 {
                    let mut args = forms.iter().skip(1);
                    let first = intern_expr_hir(db, scope, args.next().expect("has first"))?;
                    let folded = if op_name == b"-" && forms.len() == 2 {
                        db.alloc_hir(Hir::Unary(UnaryOp::Neg, first))
                    } else {
                        args.try_fold(first, |acc, arg| {
                            let next = intern_expr_hir(db, scope, arg)?;
                            let op = if op_name == b"+" {
                                BinaryOp::Add
                            } else if op_name == b"*" {
                                BinaryOp::Mul
                            } else {
                                BinaryOp::Sub
                            };
                            Ok::<HirId, CompileErr>(db.alloc_hir(Hir::Binary(op, acc, next)))
                        })?
                    };
                    return Ok(folded);
                }
                if op_name == b"f" && forms.len() == 2 {
                    let inner = intern_expr_hir(db, scope, &forms[1])?;
                    return Ok(db.alloc_hir(Hir::Unary(UnaryOp::First, inner)));
                }
                if op_name == b"r" && forms.len() == 2 {
                    let inner = intern_expr_hir(db, scope, &forms[1])?;
                    return Ok(db.alloc_hir(Hir::Unary(UnaryOp::Rest, inner)));
                }
                if op_name == b"c" && forms.len() == 3 {
                    let left = intern_expr_hir(db, scope, &forms[1])?;
                    let right = intern_expr_hir(db, scope, &forms[2])?;
                    return Ok(db.alloc_hir(Hir::Pair(left, right)));
                }
            }

            let function = intern_expr_hir(db, scope, &forms[0])?;
            let mut args = Vec::new();
            for arg in forms.iter().skip(1) {
                args.push(intern_expr_hir(db, scope, arg)?);
            }
            Ok(db.alloc_hir(Hir::FunctionCall(FunctionCall {
                function,
                args,
                nil_terminated: true,
            })))
        }
        BodyForm::Let(_, _) => Err(rue_err(
            e.loc(),
            format!(
                "let forms should be desugared before rue translation: {}",
                e.to_sexp()
            ),
        )),
        BodyForm::Mod(_, _) => Err(rue_err(
            e.loc(),
            format!("embedded mod expression not yet supported: {}", e.to_sexp()),
        )),
        BodyForm::Lambda(_) => Err(rue_err(
            e.loc(),
            format!("lambda expression not yet supported: {}", e.to_sexp()),
        )),
    }
}

impl From<Srcloc> for rue_diagnostic::SrcLoc {
    fn from(value: Srcloc) -> rue_diagnostic::SrcLoc {
        let start_line = value.line.max(1);
        let start_col = value.col.max(1);
        let (mut end_line, mut end_col) = value
            .until
            .as_ref()
            .map(|u| (u.line.max(1), u.col.max(1)))
            .unwrap_or((start_line, start_col.saturating_add(1)));
        // Rue diagnostics expect a forward, non-empty span.
        if (end_line, end_col) <= (start_line, start_col) {
            end_line = start_line;
            end_col = start_col.saturating_add(1);
        }

        let max_line = start_line.max(end_line);
        let mut line_widths = vec![0usize; max_line + 1];
        line_widths[start_line] = line_widths[start_line].max(start_col.saturating_sub(1));
        line_widths[end_line] = line_widths[end_line].max(end_col.saturating_sub(1));

        let mut text = String::new();
        let mut line_starts = vec![0usize; max_line + 1];
        for line in 1..=max_line {
            line_starts[line] = text.len();
            text.push_str(&" ".repeat(line_widths[line]));
            if line < max_line {
                text.push('\n');
            }
        }

        let start_offset = line_starts[start_line] + start_col.saturating_sub(1);
        let end_offset = line_starts[end_line] + end_col.saturating_sub(1);
        let source = rue_diagnostic::Source::new(
            text.into(),
            rue_diagnostic::SourceKind::File(value.file.as_ref().clone()),
        );
        rue_diagnostic::SrcLoc::new(source, start_offset..end_offset)
    }
}

#[allow(dead_code)]
fn param_names_and_paths_(vec: &mut Vec<(Number, Vec<u8>)>, env: Rc<SExp>, path: Number) {
    match env.borrow() {
        SExp::Atom(_, a) => {
            if printable(a, false) {
                vec.push((path, a.clone()));
            }
        }
        SExp::Cons(_, left, right) => {
            let two = 2_i32.to_bigint().unwrap();
            param_names_and_paths_(vec, left.clone(), path.clone() * two.clone());
            param_names_and_paths_(vec, right.clone(), path * two + bi_one());
        }
        _ => {}
    }
}

#[allow(dead_code)]
fn param_names_and_paths(env: Rc<SExp>) -> Vec<(Number, Vec<u8>)> {
    let mut raw = Vec::new();
    param_names_and_paths_(&mut raw, env, bi_one());
    raw
}

fn accessor_hir_for_path(db: &mut Database, args_symbol: SymbolId, path: &Number) -> HirId {
    let two = 2_i32.to_bigint().unwrap();
    let mut selectors = Vec::new();
    let mut cursor = path.clone();
    while cursor > bi_one() {
        selectors.push((cursor.clone() % two.clone()) != bi_zero());
        cursor /= two.clone();
    }

    let mut result = db.alloc_hir(Hir::Reference(args_symbol));
    for is_right in selectors.into_iter().rev() {
        result = if is_right {
            db.alloc_hir(Hir::Unary(UnaryOp::Rest, result))
        } else {
            db.alloc_hir(Hir::Unary(UnaryOp::First, result))
        };
    }
    result
}

#[allow(dead_code)]
fn create_param_helper(
    db: &mut Database,
    scope_id: ScopeId,
    any_type_id: TypeId,
    args_symbol: SymbolId,
    path: &Number,
    target: &[u8],
) -> (Vec<u8>, SymbolId) {
    let target_name = decode_string(target);
    let accessor_body = accessor_hir_for_path(db, args_symbol, path);
    let symbol_id = db.alloc_symbol(Symbol::Function(FunctionSymbol {
        name: Some(Name::new(target_name.clone(), None)),
        ty: any_type_id,
        scope: scope_id,
        vars: Default::default(),
        parameters: IndexMap::default(),
        nil_terminated: true,
        return_type: any_type_id,
        body: accessor_body,
        kind: FunctionKind::Inline,
    }));
    db.scope_mut(scope_id)
        .insert_symbol(target_name, symbol_id, false);
    (target.to_vec(), symbol_id)
}

fn install_tree_arg_accessors(
    db: &mut Database,
    scope_id: ScopeId,
    any_type_id: TypeId,
    args_spec: Rc<SExp>,
    args_symbol: SymbolId,
) {
    for (path, name) in param_names_and_paths(args_spec) {
        let _ = create_param_helper(db, scope_id, any_type_id, args_symbol, &path, &name);
    }
}

type PredeclaredHelperSymbols = HashMap<Vec<u8>, (SymbolId, ScopeId, bool)>;

fn predeclare_helper_symbols(
    db: &mut Database,
    main_scope: ScopeId,
    any_type_id: TypeId,
    helpers: &[HelperForm],
) -> Result<PredeclaredHelperSymbols, CompileErr> {
    let mut result = HashMap::new();

    for helper in helpers {
        let HelperForm::Defun(inline, data) = helper else {
            continue;
        };

        let function_scope = db.alloc_scope(Scope::new(Some(main_scope)));
        let unresolved_body = db.alloc_hir(Hir::Unresolved);
        let function_name = decode_string(helper.name());
        let function_sym = db.alloc_symbol(Symbol::Function(FunctionSymbol {
            name: Some(Name::new(
                function_name.clone(),
                Some(data.nl.clone().into()),
            )),
            ty: any_type_id,
            scope: function_scope,
            vars: Default::default(),
            nil_terminated: true,
            return_type: any_type_id,
            body: unresolved_body,
            parameters: IndexMap::default(),
            kind: if *inline {
                FunctionKind::Inline
            } else {
                FunctionKind::BinaryTree
            },
        }));
        db.scope_mut(main_scope)
            .insert_symbol(function_name, function_sym, false);

        result.insert(data.name.clone(), (function_sym, function_scope, *inline));
    }

    Ok(result)
}

fn intern_helper_hir(
    db: &mut Database,
    any_type_id: TypeId,
    h: &HelperForm,
    predeclared: &PredeclaredHelperSymbols,
) -> Result<HirId, CompileErr> {
    match h {
        HelperForm::Defun(_, data) => {
            let Some((function_sym, function_scope, is_inline)) = predeclared.get(h.name()) else {
                return Err(rue_err(
                    data.loc.clone(),
                    format!(
                        "missing predeclared symbol for helper `{}`",
                        decode_string(h.name())
                    ),
                ));
            };

            let mut plist: IndexMap<String, SymbolId> = IndexMap::default();
            if *is_inline {
                let params = data.args.proper_list().ok_or_else(|| {
                    rue_err(
                        data.args.loc(),
                        format!(
                            "inline defun args must be a simple list for phase 1: {}",
                            data.args
                        ),
                    )
                })?;
                for p in params {
                    let SExp::Atom(ploc, atom_name) = p else {
                        return Err(rue_err(
                            p.loc(),
                            format!(
                                "destructured inline defun args are not yet supported in phase 1: {}",
                                data.args
                            ),
                        ));
                    };
                    let param_name = decode_string(&atom_name);
                    let param_symbol = db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
                        name: Some(Name::new(param_name.clone(), Some(ploc.clone().into()))),
                        ty: any_type_id,
                    }));
                    db.scope_mut(*function_scope).insert_symbol(
                        param_name.clone(),
                        param_symbol,
                        false,
                    );
                    plist.insert(param_name, param_symbol);
                }
            } else {
                // Chialisp allows an arbitrary argument tree which specifies the exact environment
                // shape. Rue uses either sequential or tree shaped and choose the tree shape at
                // lowering time. The right approach is to use a single argument in BinaryTree
                // mode and make accessors for the individual destructurings chialisp would allow.
                let main_arg_symbol = db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
                    name: Some(Name::new("_$_args__", Some(data.args.loc().into()))),
                    ty: any_type_id,
                }));
                plist.insert("_$_args__".to_string(), main_arg_symbol);
                db.scope_mut(*function_scope).insert_symbol(
                    "_$_args__".to_string(),
                    main_arg_symbol,
                    false,
                );
                // Construct inline helper functions for each printable atom in the argument tree.
                install_tree_arg_accessors(
                    db,
                    *function_scope,
                    any_type_id,
                    data.args.clone(),
                    main_arg_symbol,
                );
            }

            let body_hir = intern_expr_hir(db, *function_scope, &data.body)?;
            *db.symbol_mut(*function_sym) = Symbol::Function(FunctionSymbol {
                name: Some(Name::new(
                    decode_string(h.name()),
                    Some(data.nl.clone().into()),
                )),
                ty: any_type_id,
                scope: *function_scope,
                vars: Default::default(),
                nil_terminated: true,
                return_type: any_type_id,
                body: body_hir,
                parameters: plist,
                kind: if *is_inline {
                    FunctionKind::Inline
                } else {
                    FunctionKind::BinaryTree
                },
            });
            Ok(body_hir)
        }
        _ => Err(rue_err(
            h.loc(),
            format!(
                "only defun helpers are currently translatable in phase 1: {}",
                h.to_sexp()
            ),
        )),
    }
}

fn intern_hir(
    db: &mut Database,
    any_type_id: TypeId,
    program: &CompileForm,
) -> Result<HirId, CompileErr> {
    let main_scope_id: ScopeId = db.alloc_scope(Scope::new(None));
    let predeclared = predeclare_helper_symbols(db, main_scope_id, any_type_id, &program.helpers)?;
    for h in program.helpers.iter() {
        intern_helper_hir(db, any_type_id, h, &predeclared)?;
    }

    // Program arguments are tree-shaped in chialisp, so model them the same way as
    // non-inline defun arguments: one binary-tree parameter and atom accessors.
    let main_args_symbol = db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
        name: Some(Name::new("_$_args__", Some(program.args.loc().into()))),
        ty: any_type_id,
    }));
    db.scope_mut(main_scope_id)
        .insert_symbol("_$_args__".to_string(), main_args_symbol, false);
    install_tree_arg_accessors(
        db,
        main_scope_id,
        any_type_id,
        program.args.clone(),
        main_args_symbol,
    );

    intern_expr_hir(db, main_scope_id, &program.exp)
}

fn rue_cg(opts: Rc<dyn CompilerOpts>) -> bool {
    matches!(opts.dialect().stepping, Some(1000000))
}

pub fn compile_from_compileform(
    context: &mut BasicCompileContext,
    opts: Rc<dyn CompilerOpts>,
    p2: CompileForm,
) -> Result<SExp, CompileErr> {
    let p3 = context.post_desugar_optimization(opts.clone(), p2)?;

    if rue_cg(opts.clone()) {
        let mut hir_db = Database::new();
        let any_type = Type::Any;
        let any_type_id = {
            let types_arena = hir_db.types_mut();
            types_arena.alloc(any_type)
        };
        let _hir_program = intern_hir(&mut hir_db, any_type_id, &p3)?;
        return Err(rue_err(
            p3.loc.clone(),
            "HIR->LIR->CLVM lowering is not implemented yet (phase 3)",
        ));
    }

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
        context: &mut BasicCompileContext,
        sexp: Rc<SExp>,
    ) -> Result<SExp, CompileErr> {
        let _int_conversion_bug = NewStyleIntConversion::new(self.dialect.int_fix);
        let me = Rc::new(self.clone());
        let runner = context.runner.clone();
        let mut context_wrapper = CompileContextWrapper::new(
            &mut context.allocator,
            runner,
            &mut context.symbols,
            get_optimizer(&sexp.loc(), me.clone())?,
        );
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
