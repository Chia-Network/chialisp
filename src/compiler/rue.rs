use num_bigint::ToBigInt;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use clvmr::Allocator;

use id_arena::Arena;
use indexmap::IndexMap;
use rue_diagnostic::Name;
use rue_hir::{
    Database, DependencyGraph, Environment, FunctionCall, FunctionKind, FunctionSymbol, Hir, HirId,
    Lowerer, ParameterSymbol, Scope, ScopeId, Symbol, SymbolId, UnaryOp, Value,
};
use rue_lir::{ClvmOp, Lir};
use rue_options::CompilerOptions as RueCompilerOptions;
use rue_types::{Type, TypeId};

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero};
use crate::compiler::clvm::convert_from_clvm_rs;
use crate::compiler::codegen::toposort_assign_bindings;
use crate::compiler::comptypes::{
    BindingPattern, BodyForm, CompileErr, CompileForm, CompilerOpts, HelperForm, LetFormKind,
};
use crate::compiler::gensym::gensym;
use crate::compiler::sexp::{decode_string, printable, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::util::Number;

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

fn binding_name(pattern: &BindingPattern) -> Option<Vec<u8>> {
    match pattern {
        BindingPattern::Name(name) => Some(name.clone()),
        _ => None,
    }
}

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

fn param_names_and_paths(env: Rc<SExp>) -> Vec<(Number, Vec<u8>)> {
    let mut raw = Vec::new();
    param_names_and_paths_(&mut raw, env, bi_one());
    raw
}

pub fn rue_cg(opts: Rc<dyn CompilerOpts>) -> bool {
    matches!(opts.dialect().stepping, Some(1000000))
        && opts.stdenv()
        && !opts.filename().starts_with("*macros*")
}

const STATIC_LIR_PRIMS: [ClvmOp; 48] = [
    ClvmOp::Quote,
    ClvmOp::Apply,
    ClvmOp::If,
    ClvmOp::Cons,
    ClvmOp::First,
    ClvmOp::Rest,
    ClvmOp::Listp,
    ClvmOp::Raise,
    ClvmOp::Eq,
    ClvmOp::GtBytes,
    ClvmOp::Sha256,
    ClvmOp::Substr,
    ClvmOp::Strlen,
    ClvmOp::Concat,
    ClvmOp::Add,
    ClvmOp::Sub,
    ClvmOp::Mul,
    ClvmOp::Div,
    ClvmOp::Divmod,
    ClvmOp::Gt,
    ClvmOp::Ash,
    ClvmOp::Lsh,
    ClvmOp::Logand,
    ClvmOp::Logior,
    ClvmOp::Logxor,
    ClvmOp::Lognot,
    ClvmOp::Not,
    ClvmOp::Any,
    ClvmOp::All,
    ClvmOp::Modpow,
    ClvmOp::Mod,
    ClvmOp::CoinId,
    ClvmOp::PubkeyForExp,
    ClvmOp::G1Add,
    ClvmOp::G1Subtract,
    ClvmOp::G1Multiply,
    ClvmOp::G1Negate,
    ClvmOp::G2Add,
    ClvmOp::G2Subtract,
    ClvmOp::G2Multiply,
    ClvmOp::G2Negate,
    ClvmOp::G1Map,
    ClvmOp::G2Map,
    ClvmOp::BlsPairingIdentity,
    ClvmOp::BlsVerify,
    ClvmOp::Secp256K1Verify,
    ClvmOp::Secp256R1Verify,
    ClvmOp::Keccak256,
];

fn match_prim(_opts: Rc<dyn CompilerOpts>, prim: &[u8]) -> Option<ClvmOp> {
    match prim {
        b"q" => return Some(ClvmOp::Quote),
        b"a" => return Some(ClvmOp::Apply),
        b"i" => return Some(ClvmOp::If),
        b"c" => return Some(ClvmOp::Cons),
        b"f" => return Some(ClvmOp::First),
        b"r" => return Some(ClvmOp::Rest),
        b"l" => return Some(ClvmOp::Listp),
        b"x" => return Some(ClvmOp::Raise),
        b"=" => return Some(ClvmOp::Eq),
        b">s" => return Some(ClvmOp::GtBytes),
        b"sha256" => return Some(ClvmOp::Sha256),
        b"substr" => return Some(ClvmOp::Substr),
        b"strlen" => return Some(ClvmOp::Strlen),
        b"concat" => return Some(ClvmOp::Concat),
        b"+" => return Some(ClvmOp::Add),
        b"-" => return Some(ClvmOp::Sub),
        b"*" => return Some(ClvmOp::Mul),
        b"/" => return Some(ClvmOp::Div),
        b"divmod" => return Some(ClvmOp::Divmod),
        b">" => return Some(ClvmOp::Gt),
        b"ash" => return Some(ClvmOp::Ash),
        b"lsh" => return Some(ClvmOp::Lsh),
        b"logand" => return Some(ClvmOp::Logand),
        b"logior" => return Some(ClvmOp::Logior),
        b"logxor" => return Some(ClvmOp::Logxor),
        b"lognot" => return Some(ClvmOp::Lognot),
        b"point_add" => return Some(ClvmOp::G1Add),
        b"pubkey_for_exp" => return Some(ClvmOp::PubkeyForExp),
        b"not" => return Some(ClvmOp::Not),
        b"any" => return Some(ClvmOp::Any),
        b"all" => return Some(ClvmOp::All),
        b"coinid" => return Some(ClvmOp::CoinId),
        b"g1_subtract" => return Some(ClvmOp::G1Subtract),
        b"g1_multiply" => return Some(ClvmOp::G1Multiply),
        b"g1_negate" => return Some(ClvmOp::G1Negate),
        b"g2_add" => return Some(ClvmOp::G2Add),
        b"g2_subtract" => return Some(ClvmOp::G2Subtract),
        b"g2_multiply" => return Some(ClvmOp::G2Multiply),
        b"g2_negate" => return Some(ClvmOp::G2Negate),
        b"g1_map" => return Some(ClvmOp::G1Map),
        b"g2_map" => return Some(ClvmOp::G2Map),
        b"bls_pairing_identity" => return Some(ClvmOp::BlsPairingIdentity),
        b"bls_verify" => return Some(ClvmOp::BlsVerify),
        b"modpow" => return Some(ClvmOp::Modpow),
        b"%" => return Some(ClvmOp::Mod),
        b"keccak256" => return Some(ClvmOp::Keccak256),
        b"secp256k1_verify" => return Some(ClvmOp::Secp256K1Verify),
        b"secp256r1_verify" => return Some(ClvmOp::Secp256R1Verify),
        _ => {}
    }

    for p in STATIC_LIR_PRIMS.iter() {
        if p.to_atom() == prim {
            return Some(*p);
        }
    }

    None
}

type PredeclaredHelperSymbols = HashMap<Vec<u8>, (SymbolId, ScopeId, bool)>;

struct RueConversion {
    db: Database,
    opts: Rc<dyn CompilerOpts>,
    text: Arc<str>,
    any_type_id: TypeId,
}

impl RueConversion {
    fn new(opts: Rc<dyn CompilerOpts>, text: Arc<str>) -> Self {
        let mut db = Database::new();
        let any_type = Type::Any;
        let any_type_id = {
            let types_arena = db.types_mut();
            types_arena.alloc(any_type)
        };
        RueConversion {
            db,
            opts: opts.clone(),
            text,
            any_type_id,
        }
    }

    fn intern_sexp_hir(&mut self, s: &SExp) -> HirId {
        match s {
            SExp::Nil(_) => self.db.alloc_hir(Hir::Nil),
            SExp::Integer(_, i) => self.db.alloc_hir(Hir::Int(i.clone())),
            SExp::QuotedString(_, _, bytes) => self.db.alloc_hir(Hir::Bytes(bytes.clone())),
            SExp::Atom(_, bytes) => self.db.alloc_hir(Hir::Bytes(bytes.clone())),
            SExp::Cons(_, a, b) => {
                let first = self.intern_sexp_hir(a);
                let rest = self.intern_sexp_hir(b);
                self.db.alloc_hir(Hir::Pair(first, rest))
            }
        }
    }

    fn primcall(
        &mut self,
        scope: ScopeId,
        clvm_op: ClvmOp,
        loc: &Srcloc,
        forms: &[Rc<BodyForm>],
    ) -> Result<HirId, CompileErr> {
        if matches!(clvm_op, ClvmOp::Cons) {
            if forms.len() < 3 {
                return Err(CompileErr(
                    loc.clone(),
                    "cons operator requires 2 arguments".to_string(),
                ));
            }

            let rest = self.intern_expr_hir(scope, &forms[2])?;
            let first = self.intern_expr_hir(scope, &forms[1])?;
            return Ok(self.db.alloc_hir(Hir::Pair(first, rest)));
        }

        let mut result = self.db.alloc_hir(Hir::Nil);
        for arg in forms.iter().skip(1).rev() {
            let arg_expr = self.intern_expr_hir(scope, arg)?;
            result = self.db.alloc_hir(Hir::Pair(arg_expr, result));
        }

        Ok(self.db.alloc_hir(Hir::ClvmOp(clvm_op, result)))
    }

    fn intern_expr_hir(&mut self, scope: ScopeId, e: &BodyForm) -> Result<HirId, CompileErr> {
        match e {
            BodyForm::Quoted(s) => Ok(self.intern_sexp_hir(s)),
            BodyForm::Value(SExp::Nil(_)) => Ok(self.db.alloc_hir(Hir::Nil)),
            BodyForm::Value(SExp::Integer(_, i)) => Ok(self.db.alloc_hir(Hir::Int(i.clone()))),
            BodyForm::Value(SExp::QuotedString(_, _, bytes)) => {
                Ok(self.db.alloc_hir(Hir::Bytes(bytes.clone())))
            }
            BodyForm::Value(SExp::Atom(l, atom)) => {
                let symbol_name = decode_string(atom);
                if let Some(symbol) = lookup_symbol_in_scope(&self.db, scope, &symbol_name) {
                    if let Symbol::Function(function) = self.db.symbol(symbol) {
                        if function.kind == FunctionKind::Inline
                            && function.parameters.is_empty()
                            && !matches!(self.db.hir(function.body), Hir::Unresolved)
                        {
                            return Ok(function.body);
                        }
                    }
                    Ok(self.db.alloc_hir(Hir::Reference(symbol)))
                } else {
                    Err(rue_err(
                        l.clone(),
                        format!("unresolved symbol `{symbol_name}` in `{}`", e.to_sexp()),
                    ))
                }
            }
            BodyForm::Value(v) => Ok(self.intern_sexp_hir(v)),
            BodyForm::Call(loc, forms, tail) => {
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
                    if op_name == b"f" && forms.len() == 2 {
                        let inner = self.intern_expr_hir(scope, &forms[1])?;
                        return Ok(self.db.alloc_hir(Hir::Unary(UnaryOp::First, inner)));
                    }
                    if op_name == b"r" && forms.len() == 2 {
                        let inner = self.intern_expr_hir(scope, &forms[1])?;
                        return Ok(self.db.alloc_hir(Hir::Unary(UnaryOp::Rest, inner)));
                    }
                    if op_name == b"c" && forms.len() == 3 {
                        let left = self.intern_expr_hir(scope, &forms[1])?;
                        let right = self.intern_expr_hir(scope, &forms[2])?;
                        return Ok(self.db.alloc_hir(Hir::Pair(left, right)));
                    }
                    if let Some(prim) = match_prim(self.opts.clone(), op_name) {
                        return self.primcall(scope, prim, loc, forms);
                    }
                }

                let function_result = self.intern_expr_hir(scope, &forms[0]);
                let function = match &function_result {
                    Ok(f) => f,
                    Err(_e) => {
                        if let Some(atom) = &op_atom {
                            if let Some(prim) = match_prim(self.opts.clone(), atom) {
                                return self.primcall(scope, prim, loc, forms);
                            }
                        }
                        return function_result;
                    }
                };

                let mut args = Vec::new();
                for arg in forms.iter().skip(1) {
                    args.push(self.intern_expr_hir(scope, arg)?);
                }
                Ok(self.db.alloc_hir(Hir::FunctionCall(FunctionCall {
                    function: *function,
                    args,
                    nil_terminated: true,
                })))
            }
            BodyForm::Let(let_kind, let_data) => {
                let mut statements = Vec::new();
                let mut body_scope = self.db.alloc_scope(Scope::new(Some(scope)));

                match let_kind {
                    LetFormKind::Parallel => {
                        for binding in let_data.bindings.iter() {
                            let Some(binding_name) = binding_name(&binding.pattern) else {
                                return Err(rue_err(
                                    binding.loc.clone(),
                                    format!(
                                        "complex let binding patterns are not yet supported: {}",
                                        e.to_sexp()
                                    ),
                                ));
                            };

                            let value_hir = self.intern_expr_hir(scope, &binding.body)?;
                            let symbol_name = decode_string(&binding_name);
                            let symbol =
                                self.db
                                    .alloc_symbol(Symbol::Binding(rue_hir::BindingSymbol {
                                        name: Some(Name::new(
                                            symbol_name.clone(),
                                            Some(self.to_rue_srcloc(binding.nl.clone())),
                                        )),
                                        value: Value::new(value_hir, self.any_type_id),
                                        inline: false,
                                    }));
                            self.db
                                .scope_mut(body_scope)
                                .insert_symbol(symbol_name, symbol, false);
                            statements.push(rue_hir::Statement::Let(symbol));
                        }
                    }
                    LetFormKind::Assign => {
                        body_scope = self.db.alloc_scope(Scope::new(Some(scope)));
                        let sorted_bindings =
                            toposort_assign_bindings(&let_data.loc, &let_data.bindings)?;
                        for b in sorted_bindings.iter() {
                            let binding = &let_data.bindings[b.index];
                            let value_hir = self.intern_expr_hir(body_scope, &binding.body)?;
                            let (multiple_binding, binding_env_pattern) = match &binding.pattern {
                                BindingPattern::Name(name) => (
                                    false,
                                    Rc::new(SExp::Atom(let_data.loc.clone(), name.clone())),
                                ),
                                BindingPattern::Complex(pattern) => (
                                    matches!(pattern.borrow(), SExp::Atom(_, _)),
                                    pattern.clone(),
                                ),
                            };

                            let new_name = if multiple_binding {
                                decode_string(&gensym(b"__assign_binding".to_vec()))
                            } else {
                                binding_env_pattern.to_string()
                            };

                            let symbol =
                                self.db
                                    .alloc_symbol(Symbol::Binding(rue_hir::BindingSymbol {
                                        name: Some(Name::new(
                                            new_name.clone(),
                                            Some(self.to_rue_srcloc(binding.nl.clone())),
                                        )),
                                        value: Value::new(value_hir, self.any_type_id),
                                        inline: false,
                                    }));

                            self.install_tree_arg_accessors(
                                body_scope,
                                binding_env_pattern.clone(),
                                symbol,
                            );

                            let next_scope = self.db.alloc_scope(Scope::new(Some(body_scope)));
                            self.db
                                .scope_mut(next_scope)
                                .insert_symbol(new_name, symbol, false);
                            body_scope = next_scope;
                            statements.push(rue_hir::Statement::Let(symbol));
                        }
                    }
                    LetFormKind::Sequential => {
                        body_scope = self.db.alloc_scope(Scope::new(Some(scope)));
                        for binding in let_data.bindings.iter() {
                            let Some(binding_name) = binding_name(&binding.pattern) else {
                                return Err(rue_err(
                                    binding.loc.clone(),
                                    format!(
                                        "complex let binding patterns are not yet supported: {}",
                                        e.to_sexp()
                                    ),
                                ));
                            };

                            let value_hir = self.intern_expr_hir(body_scope, &binding.body)?;
                            let symbol_name = decode_string(&binding_name);
                            let symbol =
                                self.db
                                    .alloc_symbol(Symbol::Binding(rue_hir::BindingSymbol {
                                        name: Some(Name::new(
                                            symbol_name.clone(),
                                            Some(self.to_rue_srcloc(binding.nl.clone())),
                                        )),
                                        value: Value::new(value_hir, self.any_type_id),
                                        inline: false,
                                    }));
                            let next_scope = self.db.alloc_scope(Scope::new(Some(body_scope)));
                            self.db
                                .scope_mut(next_scope)
                                .insert_symbol(symbol_name, symbol, false);
                            body_scope = next_scope;
                            statements.push(rue_hir::Statement::Let(symbol));
                        }
                    }
                }

                let body_hir = self.intern_expr_hir(body_scope, &let_data.body)?;
                Ok(self.db.alloc_hir(Hir::Block(rue_hir::Block {
                    statements,
                    body: Some(body_hir),
                })))
            }
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

    fn to_rue_srcloc(&self, value: Srcloc) -> rue_diagnostic::SrcLoc {
        // XXX Fix this.
        let start_offset = 0;
        let end_offset = 1;

        let source = rue_diagnostic::Source::new(
            self.text.clone(),
            rue_diagnostic::SourceKind::File(value.file.as_ref().clone()),
        );
        rue_diagnostic::SrcLoc::new(source, start_offset..end_offset)
    }

    fn accessor_hir_for_path(&mut self, args_symbol: SymbolId, path: &Number) -> HirId {
        let two = 2_i32.to_bigint().unwrap();
        let mut selectors = Vec::new();
        let mut cursor = path.clone();
        while cursor > bi_one() {
            selectors.push((cursor.clone() % two.clone()) != bi_zero());
            cursor /= two.clone();
        }

        let mut result = self.db.alloc_hir(Hir::Reference(args_symbol));
        for is_right in selectors.into_iter().rev() {
            result = if is_right {
                self.db.alloc_hir(Hir::Unary(UnaryOp::Rest, result))
            } else {
                self.db.alloc_hir(Hir::Unary(UnaryOp::First, result))
            };
        }
        result
    }

    fn create_param_helper(
        &mut self,
        scope_id: ScopeId,
        args_symbol: SymbolId,
        path: &Number,
        target: &[u8],
    ) -> (Vec<u8>, SymbolId) {
        let target_name = decode_string(target);
        let accessor_body = self.accessor_hir_for_path(args_symbol, path);
        let symbol_id = self.db.alloc_symbol(Symbol::Function(FunctionSymbol {
            name: Some(Name::new(target_name.clone(), None)),
            ty: self.any_type_id,
            scope: scope_id,
            vars: Default::default(),
            parameters: IndexMap::default(),
            nil_terminated: true,
            return_type: self.any_type_id,
            body: accessor_body,
            kind: FunctionKind::Inline,
        }));
        self.db
            .scope_mut(scope_id)
            .insert_symbol(target_name, symbol_id, false);
        (target.to_vec(), symbol_id)
    }

    fn create_inline_value_helper(
        &mut self,
        scope_id: ScopeId,
        name: &str,
        body: HirId,
    ) -> SymbolId {
        let symbol_id = self.db.alloc_symbol(Symbol::Function(FunctionSymbol {
            name: Some(Name::new(name.to_string(), None)),
            ty: self.any_type_id,
            scope: scope_id,
            vars: Default::default(),
            parameters: IndexMap::default(),
            nil_terminated: true,
            return_type: self.any_type_id,
            body,
            kind: FunctionKind::Inline,
        }));
        self.db
            .scope_mut(scope_id)
            .insert_symbol(name.to_string(), symbol_id, false);
        symbol_id
    }

    fn install_tree_arg_accessors(
        &mut self,
        scope_id: ScopeId,
        args_spec: Rc<SExp>,
        args_symbol: SymbolId,
    ) {
        for (path, name) in param_names_and_paths(args_spec) {
            let _ = self.create_param_helper(scope_id, args_symbol, &path, &name);
        }
    }

    fn install_env_alias_helpers(&mut self, scope_id: ScopeId, env_symbol: SymbolId) {
        // In classic codegen, @*env* is used as the current environment path (1),
        // and (r @*env*) addresses the user argument subtree. Model that directly.
        let env_ref = self.db.alloc_hir(Hir::Reference(env_symbol));
        let env_nil = self.db.alloc_hir(Hir::Nil);
        let env_value = self.db.alloc_hir(Hir::Pair(env_nil, env_ref));
        let _ = self.create_inline_value_helper(scope_id, "@*env*", env_value);
        let _ = self.create_inline_value_helper(scope_id, "@", env_value);
    }

    fn predeclare_helper_symbols(
        &mut self,
        main_scope: ScopeId,
        helpers: &[HelperForm],
    ) -> Result<PredeclaredHelperSymbols, CompileErr> {
        let mut result = HashMap::new();

        for helper in helpers {
            let HelperForm::Defun(inline, data) = helper else {
                continue;
            };

            let function_scope = self.db.alloc_scope(Scope::new(Some(main_scope)));
            let unresolved_body = self.db.alloc_hir(Hir::Unresolved);
            let function_name = decode_string(helper.name());
            let function_sym = self.db.alloc_symbol(Symbol::Function(FunctionSymbol {
                name: Some(Name::new(
                    function_name.clone(),
                    Some(self.to_rue_srcloc(data.nl.clone())),
                )),
                ty: self.any_type_id,
                scope: function_scope,
                vars: Default::default(),
                nil_terminated: *inline,
                return_type: self.any_type_id,
                body: unresolved_body,
                parameters: IndexMap::default(),
                kind: if *inline {
                    FunctionKind::Inline
                } else {
                    FunctionKind::BinaryTree
                },
            }));
            self.db
                .scope_mut(main_scope)
                .insert_symbol(function_name, function_sym, false);

            result.insert(data.name.clone(), (function_sym, function_scope, *inline));
        }

        Ok(result)
    }

    fn intern_helper_hir(
        &mut self,
        h: &HelperForm,
        predeclared: &PredeclaredHelperSymbols,
    ) -> Result<HirId, CompileErr> {
        match h {
            HelperForm::Defun(_, data) => {
                let Some((function_sym, function_scope, is_inline)) = predeclared.get(h.name())
                else {
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
                    for (arg_index, p) in params.into_iter().enumerate() {
                        if let SExp::Atom(ploc, atom_name) = p {
                            let param_name = decode_string(&atom_name);
                            let param_symbol =
                                self.db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
                                    name: Some(Name::new(
                                        param_name.clone(),
                                        Some(self.to_rue_srcloc(ploc.clone())),
                                    )),
                                    ty: self.any_type_id,
                                }));
                            self.db.scope_mut(*function_scope).insert_symbol(
                                param_name.clone(),
                                param_symbol,
                                false,
                            );
                            plist.insert(param_name, param_symbol);
                        } else {
                            // Inline helpers may still destructure argument trees.
                            // Bind the incoming argument, then create accessor helpers for leaves.
                            let param_name = format!("_$_arg_{arg_index}");
                            let param_symbol =
                                self.db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
                                    name: Some(Name::new(
                                        param_name.clone(),
                                        Some(self.to_rue_srcloc(p.loc())),
                                    )),
                                    ty: self.any_type_id,
                                }));
                            self.db.scope_mut(*function_scope).insert_symbol(
                                param_name.clone(),
                                param_symbol,
                                false,
                            );
                            plist.insert(param_name, param_symbol);
                            self.install_tree_arg_accessors(
                                *function_scope,
                                Rc::new(p.clone()),
                                param_symbol,
                            );
                        }
                    }
                } else {
                    // Chialisp allows an arbitrary argument tree which specifies the exact environment
                    // shape. Rue uses either sequential or tree shaped and choose the tree shape at
                    // lowering time. The right approach is to use a single argument in BinaryTree
                    // mode and make accessors for the individual destructurings chialisp would allow.
                    let main_arg_symbol =
                        self.db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
                            name: Some(Name::new(
                                "_$_args__",
                                Some(self.to_rue_srcloc(data.args.loc())),
                            )),
                            ty: self.any_type_id,
                        }));
                    plist.insert("_$_args__".to_string(), main_arg_symbol);
                    self.db.scope_mut(*function_scope).insert_symbol(
                        "_$_args__".to_string(),
                        main_arg_symbol,
                        false,
                    );
                    // Construct inline helper functions for each printable atom in the argument tree.
                    self.install_tree_arg_accessors(
                        *function_scope,
                        data.args.clone(),
                        main_arg_symbol,
                    );
                }

                let body_hir = self.intern_expr_hir(*function_scope, &data.body)?;
                *self.db.symbol_mut(*function_sym) = Symbol::Function(FunctionSymbol {
                    name: Some(Name::new(
                        decode_string(h.name()),
                        Some(self.to_rue_srcloc(data.nl.clone())),
                    )),
                    ty: self.any_type_id,
                    scope: *function_scope,
                    vars: Default::default(),
                    nil_terminated: *is_inline,
                    return_type: self.any_type_id,
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

    fn intern_hir(&mut self, program: &CompileForm) -> Result<SymbolId, CompileErr> {
        let main_scope_id: ScopeId = self.db.alloc_scope(Scope::new(None));
        let predeclared = self.predeclare_helper_symbols(main_scope_id, &program.helpers)?;
        for h in program.helpers.iter() {
            self.intern_helper_hir(h, &predeclared)?;
        }

        let program_scope = self.db.alloc_scope(Scope::new(Some(main_scope_id)));
        // Program arguments are tree-shaped in chialisp, so model them the same way as
        // non-inline defun arguments: one binary-tree parameter and atom accessors.
        let main_args_symbol = self.db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
            name: Some(Name::new(
                "_$_args__",
                Some(self.to_rue_srcloc(program.args.loc())),
            )),
            ty: self.any_type_id,
        }));
        self.db.scope_mut(program_scope).insert_symbol(
            "_$_args__".to_string(),
            main_args_symbol,
            false,
        );
        self.install_tree_arg_accessors(program_scope, program.args.clone(), main_args_symbol);
        self.install_env_alias_helpers(program_scope, main_args_symbol);

        let main_body = self.intern_expr_hir(program_scope, &program.exp)?;
        let mut main_params: IndexMap<String, SymbolId> = IndexMap::default();
        main_params.insert("_$_args__".to_string(), main_args_symbol);
        let main_symbol = self.db.alloc_symbol(Symbol::Function(FunctionSymbol {
            name: Some(Name::new(
                "__chia_main__",
                Some(self.to_rue_srcloc(program.loc.clone())),
            )),
            ty: self.any_type_id,
            scope: program_scope,
            vars: Default::default(),
            parameters: main_params,
            nil_terminated: false,
            return_type: self.any_type_id,
            body: main_body,
            kind: FunctionKind::BinaryTree,
        }));
        self.db.scope_mut(main_scope_id).insert_symbol(
            "__chia_main__".to_string(),
            main_symbol,
            false,
        );
        Ok(main_symbol)
    }

    fn transform(
        &mut self,
        opts: Rc<dyn CompilerOpts>,
        program: &CompileForm,
    ) -> Result<SExp, CompileErr> {
        let mut allocator = Allocator::new();
        let main_symbol = self.intern_hir(program)?;
        self.verify_no_unresolved_hir(main_symbol).map_err(|e| {
            rue_err(
                program.loc.clone(),
                format!("unresolved hir after translation: {e}"),
            )
        })?;

        let rue_options = RueCompilerOptions {
            optimize_lir: opts.optimize(),
            ..RueCompilerOptions::default()
        };
        let graph = DependencyGraph::build(&self.db, main_symbol, rue_options);
        let mut lir_arena: Arena<Lir> = Arena::new();
        let base_path = Path::new(&opts.filename())
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let mut lowerer = Lowerer::new(
            &mut self.db,
            &mut lir_arena,
            &graph,
            rue_options,
            main_symbol,
            base_path,
        );
        let mut lir = lowerer.lower_symbol_value(&Environment::default(), main_symbol);
        if rue_options.optimize_lir {
            lir = rue_lir::optimize(&mut lir_arena, lir);
        }
        let node = rue_lir::codegen(&lir_arena, &mut allocator, lir).map_err(|e| {
            rue_err(
                program.loc.clone(),
                format!("rue codegen failed while generating clvm: {e}"),
            )
        })?;
        let output =
            convert_from_clvm_rs(&mut allocator, program.loc.clone(), node).map_err(|e| {
                rue_err(
                    program.loc.clone(),
                    format!("failed to convert rue output back to sexp: {e}"),
                )
            })?;
        Ok(output.as_ref().clone())
    }

    fn visit_symbol(
        &self,
        symbol: SymbolId,
        visited_symbols: &mut HashSet<SymbolId>,
        visited_hir: &mut HashSet<HirId>,
    ) -> Result<(), String> {
        if !visited_symbols.insert(symbol) {
            return Ok(());
        }
        if let Symbol::Function(function) = self.db.symbol(symbol) {
            self.visit_hir(function.body, visited_symbols, visited_hir)
                .map_err(|ctx| {
                    format!(
                        "unresolved hir in function `{}` ({ctx})",
                        function
                            .name
                            .as_ref()
                            .map(|n| n.text().to_string())
                            .unwrap_or_else(|| "<unnamed>".to_string())
                    )
                })?;
        }
        Ok(())
    }

    fn visit_hir(
        &self,
        hir: HirId,
        visited_symbols: &mut HashSet<SymbolId>,
        visited_hir: &mut HashSet<HirId>,
    ) -> Result<(), String> {
        if !visited_hir.insert(hir) {
            return Ok(());
        }

        match self.db.hir(hir).clone() {
            Hir::Unresolved => Err(format!("hir id {}", hir.index())),
            Hir::Reference(symbol) => self.visit_symbol(symbol, visited_symbols, visited_hir),
            Hir::Pair(a, b) | Hir::Binary(_, a, b) => {
                self.visit_hir(a, visited_symbols, visited_hir)?;
                self.visit_hir(b, visited_symbols, visited_hir)
            }
            Hir::Unary(_, inner) => self.visit_hir(inner, visited_symbols, visited_hir),
            Hir::FunctionCall(call) => {
                self.visit_hir(call.function, visited_symbols, visited_hir)?;
                for arg in call.args {
                    self.visit_hir(arg, visited_symbols, visited_hir)?;
                }
                Ok(())
            }
            Hir::Block(block) => {
                for stmt in block.statements {
                    match stmt {
                        rue_hir::Statement::Expr(expr) => {
                            self.visit_hir(expr.hir, visited_symbols, visited_hir)?
                        }
                        rue_hir::Statement::If(if_stmt) => {
                            self.visit_hir(if_stmt.condition, visited_symbols, visited_hir)?;
                            self.visit_hir(if_stmt.then, visited_symbols, visited_hir)?;
                        }
                        rue_hir::Statement::Return(h) => {
                            self.visit_hir(h, visited_symbols, visited_hir)?
                        }
                        rue_hir::Statement::Assert(h, _)
                        | rue_hir::Statement::Debug(h, _)
                        | rue_hir::Statement::Raise(Some(h), _) => {
                            self.visit_hir(h, visited_symbols, visited_hir)?
                        }
                        rue_hir::Statement::Raise(None, _) | rue_hir::Statement::Let(_) => {}
                    }
                }
                if let Some(body) = block.body {
                    self.visit_hir(body, visited_symbols, visited_hir)?;
                }
                Ok(())
            }
            Hir::If(a, b, c, _) | Hir::CoinId(a, b, c) | Hir::Modpow(a, b, c) => {
                self.visit_hir(a, visited_symbols, visited_hir)?;
                self.visit_hir(b, visited_symbols, visited_hir)?;
                self.visit_hir(c, visited_symbols, visited_hir)
            }
            Hir::Substr(a, b, c) => {
                self.visit_hir(a, visited_symbols, visited_hir)?;
                self.visit_hir(b, visited_symbols, visited_hir)?;
                if let Some(c) = c {
                    self.visit_hir(c, visited_symbols, visited_hir)?;
                }
                Ok(())
            }
            Hir::G1Map(a, b) | Hir::G2Map(a, b) => {
                self.visit_hir(a, visited_symbols, visited_hir)?;
                if let Some(b) = b {
                    self.visit_hir(b, visited_symbols, visited_hir)?;
                }
                Ok(())
            }
            Hir::BlsPairingIdentity(args) | Hir::BlsVerify(_, args) => {
                for arg in args {
                    self.visit_hir(arg, visited_symbols, visited_hir)?;
                }
                Ok(())
            }
            Hir::Secp256K1Verify(a, b, c) | Hir::Secp256R1Verify(a, b, c) => {
                self.visit_hir(a, visited_symbols, visited_hir)?;
                self.visit_hir(b, visited_symbols, visited_hir)?;
                self.visit_hir(c, visited_symbols, visited_hir)
            }
            Hir::ClvmOp(_, args) => self.visit_hir(args, visited_symbols, visited_hir),
            Hir::Lambda(symbol) => self.visit_symbol(symbol, visited_symbols, visited_hir),
            Hir::String(_)
            | Hir::Nil
            | Hir::Int(_)
            | Hir::Bytes(_)
            | Hir::Bool(_)
            | Hir::InfinityG1
            | Hir::InfinityG2 => Ok(()),
        }
    }

    fn verify_no_unresolved_hir(&self, root: SymbolId) -> Result<(), String> {
        let mut visited_symbols = HashSet::new();
        let mut visited_hir = HashSet::new();
        self.visit_symbol(root, &mut visited_symbols, &mut visited_hir)
    }
}

pub fn compile_with_rue_codegen(
    opts: Rc<dyn CompilerOpts>,
    text: Arc<str>,
    program: &CompileForm,
) -> Result<SExp, CompileErr> {
    let mut rue_compiler = RueConversion::new(opts.clone(), text);
    rue_compiler.transform(opts.clone(), program)
}
