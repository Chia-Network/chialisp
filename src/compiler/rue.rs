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
use crate::classic::clvm_tools::stages::stage_0::{DefaultProgramRunner, TRunProgram};
use crate::compiler::clvm::{convert_from_clvm_rs, convert_to_clvm_rs};
use crate::compiler::codegen::toposort_assign_bindings;
use crate::compiler::compiler::is_at_capture;
use crate::compiler::comptypes::{
    BindingPattern, BodyForm, CompileErr, CompileForm, CompilerOpts, DefconstData, DefunData,
    HelperForm, LetFormKind,
};
use crate::compiler::gensym::gensym;
use crate::compiler::optimize::depgraph::{DepgraphOptions, FunctionDependencyGraph};
use crate::compiler::sexp::{decode_string, enlist, printable, SExp};
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
            // (@ capture subtree) captures the whole current subtree at this path,
            // then destructures `subtree` against the same value.
            if let Some((capture, substructure)) = is_at_capture(left.clone(), right.clone()) {
                if printable(&capture, false) {
                    vec.push((path.clone(), capture));
                }
                param_names_and_paths_(vec, substructure, path);
                return;
            }

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

// Return a value if the environment is a list with a non-nil tail.
fn improper_list(env: Rc<SExp>) -> Option<(Vec<SExp>, Rc<SExp>)> {
    let mut prefix = Vec::new();
    let mut cursor = env;
    loop {
        match cursor.borrow() {
            SExp::Cons(_, head, tail) => {
                prefix.push(head.as_ref().clone());
                cursor = tail.clone();
            }
            SExp::Nil(_) => return None,
            _ => return Some((prefix, cursor.clone())),
        }
    }
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

fn body_cons(loc: Srcloc, left: Rc<BodyForm>, right: Rc<BodyForm>) -> BodyForm {
    BodyForm::Call(
        loc.clone(),
        vec![
            Rc::new(BodyForm::Value(SExp::Atom(loc, vec![4]))),
            left,
            right,
        ],
        None,
    )
}

fn body_list(loc: Srcloc, forms: &[Rc<BodyForm>]) -> BodyForm {
    let mut result = BodyForm::Quoted(SExp::Nil(loc.clone()));
    for b in forms.iter().rev() {
        result = body_cons(loc.clone(), b.clone(), Rc::new(result));
    }
    result
}

fn skip_captures_for_lambda(args: Rc<SExp>) -> Rc<SExp> {
    if let SExp::Cons(l, a, b) = &*args {
        if let SExp::Atom(_la, name) = &**a {
            if name == b"&" {
                return b.clone();
            }
        }
        let new_a = skip_captures_for_lambda(a.clone());
        let new_b = skip_captures_for_lambda(b.clone());
        Rc::new(SExp::Cons(l.clone(), new_a, new_b))
    } else {
        args
    }
}

#[allow(dead_code)]
#[derive(Clone)]
enum PredeclaredSymbolKind {
    Constant,
    Defun,
    InlineDefunNormalArgs(Vec<SExp>),
    InlineDefunImproperListArgs(Vec<SExp>, Rc<SExp>),
    InlineDefunTreeArgs,
}

#[derive(Clone)]
struct PredeclaredHelperSymbol {
    symbol_id: SymbolId,
    scope_id: ScopeId,
    kind: PredeclaredSymbolKind,
}
type PredeclaredHelperSymbols = HashMap<Vec<u8>, PredeclaredHelperSymbol>;

struct RueConversion {
    db: Database,
    opts: Rc<dyn CompilerOpts>,
    text: Arc<str>,
    any_type_id: TypeId,
    predeclared_helpers: PredeclaredHelperSymbols,
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
            predeclared_helpers: PredeclaredHelperSymbols::default(),
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

    fn predeclared_kind_for_function_hir(&self, function: HirId) -> Option<PredeclaredSymbolKind> {
        let Hir::Reference(target_symbol) = self.db.hir(function) else {
            return None;
        };
        self.predeclared_helpers
            .values()
            .find_map(|helper| (helper.symbol_id == *target_symbol).then_some(helper.kind.clone()))
    }

    fn apply_rest_n(&mut self, value: HirId, count: usize) -> HirId {
        let mut result = value;
        for _ in 0..count {
            result = self.db.alloc_hir(Hir::Unary(UnaryOp::Rest, result));
        }
        result
    }

    fn choose_inline_argument(
        &mut self,
        callsite: &Srcloc,
        positional_args: &[HirId],
        tail: Option<HirId>,
        index: usize,
    ) -> Result<HirId, CompileErr> {
        if let Some(arg) = positional_args.get(index) {
            return Ok(*arg);
        }

        let Some(tail_expr) = tail else {
            return Err(CompileErr(
                callsite.clone(),
                format!("Lookup for argument {} that wasn't passed", index + 1),
            ));
        };

        let tail_element = self.apply_rest_n(tail_expr, index - positional_args.len());
        Ok(self.db.alloc_hir(Hir::Unary(UnaryOp::First, tail_element)))
    }

    fn normalize_inline_normal_call_args(
        &mut self,
        callsite: &Srcloc,
        fixed_args: &[SExp],
        positional_args: &[HirId],
        tail: Option<HirId>,
    ) -> Result<Vec<HirId>, CompileErr> {
        let mut rewritten_args = Vec::with_capacity(fixed_args.len());
        for index in 0..fixed_args.len() {
            rewritten_args.push(self.choose_inline_argument(
                callsite,
                positional_args,
                tail,
                index,
            )?);
        }
        Ok(rewritten_args)
    }

    fn normalize_inline_improper_call_args(
        &mut self,
        callsite: &Srcloc,
        fixed_count: usize,
        positional_args: &[HirId],
        tail: Option<HirId>,
    ) -> Result<Vec<HirId>, CompileErr> {
        let mut rewritten_args = Vec::with_capacity(fixed_count + 1);
        for index in 0..fixed_count {
            rewritten_args.push(self.choose_inline_argument(
                callsite,
                positional_args,
                tail,
                index,
            )?);
        }

        let consumed_tail_for_fixed = fixed_count.saturating_sub(positional_args.len());
        let mut final_tail_arg = if let Some(tail_expr) = tail {
            self.apply_rest_n(tail_expr, consumed_tail_for_fixed)
        } else {
            self.db.alloc_hir(Hir::Nil)
        };

        for arg in positional_args
            .iter()
            .skip(fixed_count.min(positional_args.len()))
            .rev()
        {
            final_tail_arg = self.db.alloc_hir(Hir::Pair(*arg, final_tail_arg));
        }

        rewritten_args.push(final_tail_arg);
        Ok(rewritten_args)
    }

    fn normalize_inline_tree_call_args(
        &mut self,
        positional_args: &[HirId],
        tail: Option<HirId>,
    ) -> Vec<HirId> {
        let mut packed_args = tail.unwrap_or_else(|| self.db.alloc_hir(Hir::Nil));
        for arg in positional_args.iter().rev() {
            packed_args = self.db.alloc_hir(Hir::Pair(*arg, packed_args));
        }
        vec![packed_args]
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
                    if op_name == b"com" && forms.len() == 2 {
                        // com is special.  It wraps code which will be callable.
                        let com_symbol_name = decode_string(&gensym(b"_$_com_".to_vec()));
                        let new_scope = self.db.alloc_scope(Scope::new(Some(scope)));
                        let value_hir = self.intern_expr_hir(new_scope, &forms[1])?;
                        let com_symbol = self.db.alloc_symbol(Symbol::Function(FunctionSymbol {
                            name: Some(Name::new(com_symbol_name.clone(), None)),
                            ty: self.any_type_id,
                            scope,
                            vars: Default::default(),
                            parameters: IndexMap::default(),
                            nil_terminated: true,
                            return_type: self.any_type_id,
                            body: value_hir,
                            kind: FunctionKind::BinaryTree,
                        }));
                        self.db.scope_mut(scope).insert_symbol(
                            com_symbol_name.clone(),
                            com_symbol,
                            false,
                        );
                        return Ok(self.db.alloc_hir(Hir::Reference(com_symbol)));
                    }
                    if op_name == b"softfork" && forms.len() == 5 {
                        // Softfork is special here since it isn't an opcode rue lets us produce
                        // directly.
                        let softfork_args = Rc::new(body_list(loc.clone(), &forms[1..]));
                        return self.intern_expr_hir(
                            scope,
                            &BodyForm::Call(
                                loc.clone(),
                                vec![
                                    Rc::new(BodyForm::Value(SExp::Atom(loc.clone(), vec![2]))),
                                    Rc::new(BodyForm::Quoted(enlist(
                                        loc.clone(),
                                        &[
                                            Rc::new(SExp::Atom(loc.clone(), vec![36])),
                                            Rc::new(SExp::Atom(loc.clone(), vec![2])),
                                            Rc::new(SExp::Atom(loc.clone(), vec![5])),
                                            Rc::new(SExp::Atom(loc.clone(), vec![11])),
                                            Rc::new(SExp::Atom(loc.clone(), vec![23])),
                                        ],
                                    ))),
                                    softfork_args,
                                ],
                                None,
                            ),
                        );
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
                let tail_arg = if let Some(tail_expr) = tail.as_ref() {
                    Some(self.intern_expr_hir(scope, tail_expr)?)
                } else {
                    None
                };

                let mut nil_terminated = tail_arg.is_none();
                let mut rewritten_inline = false;
                if let Some(predeclared_kind) = self.predeclared_kind_for_function_hir(*function) {
                    match predeclared_kind {
                        PredeclaredSymbolKind::InlineDefunNormalArgs(fixed_args) => {
                            args = self.normalize_inline_normal_call_args(
                                loc,
                                &fixed_args,
                                &args,
                                tail_arg,
                            )?;
                            rewritten_inline = true;
                            nil_terminated = true;
                        }
                        PredeclaredSymbolKind::InlineDefunImproperListArgs(prefix, _tail) => {
                            args = self.normalize_inline_improper_call_args(
                                loc,
                                prefix.len(),
                                &args,
                                tail_arg,
                            )?;
                            rewritten_inline = true;
                            nil_terminated = true;
                        }
                        PredeclaredSymbolKind::InlineDefunTreeArgs => {
                            args = self.normalize_inline_tree_call_args(&args, tail_arg);
                            rewritten_inline = true;
                            nil_terminated = true;
                        }
                        _ => {}
                    }
                }

                if !rewritten_inline {
                    if let Some(t) = tail_arg {
                        args.push(t);
                        nil_terminated = false;
                    }
                }
                Ok(self.db.alloc_hir(Hir::FunctionCall(FunctionCall {
                    function: *function,
                    args,
                    nil_terminated,
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
            BodyForm::Mod(_, program) => {
                let generated_code =
                    compile_with_rue_codegen(self.opts.clone(), Arc::from(""), program)?;
                Ok(self.intern_sexp_hir(&generated_code))
            }
            BodyForm::Lambda(data) => {
                let new_args = skip_captures_for_lambda(data.args.clone());
                let scope_id = self.db.alloc_scope(Scope::new(Some(scope)));
                let name = gensym(b"_$_lambda".to_vec());
                let unresolved_body = self.db.alloc_hir(Hir::Unresolved);
                let symbol_id = self.db.alloc_symbol(Symbol::Function(FunctionSymbol {
                    name: Some(Name::new(decode_string(&name), None)),
                    ty: self.any_type_id,
                    scope,
                    vars: Default::default(),
                    parameters: IndexMap::default(),
                    nil_terminated: true,
                    return_type: self.any_type_id,
                    body: unresolved_body,
                    kind: FunctionKind::BinaryTree,
                }));
                let description = PredeclaredHelperSymbol {
                    symbol_id,
                    scope_id,
                    kind: PredeclaredSymbolKind::Defun,
                };
                self.predeclared_helpers.insert(name.to_vec(), description);
                self.create_defun(
                    false,
                    &DefunData {
                        loc: data.loc.clone(),
                        name: name.clone(),
                        args: new_args.clone(),
                        orig_args: new_args,
                        kw: data.kw.clone(),
                        nl: data.loc.clone(),
                        synthetic: None,
                        body: data.body.clone(),
                    },
                )?;
                Ok(self.db.alloc_hir(Hir::Reference(symbol_id)))
            }
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

    fn function_from_defun(
        &self,
        function_name: &str,
        function_scope: ScopeId,
        inline: bool,
        data: &DefunData,
        plist: IndexMap<String, SymbolId>,
        body: HirId,
    ) -> FunctionSymbol {
        FunctionSymbol {
            name: Some(Name::new(
                function_name,
                Some(self.to_rue_srcloc(data.nl.clone())),
            )),
            ty: self.any_type_id,
            scope: function_scope,
            vars: Default::default(),
            nil_terminated: inline,
            return_type: self.any_type_id,
            body,
            parameters: plist,
            kind: if inline {
                FunctionKind::Inline
            } else {
                FunctionKind::BinaryTree
            },
        }
    }

    fn predeclare_defun(
        &mut self,
        function_name: &str,
        function_scope: ScopeId,
        inline: bool,
        data: &DefunData,
    ) -> PredeclaredHelperSymbol {
        let unresolved_body = self.db.alloc_hir(Hir::Unresolved);
        let function_sym = self
            .db
            .alloc_symbol(Symbol::Function(self.function_from_defun(
                function_name,
                function_scope,
                inline,
                data,
                IndexMap::default(),
                unresolved_body,
            )));

        let kind = if inline {
            if let Some(args) = data.args.proper_list() {
                PredeclaredSymbolKind::InlineDefunNormalArgs(args)
            } else if let Some((args, tail)) = improper_list(data.args.clone()) {
                PredeclaredSymbolKind::InlineDefunImproperListArgs(args, tail)
            } else {
                PredeclaredSymbolKind::InlineDefunTreeArgs
            }
        } else {
            PredeclaredSymbolKind::Defun
        };

        PredeclaredHelperSymbol {
            symbol_id: function_sym,
            scope_id: function_scope,
            kind,
        }
    }

    fn predeclare_constant(&mut self, constant_name: &str, data: &DefconstData) -> SymbolId {
        let unresolved_body = self.db.alloc_hir(Hir::Unresolved);
        self.db
            .alloc_symbol(Symbol::Binding(rue_hir::BindingSymbol {
                name: Some(Name::new(
                    constant_name.to_string(),
                    Some(self.to_rue_srcloc(data.nl.clone())),
                )),
                value: Value::new(unresolved_body, self.any_type_id),
                inline: false,
            }))
    }

    fn predeclare_helper_symbols(
        &mut self,
        main_scope: ScopeId,
        helpers: &[HelperForm],
    ) -> Result<(), CompileErr> {
        for helper in helpers {
            // Only these emit actual data into the program.  All other forms were eliminated
            // before we began code generation.
            if let HelperForm::Defun(inline, data) = helper {
                let function_name = decode_string(helper.name());
                let function_scope = self.db.alloc_scope(Scope::new(Some(main_scope)));
                let function_predecl =
                    self.predeclare_defun(&function_name, function_scope, *inline, data);
                self.db.scope_mut(main_scope).insert_symbol(
                    function_name,
                    function_predecl.symbol_id,
                    false,
                );

                self.predeclared_helpers
                    .insert(data.name.clone(), function_predecl);
            } else if let HelperForm::Defconstant(data) = helper {
                let constant_name = decode_string(helper.name());
                let constant_sym = self.predeclare_constant(&constant_name, data);
                self.db.scope_mut(main_scope).insert_symbol(
                    constant_name.to_string(),
                    constant_sym,
                    false,
                );
                self.predeclared_helpers.insert(
                    data.name.clone(),
                    PredeclaredHelperSymbol {
                        symbol_id: constant_sym,
                        scope_id: main_scope,
                        kind: PredeclaredSymbolKind::Constant,
                    },
                );
            }
        }
        Ok(())
    }

    fn create_defun(&mut self, inline: bool, data: &DefunData) -> Result<(), CompileErr> {
        let Some(predecl) = self.predeclared_helpers.get(&data.name).cloned() else {
            return Err(rue_err(
                data.loc.clone(),
                format!(
                    "missing predeclared symbol for helper `{}`",
                    decode_string(&data.name)
                ),
            ));
        };

        let mut install_argument =
            |plist: &mut IndexMap<String, SymbolId>, param_index: usize, param_sexp: &SExp| {
                if let SExp::Atom(ploc, atom_name) = param_sexp {
                    let param_name = decode_string(atom_name);
                    let param_symbol = self.db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
                        name: Some(Name::new(
                            param_name.clone(),
                            Some(self.to_rue_srcloc(ploc.clone())),
                        )),
                        ty: self.any_type_id,
                    }));
                    self.db.scope_mut(predecl.scope_id).insert_symbol(
                        param_name.clone(),
                        param_symbol,
                        false,
                    );
                    plist.insert(param_name, param_symbol);
                } else {
                    // Inline helpers may still destructure argument trees.
                    // Bind the incoming argument, then create accessor helpers for leaves.
                    let param_name = format!("_$_arg_{param_index}");
                    self.create_argument_helpers(
                        plist,
                        &param_name,
                        Rc::new(param_sexp.clone()),
                        predecl.scope_id,
                    );
                }
            };

        let mut plist: IndexMap<String, SymbolId> = IndexMap::default();
        match &predecl.kind {
            PredeclaredSymbolKind::InlineDefunNormalArgs(args) => {
                // In the case of inlines, it's appropriate to try to match the arguments to
                // positions as well as possible.  If the parameter list is a proper list, we can
                // just use each argument.
                for (arg_index, p) in args.iter().enumerate() {
                    install_argument(&mut plist, arg_index, p);
                }
            }
            PredeclaredSymbolKind::InlineDefunImproperListArgs(prefix, tail) => {
                // Improper argument list for inline.
                // It's possible to think about a tail improper list as a list of normal
                // arguments, then a tail of any number of further arguments.  Chialisp supports
                // this.
                for (arg_index, p) in prefix.iter().enumerate() {
                    install_argument(&mut plist, arg_index, p);
                }
                install_argument(&mut plist, prefix.len(), tail);
                self.create_argument_helpers(
                    &mut plist,
                    "_$_args__",
                    data.args.clone(),
                    predecl.scope_id,
                );
            }
            _ => {
                // Chialisp allows an arbitrary argument tree which specifies the exact environment
                // shape. Rue uses either sequential or tree shaped and choose the tree shape at
                // lowering time. The right approach is to use a single argument in BinaryTree
                // mode and make accessors for the individual destructurings chialisp would allow.
                self.create_argument_helpers(
                    &mut plist,
                    "_$_args__",
                    data.args.clone(),
                    predecl.scope_id,
                );
            }
        }

        let body_hir = self.intern_expr_hir(predecl.scope_id, &data.body)?;
        let function_name = decode_string(&data.name);
        *self.db.symbol_mut(predecl.symbol_id) = Symbol::Function(self.function_from_defun(
            &function_name,
            predecl.scope_id,
            inline,
            data,
            plist,
            body_hir,
        ));
        Ok(())
    }

    fn finalize_constant_value(
        &mut self,
        main_scope: ScopeId,
        data: &DefconstData,
        value: &SExp,
    ) -> Result<(), CompileErr> {
        // Intern in the rue data.
        let value_hir = self.intern_sexp_hir(value);
        let accessor_name = decode_string(&data.name);
        let constant_name = format!("_$_{}", accessor_name);
        let new_constant_id = self
            .db
            .alloc_symbol(Symbol::Binding(rue_hir::BindingSymbol {
                name: Some(Name::new(
                    constant_name.clone(),
                    Some(self.to_rue_srcloc(data.nl.clone())),
                )),
                value: Value::new(value_hir, self.any_type_id),
                inline: false,
            }));
        self.db
            .scope_mut(main_scope)
            .insert_symbol(constant_name.clone(), new_constant_id, false);
        let Some(predecl) = self.predeclared_helpers.get(&data.name) else {
            return Err(CompileErr(
                data.loc.clone(),
                format!("Internal error: rue predeclared constant not available: {constant_name}"),
            ));
        };
        let body = self.db.alloc_hir(Hir::Reference(new_constant_id));
        *self.db.symbol_mut(predecl.symbol_id) = Symbol::Function(FunctionSymbol {
            name: Some(Name::new(accessor_name.to_string(), None)),
            ty: self.any_type_id,
            scope: predecl.scope_id,
            vars: Default::default(),
            parameters: IndexMap::default(),
            nil_terminated: true,
            return_type: self.any_type_id,
            body,
            kind: FunctionKind::Inline,
        });
        Ok(())
    }

    fn resolve_constants(
        &mut self,
        main_scope: ScopeId,
        program: &CompileForm,
    ) -> Result<(), CompileErr> {
        let mut avail_constants: HashSet<Vec<u8>> = HashSet::new();

        for h in program.helpers.iter() {
            if let HelperForm::Defconstant(data) = h {
                if let BodyForm::Value(v) = &*data.body {
                    if matches!(v, SExp::Integer(_, _) | SExp::QuotedString(_, _, _)) {
                        self.finalize_constant_value(main_scope, data, v)?;
                        continue;
                    }
                } else if let BodyForm::Quoted(v) = &*data.body {
                    self.finalize_constant_value(main_scope, data, v)?;
                    continue;
                }

                avail_constants.insert(h.name().to_vec());
            }
        }

        let depgraph = FunctionDependencyGraph::new_with_options(
            program,
            DepgraphOptions {
                with_constants: true,
            },
        );

        // Process delayed constants until we either can't advance or
        // they're all done.
        while !avail_constants.is_empty() {
            let mut allocator = Allocator::new();
            // Find a constant that depends on no other constants.
            let mut replacement = None;

            for helper in program.helpers.iter() {
                if let HelperForm::Defconstant(data) = helper {
                    if !avail_constants.contains(&data.name) {
                        continue;
                    }

                    let mut depends_on: HashSet<Vec<u8>> = HashSet::default();
                    depgraph.get_full_depends_on(&mut depends_on, &data.name);
                    let still_depends_on: HashSet<&Vec<u8>> =
                        depends_on.intersection(&avail_constants).collect();
                    if !still_depends_on.is_empty() {
                        continue;
                    }

                    // We have a constant which depends on nothing else that we haven't generated
                    // yet.
                    //
                    // Produce its program.
                    replacement = Some((data.clone(), depends_on));
                    break;
                }
            }

            let Some((data, depends_on)) = replacement else {
                let constants_remaining: Vec<String> =
                    avail_constants.iter().map(|s| decode_string(s)).collect();
                return Err(CompileErr(
                    program.loc(),
                    format!("Deadlock generating constant with remaining {constants_remaining:?}"),
                ));
            };

            let program_with_constants = CompileForm {
                helpers: program
                    .helpers
                    .iter()
                    .filter(|h| depends_on.contains(h.name()))
                    .cloned()
                    .collect(),
                args: Rc::new(SExp::Nil(data.loc.clone())),
                exp: data.body.clone(),
                ..program.clone()
            };
            let compiled_program = compile_with_rue_codegen(
                self.opts.clone(),
                Arc::from(""),
                &program_with_constants,
            )?;
            let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
            let node =
                convert_to_clvm_rs(&mut allocator, Rc::new(compiled_program)).map_err(|e| {
                    CompileErr(
                        data.body.loc(),
                        format!("runtime error generating constant value: {:?}", e),
                    )
                })?;
            let nil = allocator.nil();
            let node_result = runner
                .run_program(&mut allocator, node, nil, None)
                .map_err(|e| {
                    CompileErr(
                        data.body.loc(),
                        format!("runtime error generating constant value: {:?}", e),
                    )
                })?;
            let clvm_value =
                convert_from_clvm_rs(&mut allocator, program.loc.clone(), node_result.1).map_err(
                    |e| {
                        CompileErr(
                            data.body.loc(),
                            format!("runtime error generating constant value: {:?}", e),
                        )
                    },
                )?;
            self.finalize_constant_value(main_scope, &data, &clvm_value)?;
            avail_constants.remove(&data.name);
        }

        Ok(())
    }

    fn create_argument_helpers(
        &mut self,
        params_map: &mut IndexMap<String, SymbolId>,
        param_name: &str,
        args: Rc<SExp>,
        program_scope: ScopeId,
    ) {
        // Program arguments are tree-shaped in chialisp, so model them the same way as
        // non-inline defun arguments: one binary-tree parameter and atom accessors.
        let main_args_symbol = self.db.alloc_symbol(Symbol::Parameter(ParameterSymbol {
            name: Some(Name::new(param_name, Some(self.to_rue_srcloc(args.loc())))),
            ty: self.any_type_id,
        }));
        params_map.insert(param_name.to_string(), main_args_symbol);
        self.db.scope_mut(program_scope).insert_symbol(
            param_name.to_string(),
            main_args_symbol,
            false,
        );
        self.install_tree_arg_accessors(program_scope, args.clone(), main_args_symbol);
        self.install_env_alias_helpers(program_scope, main_args_symbol);
        // Construct inline helper functions for each printable atom in the argument tree.
        self.install_tree_arg_accessors(program_scope, args.clone(), main_args_symbol);
    }

    fn intern_hir(&mut self, program: &CompileForm) -> Result<SymbolId, CompileErr> {
        let main_scope_id: ScopeId = self.db.alloc_scope(Scope::new(None));
        self.predeclare_helper_symbols(main_scope_id, &program.helpers)?;
        self.resolve_constants(main_scope_id, program)?;
        for h in program.helpers.iter() {
            // Macros and other forms were handled during preprocessing and other
            // passes before code generation.
            if let HelperForm::Defun(inline, data) = &h {
                self.create_defun(*inline, data)?;
            }
        }

        let program_scope = self.db.alloc_scope(Scope::new(Some(main_scope_id)));
        let mut main_params: IndexMap<String, SymbolId> = IndexMap::default();
        self.create_argument_helpers(
            &mut main_params,
            "_$_args__",
            program.args.clone(),
            program_scope,
        );

        let main_body = self.intern_expr_hir(program_scope, &program.exp)?;
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
