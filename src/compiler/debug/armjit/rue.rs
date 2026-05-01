use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use clvmr::Allocator;
use id_arena::Arena;
use indexmap::{IndexMap, IndexSet};
use rue_compiler::{normalize_path, Compiler, FileTree, SyntaxItemKind};
use rue_diagnostic::{DiagnosticSeverity, Source, SourceKind, SrcLoc};
use rue_hir::{
    BinaryOp, BindingSymbol, ConstantSymbol, Database, DependencyGraph, Environment, ExprStatement,
    FunctionCall, FunctionKind, FunctionSymbol, Hir, HirId, IfStatement, PathError, Statement,
    Symbol, SymbolId, UnaryOp,
};
use rue_lir::{bigint_atom, ClvmOp, Lir, LirId};
use rue_options::find_project;

use crate::classic::clvm_tools::binutils::assemble;
use crate::compiler::clvm::convert_from_clvm_rs;
use crate::compiler::debug::armjit::code::{ElfObject, Program, TARGET_ADDR};
use crate::compiler::debug::{debug_decode_string, DebugSExp, DebugSExpValue, Srcloc};
use crate::compiler::sexp::SExp;
use crate::util::Number;

use super::cmd::{Args, ArmElfCompileOutput};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RueSExp(Rc<SExp>);

impl RueSExp {
    fn new(sexp: SExp) -> Self {
        Self(Rc::new(sexp))
    }
}

impl fmt::Display for RueSExp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl DebugSExp for RueSExp {
    fn atom(loc: Srcloc, bytes: &[u8]) -> Self {
        Self::new(SExp::Atom(loc, bytes.to_vec()))
    }

    fn loc(&self) -> Srcloc {
        self.0.loc()
    }

    fn atomize(&self) -> Self {
        RueSExp(self.0.atomize())
    }

    fn to_number(&self) -> Option<Number> {
        self.0.get_number().ok()
    }

    fn proper_list(&self) -> Option<Vec<Self>> {
        let mut result = Vec::new();
        let mut rest = self.clone();

        loop {
            if rest.nilp() {
                return Some(result);
            }

            match rest.explode() {
                DebugSExpValue::Cons(_, left, right) => {
                    result.push(left);
                    rest = right;
                }
                _ => return None,
            }
        }
    }

    fn explode(&self) -> DebugSExpValue<Self> {
        match self.0.as_ref() {
            SExp::Nil(loc) => DebugSExpValue::Nil(loc.clone()),
            SExp::Cons(loc, left, right) => {
                DebugSExpValue::Cons(loc.clone(), RueSExp(left.clone()), RueSExp(right.clone()))
            }
            SExp::Integer(loc, number) => DebugSExpValue::Integer(loc.clone(), number.clone()),
            SExp::QuotedString(loc, quote, bytes) => {
                DebugSExpValue::QuotedString(loc.clone(), *quote, bytes.clone())
            }
            SExp::Atom(loc, bytes) => DebugSExpValue::Atom(loc.clone(), bytes.clone()),
        }
    }
}

#[derive(Debug, Clone)]
enum SymbolGroup {
    Sequential(Vec<SymbolId>),
    Tree(Environment),
}

impl SymbolGroup {
    fn is_empty(&self) -> bool {
        match self {
            Self::Sequential(symbols) => symbols.is_empty(),
            Self::Tree(Environment::Nil) => true,
            Self::Tree(_) => false,
        }
    }
}

struct DebugLowerer<'d, 'a, 'g> {
    db: &'d mut Database,
    arena: &'a mut Arena<Lir>,
    graph: &'g DependencyGraph,
    lir_locs: &'a mut HashMap<LirId, Srcloc>,
    function_body_lirs: &'a mut HashMap<SymbolId, LirId>,
    inline_symbols: Vec<HashMap<SymbolId, HirId>>,
    options: rue_options::CompilerOptions,
    main: SymbolId,
    base_path: PathBuf,
    symbol_locs: &'a HashMap<SymbolId, Srcloc>,
    current_loc: Srcloc,
}

impl<'d, 'a, 'g> DebugLowerer<'d, 'a, 'g> {
    fn new(
        db: &'d mut Database,
        arena: &'a mut Arena<Lir>,
        graph: &'g DependencyGraph,
        lir_locs: &'a mut HashMap<LirId, Srcloc>,
        function_body_lirs: &'a mut HashMap<SymbolId, LirId>,
        options: rue_options::CompilerOptions,
        main: SymbolId,
        base_path: PathBuf,
        symbol_locs: &'a HashMap<SymbolId, Srcloc>,
        fallback_loc: Srcloc,
    ) -> Self {
        Self {
            db,
            arena,
            graph,
            lir_locs,
            function_body_lirs,
            inline_symbols: Vec::new(),
            options,
            main,
            base_path,
            symbol_locs,
            current_loc: fallback_loc,
        }
    }

    fn alloc(&mut self, lir: Lir, loc: Srcloc) -> LirId {
        let id = self.arena.alloc(lir);
        self.lir_locs.insert(id, loc);
        id
    }

    fn alloc_here(&mut self, lir: Lir) -> LirId {
        self.alloc(lir, self.current_loc.clone())
    }

    fn with_loc<R>(&mut self, loc: Srcloc, f: impl FnOnce(&mut Self) -> R) -> R {
        let previous = self.current_loc.clone();
        self.current_loc = loc;
        let result = f(self);
        self.current_loc = previous;
        result
    }

    fn lower_symbol_value(&mut self, env: &Environment, symbol: SymbolId) -> LirId {
        for inline_symbols in self.inline_symbols.iter().rev() {
            if let Some(hir) = inline_symbols.get(&symbol) {
                return self.lower_hir(env, *hir);
            }
        }

        let loc = self
            .symbol_locs
            .get(&symbol)
            .cloned()
            .unwrap_or_else(|| self.current_loc.clone());
        self.with_loc(loc, |lowerer| match lowerer.db.symbol(symbol).clone() {
            Symbol::Unresolved | Symbol::Module(_) | Symbol::Parameter(_) | Symbol::Builtin(_) => {
                unreachable!()
            }
            Symbol::Function(function) => lowerer.lower_function(env, symbol, function),
            Symbol::Constant(constant) => lowerer.lower_constant(env, constant),
            Symbol::Binding(binding) => lowerer.lower_binding(env, binding),
        })
    }

    fn function_groups(
        &mut self,
        symbol: SymbolId,
        function: &FunctionSymbol,
    ) -> (Vec<SymbolGroup>, SymbolGroup) {
        let captures: Vec<SymbolId> = self
            .graph
            .dependencies(symbol, true)
            .into_iter()
            .filter(|&symbol| !self.should_inline(symbol))
            .collect();

        let capture_groups = self.group_symbols(
            captures.into_iter().collect(),
            symbol != self.main,
            !self.graph.is_closure(symbol),
        );

        let param_group = self.create_group(
            function.parameters.values().copied().collect(),
            function.kind == FunctionKind::BinaryTree
                && !self.graph.is_closure(symbol)
                && symbol != self.main,
        );

        (capture_groups, param_group)
    }

    fn lower_function(
        &mut self,
        parent_env: &Environment,
        symbol: SymbolId,
        function: FunctionSymbol,
    ) -> LirId {
        let (capture_groups, param_group) = self.function_groups(symbol, &function);

        let mut function_env = Self::apply_group(
            Environment::Nil,
            &param_group,
            function.nil_terminated && matches!(param_group, SymbolGroup::Sequential(_)),
        );

        for group in &capture_groups {
            function_env = Self::apply_group(function_env, group, true);
        }

        let mut expr = self.lower_hir(&function_env, function.body);

        if symbol == self.main {
            let mut map = HashMap::new();

            if self
                .graph
                .dependencies(symbol, true)
                .iter()
                .any(|dependency| *dependency == symbol)
            {
                let quoted = self.alloc_here(Lir::Quote(expr));
                map.insert(symbol, quoted);
                let reference = self.lower_symbol_reference(&function_env, symbol);
                let entire_env = self.alloc_here(Lir::Path(1));
                expr = self.alloc_here(Lir::Run(reference, entire_env));
            }

            for (i, group) in capture_groups.iter().enumerate().rev() {
                expr = self.alloc_here(Lir::Quote(expr));

                let mut bind_env = parent_env.clone();

                for existing_group in capture_groups.iter().take(i) {
                    bind_env = Self::apply_group(bind_env, existing_group, true);
                }

                let rest = if param_group.is_empty() {
                    self.alloc_here(Lir::Atom(vec![]))
                } else {
                    self.alloc_here(Lir::Path(1))
                };

                let group_env =
                    self.lower_group_environment(&bind_env, group, rest, false, Some(&map), true);

                expr = self.alloc_here(Lir::Run(expr, group_env));
            }

            self.function_body_lirs.insert(symbol, expr);
            expr
        } else {
            self.function_body_lirs.insert(symbol, expr);
            self.alloc_here(Lir::Quote(expr))
        }
    }

    fn lower_constant(&mut self, env: &Environment, constant: ConstantSymbol) -> LirId {
        self.lower_hir(env, constant.value.hir)
    }

    fn lower_binding(&mut self, env: &Environment, binding: BindingSymbol) -> LirId {
        self.lower_hir(env, binding.value.hir)
    }

    fn lower_hir(&mut self, env: &Environment, hir: HirId) -> LirId {
        match self.db.hir(hir).clone() {
            Hir::Unresolved => unreachable!(),
            Hir::Nil => self.alloc_here(Lir::Atom(vec![])),
            Hir::String(value) => self.alloc_here(Lir::Atom(value.as_bytes().to_vec())),
            Hir::Int(value) => self.alloc_here(Lir::Atom(bigint_atom(value.clone()))),
            Hir::Bool(value) => self.alloc_here(Lir::Atom(if value { vec![1] } else { vec![] })),
            Hir::Bytes(atom) => self.alloc_here(Lir::Atom(atom)),
            Hir::Pair(first, rest) => {
                let first = self.lower_hir(env, first);
                let rest = self.lower_hir(env, rest);
                self.alloc_here(Lir::Cons(first, rest))
            }
            Hir::Reference(symbol) => self.lower_symbol(env, symbol, false),
            Hir::Block(block) => self.lower_block(env, block.statements, block.body),
            Hir::Lambda(lambda) => self.lower_symbol(env, lambda, true),
            Hir::If(condition, then, else_, inline) => {
                let condition = self.lower_hir(env, condition);
                let then = self.lower_hir(env, then);
                let else_ = self.lower_hir(env, else_);
                self.alloc_here(Lir::If(condition, then, else_, inline))
            }
            Hir::FunctionCall(call) => self.lower_function_call(env, call),
            Hir::Unary(op, hir) => {
                let lir = self.lower_hir(env, hir);
                match op {
                    UnaryOp::Listp { can_be_truthy } => {
                        self.alloc_here(Lir::Listp(lir, can_be_truthy))
                    }
                    UnaryOp::First => self.alloc_here(Lir::First(lir)),
                    UnaryOp::Rest => self.alloc_here(Lir::Rest(lir)),
                    UnaryOp::Strlen => self.alloc_here(Lir::Strlen(lir)),
                    UnaryOp::Not => self.alloc_here(Lir::Not(lir)),
                    UnaryOp::Neg => {
                        let zero = self.alloc_here(Lir::Atom(vec![]));
                        self.alloc_here(Lir::Sub(vec![zero, lir]))
                    }
                    UnaryOp::BitwiseNot => self.alloc_here(Lir::Lognot(lir)),
                    UnaryOp::G1Negate => self.alloc_here(Lir::G1Negate(lir)),
                    UnaryOp::G2Negate => self.alloc_here(Lir::G2Negate(lir)),
                    UnaryOp::Sha256 => self.alloc_here(Lir::Sha256(vec![lir])),
                    UnaryOp::Sha256Inline => self.alloc_here(Lir::Sha256Inline(vec![lir])),
                    UnaryOp::Keccak256 => self.alloc_here(Lir::Keccak256(vec![lir])),
                    UnaryOp::Keccak256Inline => self.alloc_here(Lir::Keccak256Inline(vec![lir])),
                    UnaryOp::PubkeyForExp => self.alloc_here(Lir::PubkeyForExp(lir)),
                }
            }
            Hir::Binary(op, left, right) => {
                let left = self.lower_hir(env, left);
                let right = self.lower_hir(env, right);
                self.lower_binary(op, left, right)
            }
            Hir::CoinId(parent, puzzle, amount) => {
                let parent = self.lower_hir(env, parent);
                let puzzle = self.lower_hir(env, puzzle);
                let amount = self.lower_hir(env, amount);
                self.alloc_here(Lir::CoinId(parent, puzzle, amount))
            }
            Hir::Substr(hir, start, end) => {
                let hir = self.lower_hir(env, hir);
                let start = self.lower_hir(env, start);
                let end = end.map(|end| self.lower_hir(env, end));
                self.alloc_here(Lir::Substr(hir, start, end))
            }
            Hir::G1Map(data, dst) => {
                let data = self.lower_hir(env, data);
                let dst = dst.map(|dst| self.lower_hir(env, dst));
                self.alloc_here(Lir::G1Map(data, dst))
            }
            Hir::G2Map(data, dst) => {
                let data = self.lower_hir(env, data);
                let dst = dst.map(|dst| self.lower_hir(env, dst));
                self.alloc_here(Lir::G2Map(data, dst))
            }
            Hir::Modpow(base, exponent, modulus) => {
                let base = self.lower_hir(env, base);
                let exponent = self.lower_hir(env, exponent);
                let modulus = self.lower_hir(env, modulus);
                self.alloc_here(Lir::Modpow(base, exponent, modulus))
            }
            Hir::BlsPairingIdentity(args) => {
                let args = args
                    .into_iter()
                    .map(|arg| self.lower_hir(env, arg))
                    .collect();
                self.alloc_here(Lir::BlsPairingIdentity(args))
            }
            Hir::BlsVerify(sig, args) => {
                let sig = self.lower_hir(env, sig);
                let args = args
                    .into_iter()
                    .map(|arg| self.lower_hir(env, arg))
                    .collect();
                self.alloc_here(Lir::BlsVerify(sig, args))
            }
            Hir::Secp256K1Verify(sig, pk, msg) => {
                let sig = self.lower_hir(env, sig);
                let pk = self.lower_hir(env, pk);
                let msg = self.lower_hir(env, msg);
                self.alloc_here(Lir::K1Verify(sig, pk, msg))
            }
            Hir::Secp256R1Verify(sig, pk, msg) => {
                let sig = self.lower_hir(env, sig);
                let pk = self.lower_hir(env, pk);
                let msg = self.lower_hir(env, msg);
                self.alloc_here(Lir::R1Verify(sig, pk, msg))
            }
            Hir::InfinityG1 => self.alloc_here(Lir::G1Add(vec![])),
            Hir::InfinityG2 => self.alloc_here(Lir::G2Add(vec![])),
            Hir::ClvmOp(op, args) => {
                let args = self.lower_hir(env, args);
                self.alloc_here(Lir::Op(op, args))
            }
        }
    }

    fn lower_binary(&mut self, op: BinaryOp, left: LirId, right: LirId) -> LirId {
        match op {
            BinaryOp::Add => self.alloc_here(Lir::Add(vec![left, right])),
            BinaryOp::Sub => self.alloc_here(Lir::Sub(vec![left, right])),
            BinaryOp::Mul => self.alloc_here(Lir::Mul(vec![left, right])),
            BinaryOp::Div => self.alloc_here(Lir::Div(left, right)),
            BinaryOp::Mod => self.alloc_here(Lir::Mod(left, right)),
            BinaryOp::Divmod => self.alloc_here(Lir::Divmod(left, right)),
            BinaryOp::Concat => self.alloc_here(Lir::Concat(vec![left, right])),
            BinaryOp::G1Add => self.alloc_here(Lir::G1Add(vec![left, right])),
            BinaryOp::G1Subtract => self.alloc_here(Lir::G1Subtract(vec![left, right])),
            BinaryOp::G1Multiply => self.alloc_here(Lir::G1Multiply(left, right)),
            BinaryOp::G2Add => self.alloc_here(Lir::G2Add(vec![left, right])),
            BinaryOp::G2Subtract => self.alloc_here(Lir::G2Subtract(vec![left, right])),
            BinaryOp::G2Multiply => self.alloc_here(Lir::G2Multiply(left, right)),
            BinaryOp::BitwiseAnd => self.alloc_here(Lir::Logand(vec![left, right])),
            BinaryOp::BitwiseOr => self.alloc_here(Lir::Logior(vec![left, right])),
            BinaryOp::BitwiseXor => self.alloc_here(Lir::Logxor(vec![left, right])),
            BinaryOp::RightLogicalShift => {
                let zero = self.alloc_here(Lir::Atom(vec![]));
                let neg = self.alloc_here(Lir::Sub(vec![zero, right]));
                self.alloc_here(Lir::Lsh(left, neg))
            }
            BinaryOp::RightArithmeticShift => {
                let zero = self.alloc_here(Lir::Atom(vec![]));
                let neg = self.alloc_here(Lir::Sub(vec![zero, right]));
                self.alloc_here(Lir::Ash(left, neg))
            }
            BinaryOp::LeftArithmeticShift => self.alloc_here(Lir::Ash(left, right)),
            BinaryOp::Gt => self.alloc_here(Lir::Gt(left, right)),
            BinaryOp::Lt => self.alloc_here(Lir::Gt(right, left)),
            BinaryOp::Gte => {
                let gt = self.alloc_here(Lir::Gt(left, right));
                let eq = self.alloc_here(Lir::Eq(left, right));
                self.alloc_here(Lir::Any(vec![gt, eq]))
            }
            BinaryOp::Lte => {
                let lt = self.alloc_here(Lir::Gt(right, left));
                let eq = self.alloc_here(Lir::Eq(left, right));
                self.alloc_here(Lir::Any(vec![lt, eq]))
            }
            BinaryOp::GtBytes => self.alloc_here(Lir::GtBytes(left, right)),
            BinaryOp::LtBytes => self.alloc_here(Lir::GtBytes(right, left)),
            BinaryOp::GteBytes => {
                let gt = self.alloc_here(Lir::GtBytes(left, right));
                let eq = self.alloc_here(Lir::Eq(left, right));
                self.alloc_here(Lir::Any(vec![gt, eq]))
            }
            BinaryOp::LteBytes => {
                let lt = self.alloc_here(Lir::GtBytes(right, left));
                let eq = self.alloc_here(Lir::Eq(left, right));
                self.alloc_here(Lir::Any(vec![lt, eq]))
            }
            BinaryOp::Eq => self.alloc_here(Lir::Eq(left, right)),
            BinaryOp::Ne => {
                let eq = self.alloc_here(Lir::Eq(left, right));
                self.alloc_here(Lir::Not(eq))
            }
            BinaryOp::And => {
                let true_atom = self.alloc_here(Lir::Atom(vec![1]));
                let false_atom = self.alloc_here(Lir::Atom(vec![]));
                let right = self.alloc_here(Lir::If(right, true_atom, false_atom, false));
                self.alloc_here(Lir::If(left, right, false_atom, false))
            }
            BinaryOp::Or => {
                let true_atom = self.alloc_here(Lir::Atom(vec![1]));
                let false_atom = self.alloc_here(Lir::Atom(vec![]));
                let right = self.alloc_here(Lir::If(right, true_atom, false_atom, false));
                self.alloc_here(Lir::If(left, true_atom, right, false))
            }
            BinaryOp::All => self.alloc_here(Lir::All(vec![left, right])),
            BinaryOp::Any => self.alloc_here(Lir::Any(vec![left, right])),
        }
    }

    fn lower_function_call(&mut self, env: &Environment, call: FunctionCall) -> LirId {
        if let Some((symbol, function)) = match self.db.hir(call.function).clone() {
            Hir::Reference(symbol) => match self.db.symbol(symbol).clone() {
                Symbol::Function(function) => Some((symbol, function)),
                _ => None,
            },
            _ => None,
        } {
            let mut args = HashMap::new();

            if function.nil_terminated {
                for (i, arg) in call.args.into_iter().enumerate() {
                    args.insert(function.parameters[i], arg);
                }
            } else {
                let mut arg_iter = call.args.into_iter().enumerate();

                for (i, arg) in (&mut arg_iter).take(function.parameters.len() - 1) {
                    args.insert(function.parameters[i], arg);
                }

                let mut last_arg = self.db.alloc_hir(Hir::Nil);

                for (i, (_, arg)) in arg_iter.rev().enumerate() {
                    if i == 0 && !call.nil_terminated {
                        last_arg = arg;
                    } else {
                        last_arg = self.db.alloc_hir(Hir::Pair(arg, last_arg));
                    }
                }

                if let Some(last_param) = function.parameters.last() {
                    args.insert(*last_param.1, last_arg);
                }
            }

            if self.should_inline(symbol) {
                self.inline_symbols.push(args);
                let result = self.lower_hir(env, function.body);
                self.inline_symbols.pop().unwrap();
                return result;
            }

            let (capture_groups, param_group) = self.function_groups(symbol, &function);
            let function_lir = self.lower_symbol_reference(env, symbol);
            let rest = self.alloc_here(Lir::Atom(vec![]));

            let mut lir_args = HashMap::new();
            for (symbol, hir) in &args {
                lir_args.insert(*symbol, self.lower_hir(env, *hir));
            }

            let mut arg_env = self.lower_group_environment(
                env,
                &param_group,
                rest,
                false,
                Some(&lir_args),
                function.nil_terminated && matches!(param_group, SymbolGroup::Sequential(_)),
            );

            for group in &capture_groups {
                arg_env = self.lower_group_environment(env, group, arg_env, true, None, true);
            }

            return self.alloc_here(Lir::Run(function_lir, arg_env));
        }

        let function = self.lower_hir(env, call.function);
        let mut args = Vec::new();

        for arg in call.args {
            args.push(self.lower_hir(env, arg));
        }

        let mut env = self.alloc_here(Lir::Atom(Vec::new()));

        for (i, &arg) in args.iter().rev().enumerate() {
            if i == 0 && !call.nil_terminated {
                env = arg;
            } else {
                env = self.alloc_here(Lir::Cons(arg, env));
            }
        }

        self.alloc_here(Lir::Run(function, env))
    }

    fn lower_symbol_reference(&mut self, env: &Environment, symbol: SymbolId) -> LirId {
        for inline_symbols in self.inline_symbols.iter().rev() {
            if let Some(hir) = inline_symbols.get(&symbol) {
                return self.lower_hir(env, *hir);
            }
        }

        self.lower_path(env, symbol)
    }

    fn lower_symbol(&mut self, env: &Environment, symbol: SymbolId, is_lambda: bool) -> LirId {
        let mut reference = if self.should_inline(symbol) || is_lambda {
            self.lower_symbol_value(env, symbol)
        } else {
            self.lower_symbol_reference(env, symbol)
        };

        if let Symbol::Function(function) = self.db.symbol(symbol).clone() {
            let captures: Vec<SymbolId> = self
                .graph
                .dependencies(symbol, true)
                .into_iter()
                .filter(|&symbol| !self.should_inline(symbol))
                .collect();

            let mut refs = Vec::new();
            for capture in captures {
                refs.push(self.lower_symbol_reference(env, capture));
            }

            reference = self.alloc_here(Lir::Closure(
                reference,
                refs,
                !function.parameters.is_empty(),
            ));
        }

        reference
    }

    fn lower_path(&mut self, env: &Environment, symbol: SymbolId) -> LirId {
        self.alloc_here(Lir::Path(env.path(symbol).unwrap_or_else(
            |error| match error {
                PathError::SymbolNotFound => panic!(
                    "clvm path in environment not found for symbol {}",
                    self.db.debug_symbol(symbol)
                ),
                PathError::PathTooLarge => panic!(
                    "calculated clvm path too large for symbol {}",
                    self.db.debug_symbol(symbol)
                ),
            },
        )))
    }

    fn lower_block(
        &mut self,
        env: &Environment,
        mut stmts: Vec<Statement>,
        body: Option<HirId>,
    ) -> LirId {
        if !self.options.debug_symbols {
            stmts.retain(|stmt| !matches!(stmt, Statement::Debug(..)));
        }

        let Some(stmt) = stmts.first().cloned() else {
            return if let Some(body) = body {
                self.lower_hir(env, body)
            } else {
                self.alloc_here(Lir::Atom(vec![]))
            };
        };

        match stmt {
            Statement::Let(_) => self.lower_let_stmts(env, stmts, body),
            Statement::Return(hir) => self.lower_block(env, vec![], Some(hir)),
            Statement::Assert(condition, srcloc) => {
                self.lower_assert(env, stmts, condition, srcloc, body)
            }
            Statement::Expr(stmt) => self.lower_expr_stmts(env, stmt, stmts, body),
            Statement::Raise(hir, srcloc) => self.lower_raise(env, hir, srcloc),
            Statement::If(stmt) => self.lower_if(env, stmts, stmt, body),
            Statement::Debug(hir, srcloc) => self.lower_debug(env, stmts, hir, srcloc, body),
        }
    }

    fn lower_let_stmts(
        &mut self,
        env: &Environment,
        mut stmts: Vec<Statement>,
        body: Option<HirId>,
    ) -> LirId {
        let mut symbols = IndexSet::new();

        while let Some(Statement::Let(symbol)) = stmts.first() {
            symbols.insert(*symbol);
            stmts.remove(0);
        }

        let binding_groups = self.group_symbols(symbols, false, true);
        let mut body_env = env.clone();

        for group in &binding_groups {
            body_env = Self::apply_group(body_env, group, true);
        }

        let mut expr = self.lower_block(&body_env, stmts, body);

        for (i, group) in binding_groups.iter().enumerate().rev() {
            expr = self.alloc_here(Lir::Quote(expr));
            let mut bind_env = env.clone();

            for group in binding_groups.iter().take(i) {
                bind_env = Self::apply_group(bind_env, group, true);
            }

            let rest = self.alloc_here(Lir::Path(1));
            let group_env = self.lower_group_environment(&bind_env, group, rest, false, None, true);
            expr = self.alloc_here(Lir::Run(expr, group_env));
        }

        expr
    }

    fn lower_group_environment(
        &mut self,
        env: &Environment,
        group: &SymbolGroup,
        rest: LirId,
        by_reference: bool,
        map: Option<&HashMap<SymbolId, LirId>>,
        include_rest: bool,
    ) -> LirId {
        match group {
            SymbolGroup::Sequential(symbols) => {
                let mut result = rest;

                for (i, &symbol) in symbols.iter().rev().enumerate() {
                    let value = if let Some(lir) = map.and_then(|map| map.get(&symbol)) {
                        *lir
                    } else if by_reference {
                        self.lower_symbol_reference(env, symbol)
                    } else {
                        self.lower_symbol_value(env, symbol)
                    };
                    if !include_rest && i == 0 {
                        result = value;
                    } else {
                        result = self.alloc_here(Lir::Cons(value, result));
                    }
                }

                result
            }
            SymbolGroup::Tree(tree) => {
                let tree = self.lower_tree(env, tree, by_reference, map);
                if include_rest {
                    self.alloc_here(Lir::Cons(tree, rest))
                } else {
                    tree
                }
            }
        }
    }

    fn lower_tree(
        &mut self,
        env: &Environment,
        tree: &Environment,
        by_reference: bool,
        map: Option<&HashMap<SymbolId, LirId>>,
    ) -> LirId {
        match tree {
            Environment::Nil => self.alloc_here(Lir::Atom(vec![])),
            Environment::Leaf(symbol) => {
                if let Some(lir) = map.and_then(|map| map.get(symbol)) {
                    *lir
                } else if by_reference {
                    self.lower_symbol_reference(env, *symbol)
                } else {
                    self.lower_symbol_value(env, *symbol)
                }
            }
            Environment::Pair(first, rest) => {
                let first = self.lower_tree(env, first, by_reference, map);
                let rest = self.lower_tree(env, rest, by_reference, map);
                self.alloc_here(Lir::Cons(first, rest))
            }
        }
    }

    fn lower_assert(
        &mut self,
        env: &Environment,
        mut stmts: Vec<Statement>,
        condition: HirId,
        srcloc: SrcLoc,
        body: Option<HirId>,
    ) -> LirId {
        stmts.remove(0);
        let loc = rue_srcloc_to_chialisp(&srcloc);
        self.with_loc(loc, |lowerer| {
            let raise = if lowerer.options.debug_symbols {
                let error = lowerer.alloc_here(Lir::Atom(
                    format!("assertion failed at {}", srcloc.display(&lowerer.base_path))
                        .into_bytes(),
                ));
                vec![error]
            } else {
                vec![]
            };

            let condition = lowerer.lower_hir(env, condition);
            let then_branch = lowerer.lower_block(env, stmts, body);
            let else_branch = lowerer.alloc_here(Lir::Raise(raise));
            lowerer.alloc_here(Lir::If(condition, then_branch, else_branch, false))
        })
    }

    fn lower_raise(&mut self, env: &Environment, hir: Option<HirId>, srcloc: SrcLoc) -> LirId {
        let loc = rue_srcloc_to_chialisp(&srcloc);
        self.with_loc(loc, |lowerer| {
            if !lowerer.options.debug_symbols {
                return lowerer.alloc_here(Lir::Raise(vec![]));
            }

            let error = lowerer.alloc_here(Lir::Atom(
                format!("raise called at {}", srcloc.display(&lowerer.base_path)).into_bytes(),
            ));
            let lir = hir.map(|hir| lowerer.lower_hir(env, hir));
            let mut args = vec![error];

            if let Some(hir) = lir {
                args.push(hir);
            }

            lowerer.alloc_here(Lir::Raise(args))
        })
    }

    fn lower_debug(
        &mut self,
        env: &Environment,
        mut stmts: Vec<Statement>,
        value: HirId,
        srcloc: SrcLoc,
        body: Option<HirId>,
    ) -> LirId {
        stmts.remove(0);
        let loc = rue_srcloc_to_chialisp(&srcloc);
        self.with_loc(loc, |lowerer| {
            let rest = lowerer.lower_block(env, stmts, body);
            let value = lowerer.lower_hir(env, value);
            let print =
                lowerer.alloc_here(Lir::DebugPrint(srcloc.display(&lowerer.base_path), value));
            let nil = lowerer.alloc_here(Lir::Atom(vec![]));
            lowerer.alloc_here(Lir::If(print, nil, rest, false))
        })
    }

    fn lower_if(
        &mut self,
        env: &Environment,
        mut stmts: Vec<Statement>,
        stmt: IfStatement,
        body: Option<HirId>,
    ) -> LirId {
        stmts.remove(0);
        let condition = self.lower_hir(env, stmt.condition);
        let then_branch = self.lower_hir(env, stmt.then);
        let else_branch = self.lower_block(env, stmts, body);
        self.alloc_here(Lir::If(condition, then_branch, else_branch, stmt.inline))
    }

    fn lower_expr_stmts(
        &mut self,
        env: &Environment,
        stmt: ExprStatement,
        mut stmts: Vec<Statement>,
        body: Option<HirId>,
    ) -> LirId {
        if self.options.debug_symbols {
            stmts.remove(0);

            let rest = self.lower_block(env, stmts, body);
            let value = self.lower_hir(env, stmt.hir);
            let one = self.alloc_here(Lir::Atom(vec![1]));
            return self.alloc_here(Lir::If(one, rest, value, true));
        }

        let mut ids = Vec::new();

        while let Some(Statement::Expr(expr)) = stmts.first().cloned() {
            ids.push((self.lower_hir(env, expr.hir), expr.always_nil));
            stmts.remove(0);
        }

        let expr = self.lower_block(env, stmts, body);

        if ids.is_empty() {
            return expr;
        }

        let (condition, verifications) = if ids.len() == 1 {
            let condition = self.alloc_here(Lir::Atom(vec![]));
            (condition, ids[0].0)
        } else if let Some(index) = ids.iter().position(|(_, always_nil)| *always_nil) {
            let (condition, _) = ids.remove(index);
            let id = if ids.len() == 1 {
                ids[0].0
            } else {
                self.alloc_here(Lir::All(ids.iter().map(|(id, _)| *id).collect()))
            };
            (condition, id)
        } else {
            let condition = self.alloc_here(Lir::Atom(vec![]));
            let id = if ids.len() == 1 {
                ids[0].0
            } else {
                self.alloc_here(Lir::All(ids.iter().map(|(id, _)| *id).collect()))
            };
            (condition, id)
        };

        self.alloc_here(Lir::If(condition, verifications, expr, true))
    }

    fn group_symbols(
        &mut self,
        mut symbols: IndexSet<SymbolId>,
        by_reference: bool,
        is_tree: bool,
    ) -> Vec<SymbolGroup> {
        let mut groups = Vec::new();

        while !symbols.is_empty() {
            let mut group = Vec::new();
            let remaining = symbols.clone();

            symbols.retain(|&symbol| {
                if self
                    .graph
                    .dependencies(symbol, true)
                    .iter()
                    .all(|symbol| !remaining.contains(symbol))
                    || matches!(self.db.symbol(symbol), Symbol::Function(_))
                    || by_reference
                {
                    if !self.should_inline(symbol) && self.graph.references(symbol) > 0 {
                        group.push(symbol);
                    }
                    false
                } else {
                    true
                }
            });

            if !group.is_empty() {
                groups.push(self.create_group(group, is_tree));
            }
        }

        groups
    }

    fn create_group(&self, symbols: Vec<SymbolId>, is_tree: bool) -> SymbolGroup {
        if is_tree {
            let mut referenced_symbols = IndexMap::new();

            for symbol in symbols {
                referenced_symbols.insert(symbol, self.graph.references(symbol));
            }

            SymbolGroup::Tree(Environment::tree(referenced_symbols))
        } else {
            SymbolGroup::Sequential(symbols)
        }
    }

    fn apply_group(mut env: Environment, group: &SymbolGroup, include_rest: bool) -> Environment {
        match group {
            SymbolGroup::Sequential(symbols) => {
                for (i, &symbol) in symbols.iter().rev().enumerate() {
                    if i == 0 && !include_rest {
                        env = Environment::Leaf(symbol);
                    } else {
                        env = Environment::Pair(Box::new(Environment::Leaf(symbol)), Box::new(env));
                    }
                }

                env
            }
            SymbolGroup::Tree(tree) => {
                if include_rest {
                    Environment::Pair(Box::new(tree.clone()), Box::new(env))
                } else {
                    tree.clone()
                }
            }
        }
    }

    fn should_inline(&self, symbol: SymbolId) -> bool {
        if self
            .inline_symbols
            .iter()
            .any(|symbols| symbols.contains_key(&symbol))
        {
            return true;
        }

        let references = self.graph.references(symbol);

        match self.db.symbol(symbol) {
            Symbol::Unresolved | Symbol::Module(_) | Symbol::Parameter(_) | Symbol::Builtin(_) => {
                false
            }
            Symbol::Function(function) => {
                if self.graph.dependencies(symbol, false).contains(&symbol) {
                    return false;
                }

                if function.kind == FunctionKind::Inline {
                    return true;
                }

                for (_, parameter) in &function.parameters {
                    if self.graph.references(*parameter) > 1 {
                        return false;
                    }
                }

                references <= 1 && self.options.auto_inline
            }
            Symbol::Constant(constant) => {
                if constant.inline {
                    return true;
                }

                references <= 1 && self.options.auto_inline
            }
            Symbol::Binding(binding) => {
                if binding.inline {
                    return true;
                }

                references <= 1 && self.options.auto_inline
            }
        }
    }
}

fn atom(loc: Srcloc, bytes: Vec<u8>) -> RueSExp {
    RueSExp::new(SExp::Atom(loc, bytes))
}

fn nil(loc: Srcloc) -> RueSExp {
    RueSExp::new(SExp::Nil(loc))
}

fn cons(loc: Srcloc, left: RueSExp, right: RueSExp) -> RueSExp {
    RueSExp::new(SExp::Cons(loc, left.0, right.0))
}

fn pair(left: RueSExp, right: RueSExp) -> RueSExp {
    let loc = left.loc().ext(&right.loc());
    cons(loc, left, right)
}

fn list(loc: Srcloc, items: Vec<RueSExp>) -> RueSExp {
    let mut result = nil(loc);
    for item in items.into_iter().rev() {
        result = pair(item, result);
    }
    result
}

fn op_list(loc: Srcloc, op: ClvmOp, args: Vec<RueSExp>) -> RueSExp {
    let mut items = Vec::with_capacity(args.len() + 1);
    items.push(atom(loc.clone(), op.to_atom()));
    items.extend(args);
    list(loc, items)
}

fn quote(loc: Srcloc, value: RueSExp) -> RueSExp {
    cons(loc.clone(), atom(loc, ClvmOp::Quote.to_atom()), value)
}

fn codegen_debug(arena: &Arena<Lir>, locs: &HashMap<LirId, Srcloc>, lir: LirId) -> RueSExp {
    let loc = locs
        .get(&lir)
        .cloned()
        .unwrap_or_else(|| Srcloc::start("*rue*"));
    match &arena[lir] {
        Lir::Atom(bytes) => {
            if bytes.is_empty() {
                nil(loc)
            } else {
                quote(loc.clone(), atom(loc, bytes.clone()))
            }
        }
        Lir::Quote(arg) => {
            let arg = codegen_debug(arena, locs, *arg);
            quote(loc, arg)
        }
        Lir::Path(path) => {
            let mut bytes = bigint_atom((*path).into());
            while !bytes.is_empty() && bytes[0] == 0 {
                bytes = bytes[1..].to_vec();
            }
            atom(loc, bytes)
        }
        Lir::Run(callee, env) => op_list(
            loc,
            ClvmOp::Apply,
            vec![
                codegen_debug(arena, locs, *callee),
                codegen_debug(arena, locs, *env),
            ],
        ),
        Lir::Closure(function, captures, has_parameters) => {
            let function = codegen_debug(arena, locs, *function);
            let mut args = if *has_parameters {
                quote(loc.clone(), atom(loc.clone(), vec![1]))
            } else {
                nil(loc.clone())
            };

            for capture in captures.iter().rev() {
                let capture = codegen_debug(arena, locs, *capture);
                args = op_list(
                    loc.clone(),
                    ClvmOp::Cons,
                    vec![
                        quote(loc.clone(), atom(loc.clone(), ClvmOp::Cons.to_atom())),
                        op_list(
                            loc.clone(),
                            ClvmOp::Cons,
                            vec![
                                op_list(
                                    loc.clone(),
                                    ClvmOp::Cons,
                                    vec![
                                        quote(
                                            loc.clone(),
                                            atom(loc.clone(), ClvmOp::Quote.to_atom()),
                                        ),
                                        capture,
                                    ],
                                ),
                                op_list(loc.clone(), ClvmOp::Cons, vec![args, nil(loc.clone())]),
                            ],
                        ),
                    ],
                );
            }

            op_list(
                loc.clone(),
                ClvmOp::Cons,
                vec![
                    quote(loc.clone(), atom(loc.clone(), ClvmOp::Apply.to_atom())),
                    op_list(
                        loc.clone(),
                        ClvmOp::Cons,
                        vec![
                            op_list(
                                loc.clone(),
                                ClvmOp::Cons,
                                vec![
                                    quote(loc.clone(), atom(loc.clone(), ClvmOp::Quote.to_atom())),
                                    function,
                                ],
                            ),
                            op_list(loc.clone(), ClvmOp::Cons, vec![args, nil(loc.clone())]),
                        ],
                    ),
                ],
            )
        }
        Lir::First(arg) => op_list(loc, ClvmOp::First, vec![codegen_debug(arena, locs, *arg)]),
        Lir::Rest(arg) => op_list(loc, ClvmOp::Rest, vec![codegen_debug(arena, locs, *arg)]),
        Lir::Cons(first, rest) => op_list(
            loc,
            ClvmOp::Cons,
            vec![
                codegen_debug(arena, locs, *first),
                codegen_debug(arena, locs, *rest),
            ],
        ),
        Lir::Listp(arg, _) => op_list(loc, ClvmOp::Listp, vec![codegen_debug(arena, locs, *arg)]),
        Lir::Add(args) => op_list_args(arena, locs, loc, ClvmOp::Add, args),
        Lir::Sub(args) => op_list_args(arena, locs, loc, ClvmOp::Sub, args),
        Lir::Mul(args) => op_list_args(arena, locs, loc, ClvmOp::Mul, args),
        Lir::Div(first, second) => op_list_binary(arena, locs, loc, ClvmOp::Div, *first, *second),
        Lir::Divmod(first, second) => {
            op_list_binary(arena, locs, loc, ClvmOp::Divmod, *first, *second)
        }
        Lir::Mod(first, second) => op_list_binary(arena, locs, loc, ClvmOp::Mod, *first, *second),
        Lir::Modpow(a, b, c) => op_list(
            loc,
            ClvmOp::Modpow,
            vec![
                codegen_debug(arena, locs, *a),
                codegen_debug(arena, locs, *b),
                codegen_debug(arena, locs, *c),
            ],
        ),
        Lir::Eq(first, second) => op_list_binary(arena, locs, loc, ClvmOp::Eq, *first, *second),
        Lir::Gt(first, second) => op_list_binary(arena, locs, loc, ClvmOp::Gt, *first, *second),
        Lir::GtBytes(first, second) => {
            op_list_binary(arena, locs, loc, ClvmOp::GtBytes, *first, *second)
        }
        Lir::Not(arg) => op_list(loc, ClvmOp::Not, vec![codegen_debug(arena, locs, *arg)]),
        Lir::All(args) => op_list_args(arena, locs, loc, ClvmOp::All, args),
        Lir::Any(args) => op_list_args(arena, locs, loc, ClvmOp::Any, args),
        Lir::If(cond, then_lir, else_lir, inline) => {
            let cond = codegen_debug(arena, locs, *cond);
            let then_ptr = codegen_debug(arena, locs, *then_lir);
            let else_ptr = codegen_debug(arena, locs, *else_lir);
            if *inline {
                op_list(loc, ClvmOp::If, vec![cond, then_ptr, else_ptr])
            } else {
                op_list(
                    loc.clone(),
                    ClvmOp::Apply,
                    vec![
                        op_list(
                            loc.clone(),
                            ClvmOp::If,
                            vec![
                                cond,
                                quote(loc.clone(), then_ptr),
                                quote(loc.clone(), else_ptr),
                            ],
                        ),
                        atom(loc, vec![1]),
                    ],
                )
            }
        }
        Lir::Raise(args) => op_list_args(arena, locs, loc, ClvmOp::Raise, args),
        Lir::Concat(args) => op_list_args(arena, locs, loc, ClvmOp::Concat, args),
        Lir::Strlen(arg) => op_list(loc, ClvmOp::Strlen, vec![codegen_debug(arena, locs, *arg)]),
        Lir::Substr(arg, start, end) => {
            let mut args = vec![
                codegen_debug(arena, locs, *arg),
                codegen_debug(arena, locs, *start),
            ];
            if let Some(end) = end {
                args.push(codegen_debug(arena, locs, *end));
            }
            op_list(loc, ClvmOp::Substr, args)
        }
        Lir::Logand(args) => op_list_args(arena, locs, loc, ClvmOp::Logand, args),
        Lir::Logior(args) => op_list_args(arena, locs, loc, ClvmOp::Logior, args),
        Lir::Logxor(args) => op_list_args(arena, locs, loc, ClvmOp::Logxor, args),
        Lir::Lognot(arg) => op_list(loc, ClvmOp::Lognot, vec![codegen_debug(arena, locs, *arg)]),
        Lir::Ash(arg, shift) => op_list_binary(arena, locs, loc, ClvmOp::Ash, *arg, *shift),
        Lir::Lsh(arg, shift) => op_list_binary(arena, locs, loc, ClvmOp::Lsh, *arg, *shift),
        Lir::PubkeyForExp(arg) => op_list(
            loc,
            ClvmOp::PubkeyForExp,
            vec![codegen_debug(arena, locs, *arg)],
        ),
        Lir::G1Add(args) => op_list_args(arena, locs, loc, ClvmOp::G1Add, args),
        Lir::G1Subtract(args) => op_list_args(arena, locs, loc, ClvmOp::G1Subtract, args),
        Lir::G1Multiply(first, second) => {
            op_list_binary(arena, locs, loc, ClvmOp::G1Multiply, *first, *second)
        }
        Lir::G1Negate(arg) => op_list(
            loc,
            ClvmOp::G1Negate,
            vec![codegen_debug(arena, locs, *arg)],
        ),
        Lir::G1Map(value, dst) => {
            let mut args = vec![codegen_debug(arena, locs, *value)];
            if let Some(dst) = dst {
                args.push(codegen_debug(arena, locs, *dst));
            }
            op_list(loc, ClvmOp::G1Map, args)
        }
        Lir::G2Add(args) => op_list_args(arena, locs, loc, ClvmOp::G2Add, args),
        Lir::G2Subtract(args) => op_list_args(arena, locs, loc, ClvmOp::G2Subtract, args),
        Lir::G2Multiply(first, second) => {
            op_list_binary(arena, locs, loc, ClvmOp::G2Multiply, *first, *second)
        }
        Lir::G2Negate(arg) => op_list(
            loc,
            ClvmOp::G2Negate,
            vec![codegen_debug(arena, locs, *arg)],
        ),
        Lir::G2Map(value, dst) => {
            let mut args = vec![codegen_debug(arena, locs, *value)];
            if let Some(dst) = dst {
                args.push(codegen_debug(arena, locs, *dst));
            }
            op_list(loc, ClvmOp::G2Map, args)
        }
        Lir::BlsPairingIdentity(args) => {
            op_list_args(arena, locs, loc, ClvmOp::BlsPairingIdentity, args)
        }
        Lir::BlsVerify(arg, args) => {
            let mut items = vec![codegen_debug(arena, locs, *arg)];
            items.extend(args.iter().map(|arg| codegen_debug(arena, locs, *arg)));
            op_list(loc, ClvmOp::BlsVerify, items)
        }
        Lir::Sha256(args) | Lir::Sha256Inline(args) => {
            op_list_args(arena, locs, loc, ClvmOp::Sha256, args)
        }
        Lir::Keccak256(args) | Lir::Keccak256Inline(args) => {
            op_list_args(arena, locs, loc, ClvmOp::Keccak256, args)
        }
        Lir::CoinId(parent, puzzle, amount) => op_list(
            loc,
            ClvmOp::CoinId,
            vec![
                codegen_debug(arena, locs, *parent),
                codegen_debug(arena, locs, *puzzle),
                codegen_debug(arena, locs, *amount),
            ],
        ),
        Lir::K1Verify(pubkey, message, signature) => op_list(
            loc,
            ClvmOp::Secp256K1Verify,
            vec![
                codegen_debug(arena, locs, *pubkey),
                codegen_debug(arena, locs, *message),
                codegen_debug(arena, locs, *signature),
            ],
        ),
        Lir::R1Verify(pubkey, message, signature) => op_list(
            loc,
            ClvmOp::Secp256R1Verify,
            vec![
                codegen_debug(arena, locs, *pubkey),
                codegen_debug(arena, locs, *message),
                codegen_debug(arena, locs, *signature),
            ],
        ),
        Lir::Op(op, args) => {
            let args = codegen_debug(arena, locs, *args);
            op_list(
                loc.clone(),
                ClvmOp::Apply,
                vec![
                    op_list(
                        loc.clone(),
                        ClvmOp::Cons,
                        vec![
                            op_list(
                                loc.clone(),
                                ClvmOp::Cons,
                                vec![
                                    quote(loc.clone(), atom(loc.clone(), op.to_atom())),
                                    nil(loc.clone()),
                                ],
                            ),
                            args,
                        ],
                    ),
                    nil(loc),
                ],
            )
        }
        Lir::DebugPrint(srcloc, value) => op_list(
            loc.clone(),
            ClvmOp::DebugPrint,
            vec![
                quote(loc.clone(), atom(loc.clone(), srcloc.as_bytes().to_vec())),
                codegen_debug(arena, locs, *value),
            ],
        ),
    }
}

fn op_list_binary(
    arena: &Arena<Lir>,
    locs: &HashMap<LirId, Srcloc>,
    loc: Srcloc,
    op: ClvmOp,
    first: LirId,
    second: LirId,
) -> RueSExp {
    op_list(
        loc,
        op,
        vec![
            codegen_debug(arena, locs, first),
            codegen_debug(arena, locs, second),
        ],
    )
}

fn op_list_args(
    arena: &Arena<Lir>,
    locs: &HashMap<LirId, Srcloc>,
    loc: Srcloc,
    op: ClvmOp,
    args: &[LirId],
) -> RueSExp {
    op_list(
        loc,
        op,
        args.iter()
            .map(|arg| codegen_debug(arena, locs, *arg))
            .collect(),
    )
}

fn rue_srcloc_to_chialisp(loc: &SrcLoc) -> Srcloc {
    let start = loc.start();
    let end = loc.end();
    let file = loc.source.kind.display(Path::new("."));
    let mut out = Srcloc::new(Rc::new(file), start.line + 1, start.col + 1);
    out.until = Some(crate::compiler::srcloc::Until {
        line: end.line + 1,
        col: end.col + 1,
    });
    out
}

fn source_to_start_loc(source: &Source) -> Srcloc {
    Srcloc::start(&source.kind.display(Path::new(".")))
}

fn syntax_item_loc(
    sources: &HashMap<SourceKind, Source>,
    source_kind: &SourceKind,
    span: rowan::TextRange,
) -> Option<Srcloc> {
    let source = sources.get(source_kind)?;
    let start = rue_diagnostic::LineCol::new(&source.text, span.start().into());
    let end = rue_diagnostic::LineCol::new(&source.text, span.end().into());
    let mut loc = Srcloc::new(
        Rc::new(source.kind.display(Path::new("."))),
        start.line + 1,
        start.col + 1,
    );
    loc.until = Some(crate::compiler::srcloc::Until {
        line: end.line + 1,
        col: end.col + 1,
    });
    Some(loc)
}

fn build_symbol_locs(ctx: &Compiler, tree: &FileTree) -> HashMap<SymbolId, Srcloc> {
    let sources: HashMap<SourceKind, Source> = tree
        .all_files()
        .into_iter()
        .map(|file| (file.source.kind.clone(), file.source.clone()))
        .collect();
    let mut result = HashMap::new();

    for item in ctx.syntax_map().items() {
        if let SyntaxItemKind::SymbolDeclaration(symbol) = item.kind {
            if let Some(loc) = syntax_item_loc(&sources, &item.source_kind, item.span) {
                result.insert(symbol, loc);
            }
        }
    }

    result
}

fn parameter_expression(names: &[String]) -> String {
    if std::env::var_os("ARMTX_RUE_FORCE_ENV_ARGS").is_some() {
        return "ENV".to_string();
    }
    match names {
        [] => "()".to_string(),
        [name] => name.clone(),
        _ => format!("({})", names.join(" ")),
    }
}

fn add_function_symbol_metadata(
    symbol_table: &mut HashMap<String, String>,
    hash: String,
    name: &str,
    function: &FunctionSymbol,
) {
    symbol_table.insert(hash.clone(), name.to_string());
    let arguments = function.parameters.keys().cloned().collect::<Vec<_>>();
    let mut key = debug_decode_string(format!("{hash}_arguments").as_bytes());
    symbol_table.insert(key, parameter_expression(&arguments));
    key = debug_decode_string(format!("{hash}_left_env").as_bytes());
    symbol_table.insert(key, "0".to_string());
}

fn compile_main(
    ctx: &mut Compiler,
    tree: &FileTree,
    main: SymbolId,
    base_path: PathBuf,
    symbol_locs: &HashMap<SymbolId, Srcloc>,
) -> Result<(RueSExp, HashMap<String, String>, HashMap<String, Srcloc>), String> {
    let options = *ctx.options();
    let graph = DependencyGraph::build(ctx, main, options);
    let mut arena = Arena::new();
    let mut lir_locs = HashMap::new();
    let mut function_body_lirs = HashMap::new();
    let fallback_loc = tree
        .all_files()
        .first()
        .map(|file| source_to_start_loc(&file.source))
        .unwrap_or_else(|| Srcloc::start("*rue*"));
    let lir = {
        let mut lowerer = DebugLowerer::new(
            ctx,
            &mut arena,
            &graph,
            &mut lir_locs,
            &mut function_body_lirs,
            options,
            main,
            base_path,
            symbol_locs,
            fallback_loc,
        );
        lowerer.lower_symbol_value(&Environment::default(), main)
    };

    let compiled = codegen_debug(&arena, &lir_locs, lir);
    let mut symbol_table = HashMap::new();
    let mut program_locations = HashMap::new();

    let relevant_declarations = ctx.relevant_declarations().collect::<Vec<_>>();
    for item in relevant_declarations {
        let rue_hir::Declaration::Symbol(symbol) = item else {
            continue;
        };
        let Symbol::Function(function) = ctx.symbol(symbol).clone() else {
            continue;
        };
        let Some(name) = function.name.as_ref().map(|name| name.text().to_string()) else {
            continue;
        };

        if let Some(loc) = symbol_locs.get(&symbol).cloned() {
            program_locations.insert(name.clone(), loc);
        }

        let Some(body_lir) = function_body_lirs.get(&symbol).copied() else {
            continue;
        };
        let function_sexp = codegen_debug(&arena, &lir_locs, body_lir);
        let function_hash = hex::encode(crate::compiler::debug::debug_sha256tree(function_sexp));
        add_function_symbol_metadata(&mut symbol_table, function_hash, &name, &function);
    }

    Ok((compiled, symbol_table, program_locations))
}

pub fn compile_rue_to_arm_elf(args: &Args) -> Result<ArmElfCompileOutput, String> {
    let mut allocator = Allocator::new();
    let search_path = Path::new(&args.filename)
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize {}: {e:?}", args.filename))?;
    let project = find_project(&search_path, true)
        .map_err(|e| format!("failed to find rue project: {e:?}"))?
        .ok_or_else(|| format!("no rue project found for {}", args.filename))?;

    let file_kind = Some(normalize_path(Path::new(&args.filename)).map_err(|e| format!("{e:?}"))?);
    let main_kind = file_kind.clone().or_else(|| {
        let main = project.entrypoint.join("main.rue");
        if main.exists() {
            normalize_path(&main).ok()
        } else {
            None
        }
    });

    let mut ctx = Compiler::new(project.options);
    let tree = FileTree::compile_path(&mut ctx, &project.entrypoint, &mut HashMap::new())
        .map_err(|e| format!("failed to compile rue: {e:?}"))?;
    let base_path = if project.entrypoint.is_file() {
        project
            .entrypoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()
            .map_err(|e| format!("failed to canonicalize rue base path: {e:?}"))?
    } else {
        project
            .entrypoint
            .canonicalize()
            .map_err(|e| format!("failed to canonicalize rue entrypoint: {e:?}"))?
    };

    let mut codegen_allowed = true;
    for diagnostic in ctx.take_diagnostics() {
        let message = diagnostic.message(&base_path);
        if diagnostic.kind.severity() == DiagnosticSeverity::Error {
            codegen_allowed = false;
            eprintln!("Error: {message}");
        } else {
            eprintln!("Warning: {message}");
        }
    }
    if !codegen_allowed {
        return Err("rue compilation failed".to_string());
    }

    let main_kind = main_kind.ok_or_else(|| "no rue main function found".to_string())?;
    let file = tree
        .find(&main_kind)
        .ok_or_else(|| format!("source not found in rue compilation unit: {main_kind:?}"))?;
    let scope = ctx.module(file.module).scope;
    let main = ctx
        .scope(scope)
        .symbol("main")
        .ok_or_else(|| "no `main` function found in rue file".to_string())?;
    let symbol_locs = build_symbol_locs(&ctx, &tree);
    let (compiled, symbol_table, program_locations) =
        compile_main(&mut ctx, &tree, main, base_path, &symbol_locs)?;

    let env_node =
        assemble(&mut allocator, &args.env).map_err(|e| format!("failed to read env: {e:?}"))?;
    let env = convert_from_clvm_rs(&mut allocator, Srcloc::start("*env*"), env_node)
        .map_err(|e| format!("failed to convert env program: {e:?}"))?;
    let symbols = Rc::new(symbol_table.clone());

    let program = Program::new(
        program_locations,
        &args.filename,
        &args.output,
        compiled,
        RueSExp(env),
        TARGET_ADDR,
        symbols.clone(),
    )?;

    let ElfObject {
        object_file,
        synthetic_source,
    } = program
        .to_elf(&args.output)
        .map_err(|e| format!("failed to create elf output: {e:?}"))?;

    Ok(ArmElfCompileOutput {
        object_file,
        synthetic_source,
        symbol_table,
    })
}
