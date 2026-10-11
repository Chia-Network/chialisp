//! Runtime recognition of the established Chialisp and Rue print encodings.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt;

use clvm_rs::allocator::{Allocator, NodePtr, SExp};
use clvm_rs::chia_dialect::{ChiaDialect, ClvmFlags};
use clvm_rs::cost::Cost;
use clvm_rs::dialect::{Dialect, OperatorSet};
use clvm_rs::error::EvalErr;
use clvm_rs::reduction::{Reduction, Response};
use clvm_rs::run_program::run_program;

use crate::classic::clvm_tools::binutils::disassemble;

pub const MAX_RUNTIME_PRINT_RECORDS: usize = 256;
pub const MAX_RUNTIME_PRINT_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimePrintKind {
    Chialisp,
    Rue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePrintRecord {
    pub kind: RuntimePrintKind,
    pub source: Option<String>,
    pub value: String,
}

impl RuntimePrintRecord {
    fn size(&self) -> usize {
        self.source.as_ref().map_or(0, String::len) + self.value.len()
    }
}

impl fmt::Display for RuntimePrintRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.source {
            Some(source) => write!(formatter, "{source}: {}", self.value),
            None => formatter.write_str(&self.value),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimePrintOutput {
    pub records: Vec<RuntimePrintRecord>,
    pub dropped: usize,
}

#[derive(Debug, Default)]
struct RuntimePrintState {
    records: VecDeque<RuntimePrintRecord>,
    bytes: usize,
    dropped: usize,
}

#[derive(Debug, Default)]
pub struct RuntimePrintCollector {
    state: RefCell<RuntimePrintState>,
}

impl RuntimePrintCollector {
    fn push(&self, record: RuntimePrintRecord) {
        let size = record.size();
        let mut state = self.state.borrow_mut();

        if size > MAX_RUNTIME_PRINT_BYTES {
            state.dropped += 1;
            return;
        }

        while state.records.len() >= MAX_RUNTIME_PRINT_RECORDS
            || state.bytes + size > MAX_RUNTIME_PRINT_BYTES
        {
            let Some(retired) = state.records.pop_front() else {
                break;
            };
            state.bytes -= retired.size();
            state.dropped += 1;
        }

        state.bytes += size;
        state.records.push_back(record);
    }

    pub fn take(&self) -> RuntimePrintOutput {
        let mut state = self.state.borrow_mut();
        let records = state.records.drain(..).collect();
        let dropped = state.dropped;
        state.bytes = 0;
        state.dropped = 0;
        RuntimePrintOutput { records, dropped }
    }
}

fn pair(allocator: &Allocator, node: NodePtr) -> Result<(NodePtr, NodePtr), EvalErr> {
    match allocator.sexp(node) {
        SExp::Pair(first, rest) => Ok((first, rest)),
        SExp::Atom => Err(EvalErr::InvalidNilTerminator(node)),
    }
}

pub(crate) fn detect_runtime_print(
    collector: &RuntimePrintCollector,
    allocator: &mut Allocator,
    op: NodePtr,
    args: NodePtr,
) -> Result<Option<Reduction>, EvalErr> {
    if !matches!(allocator.sexp(op), SExp::Atom) {
        return Ok(None);
    }

    let op_atom = allocator.atom(op);
    if op_atom.as_ref() == b"debug_print" {
        let (source, rest) = pair(allocator, args)?;
        let (value, _) = pair(allocator, rest)?;
        if !matches!(allocator.sexp(source), SExp::Atom) {
            return Err(EvalErr::InternalError(
                source,
                "debug_print source location must be an atom".to_string(),
            ));
        }
        let source = std::str::from_utf8(allocator.atom(source).as_ref())
            .map_err(|_| {
                EvalErr::InternalError(
                    source,
                    "debug_print source location must be UTF-8".to_string(),
                )
            })?
            .to_string();
        collector.push(RuntimePrintRecord {
            kind: RuntimePrintKind::Rue,
            source: Some(source),
            value: disassemble(allocator, value, None),
        });
        return Ok(Some(Reduction(0, NodePtr::NIL)));
    }

    if allocator.small_number(op) != Some(34) {
        return Ok(None);
    }
    let Ok((marker, values)) = pair(allocator, args) else {
        return Ok(None);
    };
    if !matches!(allocator.sexp(marker), SExp::Atom)
        || allocator.atom(marker).as_ref() != b"$print$"
    {
        return Ok(None);
    }

    collector.push(RuntimePrintRecord {
        kind: RuntimePrintKind::Chialisp,
        source: None,
        value: disassemble(allocator, values, None),
    });
    Ok(None)
}

#[derive(Debug)]
pub struct RuntimePrintDialect {
    flags: ClvmFlags,
    collector: RuntimePrintCollector,
}

impl RuntimePrintDialect {
    pub fn new(flags: ClvmFlags) -> Self {
        Self {
            flags,
            collector: RuntimePrintCollector::default(),
        }
    }

    pub fn take_prints(&self) -> RuntimePrintOutput {
        self.collector.take()
    }

    fn inner(&self) -> ChiaDialect {
        ChiaDialect::new(self.flags)
    }
}

impl Dialect for RuntimePrintDialect {
    fn op(
        &self,
        allocator: &mut Allocator,
        op: NodePtr,
        args: NodePtr,
        max_cost: Cost,
        extension: OperatorSet,
    ) -> Response {
        if let Some(reduction) = detect_runtime_print(&self.collector, allocator, op, args)? {
            return Ok(reduction);
        }
        self.inner().op(allocator, op, args, max_cost, extension)
    }

    fn quote_kw(&self) -> u32 {
        self.inner().quote_kw()
    }

    fn apply_kw(&self) -> u32 {
        self.inner().apply_kw()
    }

    fn softfork_kw(&self) -> u32 {
        self.inner().softfork_kw()
    }

    fn softfork_extension(&self, ext: u32) -> OperatorSet {
        self.inner().softfork_extension(ext)
    }

    fn allow_unknown_ops(&self) -> bool {
        self.inner().allow_unknown_ops()
    }

    fn flags(&self) -> ClvmFlags {
        self.flags
    }

    fn gc_candidate(&self, allocator: &Allocator, node: NodePtr) -> bool {
        self.inner().gc_candidate(allocator, node)
    }
}

pub struct RuntimePrintRun {
    pub result: Response,
    pub prints: RuntimePrintOutput,
}

pub fn run_program_with_runtime_prints(
    allocator: &mut Allocator,
    flags: ClvmFlags,
    program: NodePtr,
    args: NodePtr,
    max_cost: Cost,
) -> RuntimePrintRun {
    let dialect = RuntimePrintDialect::new(flags);
    let result = run_program(allocator, &dialect, program, args, max_cost);
    RuntimePrintRun {
        result,
        prints: dialect.take_prints(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classic::clvm_tools::binutils::assemble;

    fn run(source: &str) -> (Allocator, RuntimePrintRun) {
        let mut allocator = Allocator::new();
        let program = assemble(&mut allocator, source).expect("assemble test program");
        let run = run_program_with_runtime_prints(
            &mut allocator,
            ClvmFlags::NO_UNKNOWN_OPS,
            program,
            NodePtr::NIL,
            0,
        );
        (allocator, run)
    }

    #[test]
    fn recognizes_chialisp_print_without_changing_result() {
        let source = r#"(all (q . "$print$") (q . "label") (q . 42))"#;
        let (allocator, run) = run(source);
        let reduction = run.result.expect("print run succeeds");
        assert_eq!(disassemble(&allocator, reduction.1, None), "1");
        assert_eq!(
            run.prints.records,
            vec![RuntimePrintRecord {
                kind: RuntimePrintKind::Chialisp,
                source: None,
                value: "(\"label\" 42)".to_string(),
            }]
        );

        let mut plain_allocator = Allocator::new();
        let plain_program = assemble(&mut plain_allocator, source).expect("assemble plain program");
        let plain = run_program(
            &mut plain_allocator,
            &ChiaDialect::new(ClvmFlags::NO_UNKNOWN_OPS),
            plain_program,
            NodePtr::NIL,
            0,
        )
        .expect("plain run succeeds");
        assert_eq!(reduction.0, plain.0);
        assert_eq!(
            disassemble(&allocator, reduction.1, None),
            disassemble(&plain_allocator, plain.1, None)
        );
    }

    #[test]
    fn recognizes_rue_print_and_preserves_order() {
        let (_, run) = run(r#"(c
                (all (q . "$print$") (q . "first") (q . 1))
                (c
                    ("debug_print" (q . "game.rue:2:3") (q . ("second" 2)))
                    (q . ())
                )
            )"#);
        run.result.expect("mixed print run succeeds");
        assert_eq!(
            run.prints.records,
            vec![
                RuntimePrintRecord {
                    kind: RuntimePrintKind::Rue,
                    source: Some("game.rue:2:3".to_string()),
                    value: "(\"second\" 2)".to_string(),
                },
                RuntimePrintRecord {
                    kind: RuntimePrintKind::Chialisp,
                    source: None,
                    value: "(\"first\" 1)".to_string(),
                },
            ]
        );
    }

    #[test]
    fn retains_prints_before_an_error() {
        let (_, run) = run(r#"(c
                ("not_an_operator")
                ("debug_print" (q . "game.rue:4:5") (q . "before error"))
            )"#);
        assert!(run.result.is_err());
        assert_eq!(run.prints.records.len(), 1);
        assert_eq!(run.prints.records[0].value, "\"before error\"");
    }

    #[test]
    fn rejects_malformed_rue_print() {
        let (_, run) = run(r#"("debug_print" (q . "game.rue:1:1"))"#);
        assert!(matches!(run.result, Err(EvalErr::InvalidNilTerminator(_))));
        assert!(run.prints.records.is_empty());
    }

    #[test]
    fn requires_exact_chialisp_marker() {
        let (_, run) = run(r#"(all (q . "$print") (q . "ignored") (q . 1))"#);
        run.result.expect("ordinary all succeeds");
        assert!(run.prints.records.is_empty());
    }

    #[test]
    fn collector_keeps_newest_records_within_bounds() {
        let collector = RuntimePrintCollector::default();
        for index in 0..=MAX_RUNTIME_PRINT_RECORDS {
            collector.push(RuntimePrintRecord {
                kind: RuntimePrintKind::Chialisp,
                source: None,
                value: index.to_string(),
            });
        }
        let output = collector.take();
        assert_eq!(output.records.len(), MAX_RUNTIME_PRINT_RECORDS);
        assert_eq!(output.records[0].value, "1");
        assert_eq!(output.dropped, 1);
        assert_eq!(collector.take(), RuntimePrintOutput::default());
    }
}
