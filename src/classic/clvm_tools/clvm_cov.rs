// clvm_cov -- source-level coverage for Chialisp/CLVM.
//
// Given a corpus of CLVM invocations `(program, args)` plus the `.clsp` source each
// program was compiled from, report which SOURCE lines were executed vs. which are cold
// (never run).  This is assembly on top of the existing `cldb` machinery:
//
//   * We compile each distinct source in-process (the same pipeline `cldb`/`run` use via
//     `RunAndCompileInputData`) and build a pure `sha256tree -> srcloc` table with
//     `build_symbol_table_mut`.  That table's values are the REACHABLE set (the
//     denominator) and simultaneously the map used to re-attach source locations to the
//     compiled program.
//   * We load each corpus `program` hex through `hex_to_modern_sexp`, which looks up every
//     node's `sha256tree` in that table and stamps the matching source `Srcloc` back onto
//     it (see `src/compiler/cldb.rs` `hex_to_modern_sexp_inner`).
//   * We step the program with a `CldbRun` (exactly as `cldb` does) and union every executed
//     node's `.loc()` into the EXECUTED set.
//
// `covered = reachable ∩ executed`, intersected at (file, line) granularity -- coverage of the
// COMPILED program mapped back to source, since the compiler inlines and optimizes.

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use clvmr::Allocator;

use serde::Deserialize;

use crate::classic::clvm::__type_compatibility__::{sha256, Bytes, BytesFromType};
use crate::classic::clvm_tools::clvmc::compile_clvm_text_maybe_opt;
use crate::classic::clvm_tools::comp_input::{CompiledModernProgram, RunAndCompileInputData};
use crate::classic::clvm_tools::sha256tree::sha256tree;
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::classic::platform::argparse::ArgumentValue;
use crate::compiler::cldb::{hex_to_modern_sexp, CldbNoOverride, CldbRun, CldbRunEnv};
use crate::compiler::clvm::start_step;
use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::debug::build_symbol_table_mut;
use crate::compiler::prims;
use crate::compiler::sexp::SExp;
use crate::compiler::srcloc::Srcloc;
use crate::util::u8_from_number;

/// Hard bound on the number of steps we will run a single corpus program.  An unbounded
/// step loop is a hang waiting to happen; a real on-chain puzzle is cost-bounded far below
/// this, so hitting it means the record is malformed.
const MAX_STEPS: u64 = 50_000_000;

fn default_layer() -> String {
    "unknown".to_string()
}

/// One line of the corpus file (JSONL, one object per line).
///
/// ```json
/// {"program": "<hex>", "args": "<hex>", "source": "path/to/puzzle.clsp", "layer": "unit"}
/// ```
///
/// * `program` -- compiled-CLVM program, hex.
/// * `args`    -- its environment/solution, hex.
/// * `source`  -- OPTIONAL `.clsp` the program was compiled from. Relative paths resolve
///   against `--source`; absolute paths are used as-is. When present, only that source is
///   compiled and the program is attributed to it. When absent, the program is attributed
///   against a global `sha256tree -> source` table built from every `.clsp` under
///   `--source`/`--include` (the shape runtime captures have, since a run only sees
///   `(program, args)`).
/// * `layer`   -- OPTIONAL free-form label for the mechanism that produced the run. Coverage
///   is reported as a union and per-layer. Default `unknown`.
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusRecord {
    pub program: String,
    pub args: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default = "default_layer")]
    pub layer: String,
}

/// Parsed CLI arguments.
#[derive(Debug, Clone)]
struct CovArgs {
    corpus: String,
    source_root: String,
    includes: Vec<String>,
    out: Option<String>,
    summary: bool,
    /// Force the classic optimizer on when recompiling every source.  `None` = auto:
    /// honor the dialect the source sigil selects (a `*standard-cl-2N*` sigil with N > 22
    /// already forces optimization, matching production; comp_input.rs:189-193).
    force_optimize: bool,
}

const USAGE: &str = "\
clvm_cov -- source-level coverage for Chialisp/CLVM

USAGE:
    clvm_cov --corpus <corpus.jsonl> --source <clsp_dir> [--include <dir>]... \
[--out <coverage.lcov>] [--summary] [--optimize]

OPTIONS:
    --corpus  <file>   JSONL corpus, one object per line:
                       {\"program\":\"<hex>\",\"args\":\"<hex>\",\"source\":\"<path.clsp>\",\
\"layer\":\"unit\"}
                       program+args are compiled-CLVM hex.  source is OPTIONAL: when given it
                       is the .clsp the program was compiled from (relative paths resolve
                       against --source); when OMITTED (the shape real runtime captures have),
                       the program is attributed automatically against a global table built by
                       compiling every .clsp under --source/--include.
                       layer is optional (default \"unknown\").
    --source  <dir>    Base directory for resolving record `source` paths; also added to the
                       include search path.  Every .clsp found under it (recursively) is
                       compiled into the global attribution table used for source-less records.
    --include <dir>    Extra include search directory (repeatable).
    --out     <file>   Write union LCOV here, plus one <stem>.<layer>.<ext> per layer.
    --summary          Print per-file union% over the MEASURABLE lines (TOTAL), the count
                       of lines the matched corpus artifacts cannot express (EXCL -- e.g. a
                       classic stage_2 producer only shares some subtrees with the modern
                       srcloc-bearing recompile), per-layer totals, and the cold worklist.
    --optimize         Force the classic optimizer on when recompiling.  REFUSED when it
                       would discard source locations (always for a classic no-sigil source,
                       and whenever the table collapses): a locationless compile scores a
                       vacuous 100%.  The source sigil already selects the production compile
                       (a *standard-cl-2N* (N>22) source compiles optimized, with locations).
    -h, --help         Show this help.

A COLD line is a reachable source line never hit by any record.  Coverage is measured at
(file, line) granularity of the COMPILED program mapped back to source.

GRANULARITY / CONTRACT: clvm_cov recompiles each source through the SAME pipeline run/cldb use
and honors the source's dialect sigil, so for *standard-cl-25* (production) it compiles WITH
the optimizer -- byte-identical to the captured on-chain hex, so node treehashes match and
decoration is clean.  Cold-branch fidelity through that optimizer:
  * DEFUN-structured branches (the normal style, `(if X (fn-a ...) (fn-b ...))`): the untaken
    function's BODY line is reported COLD -- line-accurate.  Measured on cl-25 + optimize.
  * INLINE arms folded into the call site (`defun-inline`, constant/expr arms): collapse to a
    shared location and cannot be told apart -- write branch bodies as (non-inline) defun
    calls for line fidelity.
  * CLASSIC (no-sigil) sources: a classic producer's stage_2 bytes (`run`, `cdv clsp build`)
    do not byte-match the modern srcloc-bearing recompile.  The classic roots are registered
    too, so a captured classic program still IDENTIFIES its source; attribution then flows
    through the subtrees the two artifacts share, and lines the captured bytes cannot express
    are EXCLUDED from the denominator (the EXCL column) instead of reported as cold.
  * A subtree occurring at more than one source location is ambiguous BY HASH and never
    credits a line (its enclosing unique ancestor does); an identical subexpression on two
    lines cannot mark the untaken line executed.
  * A corpus `program` whose treehash differs from the recompiled source (compiled with
    different settings) triggers a one-line treehash-mismatch warning; mapping is best-effort.
    Pass --optimize (or add the matching sigil) so the recompile matches.";

fn parse_args(args: &[String]) -> Result<CovArgs, String> {
    let mut corpus = None;
    let mut source_root = None;
    let mut includes = Vec::new();
    let mut out = None;
    let mut summary = false;
    let mut force_optimize = false;

    let mut i = 1; // args[0] is the program name
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => return Err(USAGE.to_string()),
            "--summary" => summary = true,
            "--optimize" => force_optimize = true,
            "--corpus" | "--source" | "--include" | "--out" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| format!("{a} requires a value"))?
                    .clone();
                i += 1;
                match a {
                    "--corpus" => corpus = Some(v),
                    "--source" => source_root = Some(v),
                    "--include" => includes.push(v),
                    "--out" => out = Some(v),
                    _ => unreachable!(),
                }
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
        i += 1;
    }

    Ok(CovArgs {
        corpus: corpus.ok_or("--corpus is required")?,
        source_root: source_root.ok_or("--source is required")?,
        includes,
        out,
        summary,
        force_optimize,
    })
}

/// The reachable set for one compiled source: the union of every `Srcloc` (as its raw
/// string) attached to a node of the compiled program, plus the `sha256tree -> srcloc`
/// table used to re-decorate the corpus hex.
pub struct CompiledSource {
    /// `sha256tree_hex -> srcloc_string`, pure (built fresh, no name-symbol contamination).
    pub srcloc_syms: HashMap<String, String>,
    /// Every reachable srcloc string (real source files only).
    pub reachable: BTreeSet<String>,
    /// Subtree hashes occurring at MORE THAN ONE source location in this compile -- dropped
    /// from `srcloc_syms` as ambiguous.  Retained so the union/overlay machinery can keep a
    /// sibling source from claiming an occurrence that is ambiguous HERE.
    pub ambiguous: HashSet<String>,
    /// sha256tree of each compiled program root (one per export of a multi-export module) --
    /// used to detect a corpus `program` hex compiled differently (e.g. optimized), whose node
    /// hashes will not match our srcloc table, and to identify a source-less capture by root.
    pub root_hashes: Vec<String>,
}

/// Compile one `.clsp` source in-process and return its symbol/srcloc table.
///
/// Uses the same `RunAndCompileInputData` pipeline as `cldb`/`run` (dialect detection,
/// includes, optimization), then rebuilds a *pure* `sha256tree -> srcloc` map with
/// `build_symbol_table_mut` over the compiled output so we are not polluted by the
/// function-name entries `compile_file` also emits.
///
/// The dialect (and thus whether the optimizer runs) is auto-detected from the source sigil,
/// so a `*standard-cl-25*` production source compiles optimized and byte-matches captured
/// on-chain hex.  `force_optimize` additionally turns the optimizer on for sources that would
/// otherwise compile un-optimized (e.g. no sigil), to match a producer that optimizes those.
pub fn compile_source(
    source_path: &str,
    includes: &[String],
    force_optimize: bool,
) -> Result<CompiledSource, String> {
    let content = fs::read_to_string(source_path)
        .map_err(|e| format!("could not read source {source_path}: {e}"))?;

    let mut allocator = Allocator::new();

    let mut parsed_args: HashMap<String, ArgumentValue> = HashMap::new();
    parsed_args.insert(
        "path_or_code".to_string(),
        ArgumentValue::ArgString(Some(source_path.to_string()), content.clone()),
    );
    let include_vals: Vec<ArgumentValue> = includes
        .iter()
        .map(|d| ArgumentValue::ArgString(Some(d.clone()), d.clone()))
        .collect();
    parsed_args.insert("include".to_string(), ArgumentValue::ArgArray(include_vals));
    if force_optimize {
        // RunAndCompileInputData reads "optimize" as do_optimize, then ORs in stepping > 22.
        parsed_args.insert("optimize".to_string(), ArgumentValue::ArgBool(true));
    }

    let input = RunAndCompileInputData::new(&mut allocator, &parsed_args)
        .map_err(|e| format!("compile setup failed for {source_path}: {e}"))?;

    // --optimize on a classic (no-sigil) source runs the classic optimizer over the modern
    // compile, which rebuilds nodes without source locations: the srcloc table collapses to a
    // single location and every corpus scores a vacuous 100%.  Refuse rather than report a
    // perfect score that measures nothing.  (A *standard-cl-2N* sigil source optimizes with
    // locations intact and is unaffected.)
    if force_optimize && input.dialect.stepping.is_none() {
        return Err(format!(
            "--optimize with classic-dialect source {source_path} discards source locations \
             (the srcloc table collapses to a single location); refusing to report vacuous \
             coverage -- drop --optimize or add a *standard-cl-2N* sigil"
        ));
    }

    // We only want the srcloc table; the name-symbol map is a throwaway here.
    //
    // Use `compile_modern_programs`, NOT `compile_modern`: for a module-style source the latter
    // returns the export SUMMARY (a `(("name" . <hash>))` pair at a single source location), so
    // its srcloc table collapses to one reachable line and the captured on-chain program cannot
    // be attributed.  `compile_modern_programs` returns the ACTUAL compiled program(s) -- the
    // same bytes the `.hex` carries, with per-line source locations on their nodes -- so the
    // table is dense and the corpus program's node treehashes resolve.
    let mut name_syms: HashMap<String, String> = HashMap::new();
    let programs = input
        .compile_modern_programs(&mut allocator, &mut name_syms)
        .map_err(|e| format!("compile failed for {source_path}: {}: {}", e.0, e.1))?;

    // Pure sha256tree -> srcloc.to_string() over every subtree of every compiled program (a
    // multi-export module contributes one program per export; the tables union).  We record the
    // root of EVERY export so an explicit-`source` record for any export matches, and a source-less
    // capture of any export can be identified by root.
    // A subtree occurring at MORE THAN ONE source location is dropped as ambiguous: execution
    // of one occurrence cannot be told from another by hash, so crediting either line would be
    // a guess -- the decorator instead inherits the nearest uniquely-located ancestor, the same
    // policy the cross-source union applies.  (Previously one occurrence won arbitrarily,
    // crediting the WRONG line whenever a shared subexpression ran: a false positive on the
    // line that did not run and a false negative on the one that did.)
    let (srcloc_syms, ambiguous, mut root_hashes) = build_unambiguous_srcloc_table(&programs);

    // A classic (no-sigil) producer -- `run` / `cdv clsp build` -- compiles through the classic
    // stage_2 pipeline, whose output does NOT byte-match the modern compiler's srcloc-bearing
    // artifact for the same source.  Register the classic root(s) too (optimized and not), so a
    // captured classic program still IDENTIFIES this source (expected-root / by-root matching);
    // its nodes then attribute through whichever subtrees the two artifacts share.
    if input.dialect.stepping.is_none() {
        for h in classic_root_hashes(source_path, &content, includes) {
            if !root_hashes.contains(&h) {
                root_hashes.push(h);
            }
        }
    }

    let mut reachable = BTreeSet::new();
    for loc_str in srcloc_syms.values() {
        if parse_srcloc_string(loc_str).is_some() {
            reachable.insert(loc_str.clone());
        }
    }

    // The same guard, post-compile and dialect-independent: --optimize routes the output
    // through the CLASSIC finalizer, which rebuilds nodes without source locations for any
    // dialect.  A collapsed table means every corpus would score a vacuous 100%.
    if force_optimize && reachable.len() <= 1 {
        return Err(format!(
            "--optimize collapsed {source_path}'s srcloc table to {} location(s); refusing \
             to report vacuous coverage -- drop --optimize (the sigil already selects the \
             production compile)",
            reachable.len()
        ));
    }

    Ok(CompiledSource {
        srcloc_syms,
        ambiguous,
        reachable,
        root_hashes,
    })
}

/// A GLOBAL attribution table: the union of every discovered source's `sha256tree -> srcloc`
/// map, used to attribute a corpus `program` that carries NO `source` field to whichever
/// source file each of its nodes maps to.
///
/// The values of `srcloc_syms` embed the source-file path (a `Srcloc::to_string()` is
/// `file(line):col...`), so once a program is decorated against this union, `lines_by_file`
/// splits its executed nodes back out per source automatically -- no separate source->node
/// bookkeeping is needed.  `reachable` is the union denominator, identical to what the
/// per-source path accumulates for the same set of sources.
pub struct GlobalSourceTable {
    /// Union `sha256tree_hex -> srcloc_string`, keeping only hashes that are UNIQUE to one
    /// source.  A hash produced by two or more sources (e.g. a bare env-path atom, or a shared
    /// `sha256tree.clib` subtree) is AMBIGUOUS -- attributing it to any single source would be
    /// wrong -- so it is dropped.  During decoration (`hex_to_modern_sexp_inner`) a dropped
    /// node inherits its nearest enclosing table-resolved ancestor's srcloc, which is the
    /// unique subtree that actually pins the source: this makes the global path reproduce the
    /// per-source attribution instead of leaking a shared atom to an arbitrary file.
    pub srcloc_syms: HashMap<String, String>,
    /// Union of every compiled source's reachable srcloc strings (the denominator).
    pub reachable: BTreeSet<String>,
    /// Sources that compiled cleanly (diagnostic).
    pub compiled: Vec<String>,
    /// Sources that FAILED to compile, `(path, error)` -- warned-and-skipped, not fatal.
    pub skipped: Vec<(String, String)>,
    /// PER-SOURCE attribution tables, `(source_path, sha256tree -> srcloc)`, one per cleanly
    /// compiled source.  Unlike `srcloc_syms` (the union, which DROPS any subtree shared across
    /// sources as ambiguous), each entry keeps the source's OWN collision-free map, including
    /// subtrees it shares with sibling sources.  This lets a curried program re-identified with a
    /// specific source (via `by_root`) attribute a body that sibling templates share -- which the
    /// union drops as ambiguous but the per-source table retains.
    pub sources: Vec<(String, HashMap<String, String>, HashSet<String>)>,
    /// `compiled_program_root_hash -> index into `sources``.  A source's compiled-program root
    /// treehash is its identity: when a captured curried program is uncurried (see
    /// `identify_sources_by_uncurry`) to an inner puzzle whose root hash is a key here, that inner
    /// puzzle IS the source and decorates against its collision-free table.  First-wins on a shared
    /// root (identical siblings).
    pub by_root: HashMap<String, usize>,
}

/// Walk every subtree of a compiled `SExp`, calling `visit` with each subtree's canonical
/// treehash (hex) and the subtree, and return the root treehash.  The hash space is identical
/// to `build_symbol_table_mut`'s keys (sha256 of `2|left|right` for a pair, `1|atom-bytes`
/// for an atom), so hashes here are directly comparable against the attribution table,
/// `program_root_hash`, and the classic `sha256tree`.
fn walk_subtree_hashes(code: &SExp, visit: &mut dyn FnMut(&str, &SExp)) -> Bytes {
    match code {
        SExp::Cons(_, a, b) => {
            let left = walk_subtree_hashes(a.borrow(), visit);
            let right = walk_subtree_hashes(b.borrow(), visit);
            let treehash = sha256(
                Bytes::new(Some(BytesFromType::Raw(vec![2])))
                    .concat(&left)
                    .concat(&right),
            );
            visit(&treehash.hex(), code);
            treehash
        }
        SExp::Atom(_, a) => {
            let treehash = sha256(
                Bytes::new(Some(BytesFromType::Raw(vec![1])))
                    .concat(&Bytes::new(Some(BytesFromType::Raw(a.clone())))),
            );
            visit(&treehash.hex(), code);
            treehash
        }
        SExp::QuotedString(l, _, a) => {
            walk_subtree_hashes(&SExp::Atom(l.clone(), a.clone()), visit)
        }
        SExp::Integer(l, i) => {
            walk_subtree_hashes(&SExp::Atom(l.clone(), u8_from_number(i.clone())), visit)
        }
        SExp::Nil(l) => walk_subtree_hashes(&SExp::Atom(l.clone(), Vec::new()), visit),
    }
}

/// The pure `sha256tree -> srcloc` table over a compile's program(s), with any subtree whose
/// hash occurs at MORE THAN ONE source location DROPPED as ambiguous (see compile_source),
/// plus each program's root hash.
fn build_unambiguous_srcloc_table(
    programs: &[CompiledModernProgram],
) -> (HashMap<String, String>, HashSet<String>, Vec<String>) {
    let mut table: HashMap<String, String> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    let mut roots: Vec<String> = Vec::new();
    for (_shortname, prog) in programs {
        let prog_ref: &SExp = prog.borrow();
        let root = walk_subtree_hashes(prog_ref, &mut |h, s| {
            if ambiguous.contains(h) {
                return;
            }
            let loc = s.loc().to_string();
            match table.get(h) {
                Some(prev) if *prev != loc => {
                    table.remove(h);
                    ambiguous.insert(h.to_string());
                }
                Some(_) => {}
                None => {
                    table.insert(h.to_string(), loc);
                }
            }
        });
        let rh = root.hex();
        if !roots.contains(&rh) {
            roots.push(rh);
        }
    }
    (table, ambiguous, roots)
}

/// The root treehash(es) a CLASSIC producer would emit for this source: the stage_2 pipeline
/// (`run` / `cdv clsp build`), optimized and unoptimized.  Best-effort -- a source the classic
/// compiler rejects contributes no extra roots.
fn classic_root_hashes(source_path: &str, content: &str, includes: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for do_optimize in [false, true] {
        let mut allocator = Allocator::new();
        let mut sym: HashMap<String, String> = HashMap::new();
        let opts: Rc<dyn CompilerOpts> =
            Rc::new(DefaultCompilerOpts::new(source_path)).set_search_paths(includes);
        if let Ok(node) = compile_clvm_text_maybe_opt(
            &mut allocator,
            do_optimize,
            opts,
            &mut sym,
            content,
            source_path,
            false,
        ) {
            let h = sha256tree(&mut allocator, node).hex();
            if !out.contains(&h) {
                out.push(h);
            }
        }
    }
    out
}

/// Bound on how deep we peel curry wrappers when re-identifying a captured program's source.
/// Nesting is usually shallow; the bound just caps wasted work.
const MAX_UNCURRY_DEPTH: usize = 16;
/// Hard bound on the number of candidate inner-puzzle roots collected from one program.  A
/// pathological deeply-consed env can't force unbounded work.
const MAX_UNCURRY_CANDIDATES: usize = 512;

/// Recursively collect every `*.clsp` file under `dir` into `out` (a sorted set for
/// determinism).  Unreadable directories are silently skipped -- discovery is best-effort.
fn discover_clsp_sources(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            discover_clsp_sources(&path, out);
        } else if path.extension().and_then(|x| x.to_str()) == Some("clsp") {
            out.insert(path);
        }
    }
}

/// Build the GLOBAL attribution table by compiling every `.clsp` discovered under
/// `source_root` and each `--include` dir (recursively), honoring each source's dialect
/// sigil exactly as the per-source path does.  A source that fails to compile is WARNED via
/// the returned `skipped` list and skipped -- it does not abort the run.
pub fn build_global_table(
    source_root: &str,
    includes: &[String],
    force_optimize: bool,
) -> GlobalSourceTable {
    let mut files: BTreeSet<PathBuf> = BTreeSet::new();
    discover_clsp_sources(Path::new(source_root), &mut files);
    for inc in includes {
        discover_clsp_sources(Path::new(inc), &mut files);
    }

    let mut inc_paths = vec![source_root.to_string()];
    inc_paths.extend(includes.iter().cloned());

    let mut srcloc_syms: HashMap<String, String> = HashMap::new();
    // Hashes seen in >1 source (or in one source mapping to >1 srcloc): AMBIGUOUS, kept out of
    // the table so the decorator falls back to the enclosing unique ancestor's srcloc.
    let mut collided: HashSet<String> = HashSet::new();
    let mut reachable: BTreeSet<String> = BTreeSet::new();
    let mut compiled: Vec<String> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    // Per-source collision-free tables + root-hash -> index, for curried re-identification.
    let mut sources: Vec<(String, HashMap<String, String>, HashSet<String>)> = Vec::new();
    let mut by_root: HashMap<String, usize> = HashMap::new();

    // `files` is a BTreeSet so this Vec is sorted-by-path.  Compiles run in PARALLEL (each
    // source is independent: its own `Allocator`, `gensym` is a process-wide atomic whose
    // internal variable numbers never reach the compiled bytes or the file:line srcloc strings,
    // and the int-conversion flag is thread-local).  Results are stored by index and MERGED
    // BELOW strictly in sorted-path order, so the collision set -- and thus the whole attribution
    // table -- is byte-for-byte identical to the sequential build regardless of thread
    // scheduling.  A single module compile is ~8s (CSE + dep-graph + common/standalone phases +
    // the classic optimizer); fanning 181 of them across the box turns ~24min into ~1-2min.
    let files_vec: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().to_string())
        .collect();
    let n_files = files_vec.len();

    let jobs = std::env::var("CLVM_COV_JOBS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .clamp(1, n_files.max(1));

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<CompiledSource, String>>>> =
        Mutex::new((0..n_files).map(|_| None).collect());

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n_files {
                    break;
                }
                let path = &files_vec[i];
                let r = compile_source(path, &inc_paths, force_optimize);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                match &r {
                    Ok(cs) => eprintln!(
                        "[clvm_cov] compiled {d}/{n_files} ({} reachable): {path}",
                        cs.reachable.len()
                    ),
                    Err(e) => eprintln!("[clvm_cov] compiled {d}/{n_files} (FAILED: {e}): {path}"),
                }
                results.lock().expect("clvm_cov compile mutex poisoned")[i] = Some(r);
            });
        }
    });

    let results = results
        .into_inner()
        .expect("clvm_cov compile mutex poisoned");
    for (i, slot) in results.into_iter().enumerate() {
        let path = files_vec[i].clone();
        // Every slot was filled: work indices are handed out uniquely by the atomic counter.
        match slot.expect("every source index compiled exactly once") {
            Ok(cs) => {
                // A hash one source dropped as INTRA-ambiguous must not survive in the union
                // under a sibling's location: an execution of it in a capture of THIS source
                // would credit the sibling's line.  Treat it as a union collision outright.
                for hash in &cs.ambiguous {
                    if !collided.contains(hash) {
                        srcloc_syms.remove(hash);
                        collided.insert(hash.clone());
                    }
                }
                // Merge into the collision-pruned UNION table (iterating by reference so the
                // per-source table can be retained below for curried re-identification).
                for (hash, loc) in &cs.srcloc_syms {
                    if collided.contains(hash) {
                        continue;
                    }
                    match srcloc_syms.get(hash) {
                        Some(existing) if existing != loc => {
                            // Same subtree, two different source locations -> ambiguous. Drop it.
                            srcloc_syms.remove(hash);
                            collided.insert(hash.clone());
                        }
                        Some(_) => {} // identical loc string: nothing to do (never happens x-file).
                        None => {
                            srcloc_syms.insert(hash.clone(), loc.clone());
                        }
                    }
                }
                reachable.extend(cs.reachable);
                // Register the source's identity (its compiled root) and retain its collision-free
                // per-source table, so a curried capture re-identified with this root attributes
                // its shared body here even though the union dropped it.  First-wins on a shared
                // root (identical sibling sources map to the same root).
                let idx = sources.len();
                for h in &cs.root_hashes {
                    by_root.entry(h.clone()).or_insert(idx);
                }
                sources.push((path.clone(), cs.srcloc_syms, cs.ambiguous));
                compiled.push(path);
            }
            Err(e) => skipped.push((path, e)),
        }
    }

    GlobalSourceTable {
        srcloc_syms,
        reachable,
        compiled,
        skipped,
        sources,
        by_root,
    }
}

/// Run one corpus record through the `CldbRun` step loop and return the set of executed
/// srcloc strings (union of every stepped node's `.loc()` file field).
///
/// Both the program hex AND the args hex are decorated with `srcloc_syms` so executed nodes
/// carry their source location. The args are decorated because a program can pass a
/// sub-program in its solution and apply it (`(a subprog ...)`); those solution nodes are
/// code. Inert solution data is never spuriously attributed -- only STEPPED nodes count.
///
/// * `expected_roots` -- when `Some(roots)`, the corpus program's own root treehash is compared
///   against them (a multi-export module has one root per export) and the returned `bool` reports
///   whether any matched (used by the per-record `source` path to flag a program compiled with
///   different settings).  When `None` (the GLOBAL-table path, no single source root to compare
///   against), the check is skipped and `matched` is reported as `true`.
/// * `fallback_name` -- the synthetic `Srcloc` file stamped onto program nodes that the table
///   does not resolve (and their descendants).  The per-source path uses `*program*`; the
///   global path uses `*unmapped*` so genuinely-unattributed program nodes are distinguishable
///   from ordinary synthetic locations in the executed set.
///
/// Also returns the record's ARTIFACT: every subtree treehash of the captured program.  The
/// caller intersects the source's srcloc table with the union of matched artifacts to compute
/// the lines the producer's actual bytes can express (the measurable denominator).
pub fn run_record(
    program_hex: &str,
    args_hex: &str,
    srcloc_syms: &HashMap<String, String>,
    expected_roots: Option<&[String]>,
    fallback_name: &str,
) -> Result<(BTreeSet<String>, bool, HashSet<String>), String> {
    let mut allocator = Allocator::new();

    let program = hex_to_modern_sexp(
        &mut allocator,
        srcloc_syms,
        Srcloc::start(fallback_name),
        program_hex,
    )
    .map_err(|e| format!("bad program hex: {e}"))?;

    let mut artifact: HashSet<String> = HashSet::new();
    {
        let program_ref: &SExp = program.borrow();
        walk_subtree_hashes(program_ref, &mut |h, _| {
            artifact.insert(h.to_string());
        });
    }

    // Does this corpus program actually match the source we compiled?  sha256tree ignores
    // srclocs, so a match means the hashes in `srcloc_syms` decorate real nodes.  On the
    // global-table path there is no single expected root, so we skip the check.
    let matched = match expected_roots {
        Some(roots) => {
            let mut throwaway: HashMap<String, String> = HashMap::new();
            let loaded_root = build_symbol_table_mut(&mut throwaway, program.borrow()).hex();
            roots.contains(&loaded_root)
        }
        None => true,
    };

    // Decorate the ARGS with the same source map, not just the program. A program can pass a
    // sub-program in its solution and apply it via `(a subprog ...)`, so those solution nodes
    // are CODE, not inert data. Only STEPPED nodes enter the executed set, so inert solution
    // data is never spuriously attributed -- but a solution-passed sub-program that runs now
    // attributes to its own source instead of the synthetic `*args*`.
    let env = hex_to_modern_sexp(
        &mut allocator,
        srcloc_syms,
        Srcloc::start("*args*"),
        args_hex,
    )
    .map_err(|e| format!("bad args hex: {e}"))?;

    let runner = Rc::new(DefaultProgramRunner::new());
    let mut prim_map = HashMap::new();
    for p in prims::prims().iter() {
        prim_map.insert(p.0.clone(), Rc::new(p.1.clone()));
    }

    let cldbenv = CldbRunEnv::new(None, Rc::new(Vec::new()), Box::new(CldbNoOverride::new()));
    let step = start_step(program, env);
    let mut run = CldbRun::new(runner, Rc::new(prim_map), Box::new(cldbenv), step);

    let mut executed: BTreeSet<String> = BTreeSet::new();
    let mut steps: u64 = 0;
    loop {
        if run.is_ended() {
            break;
        }
        if steps > MAX_STEPS {
            return Err(format!("program exceeded {MAX_STEPS} steps; aborting"));
        }
        // Record the location we are about to leave and the one we advance to; the union
        // over the whole run is the executed set.
        let before = run.current_step().loc().file.to_string();
        executed.insert(before);
        run.step(&mut allocator);
        let after = run.current_step().loc().file.to_string();
        executed.insert(after);
        steps += 1;
    }

    Ok((executed, matched, artifact))
}

/// The sha256 treehash (hex) of a modern `SExp`, computed the SAME way `build_symbol_table_mut`
/// keys the attribution table (`build_table_mut`), so a hash produced here is directly comparable
/// against a table key or a source's compiled root.  Uses a throwaway symbol map.
fn program_root_hash(s: &SExp) -> String {
    let mut throwaway: HashMap<String, String> = HashMap::new();
    build_symbol_table_mut(&mut throwaway, s).hex()
}

/// If `s` is a standard Chia curry/apply wrapper `(a (q . INNER) ENV)` -- i.e. `(2 (1 . INNER) ENV)`
/// -- return `(INNER, ENV)`.  This is the shape `CurriedProgram`/`clvm_curried_args!` emit and the
/// shape a compiled module carries (`(a (q . BODY) 1)`), so peeling it recovers the puzzle whose
/// nodes match the raw-source compile.
fn as_apply(s: &SExp) -> Option<(Rc<SExp>, Rc<SExp>)> {
    if let SExp::Cons(_, op, rest) = s {
        if op.to_bigint() != Some(2u32.into()) {
            return None;
        }
        if let SExp::Cons(_, p, rest1) = rest.borrow() {
            if let SExp::Cons(_, env, tail) = rest1.borrow() {
                if !tail.nilp() {
                    return None; // `a` is arity-2; anything else is not a clean apply wrapper
                }
                if let SExp::Cons(_, q, inner) = p.borrow() {
                    if q.to_bigint() == Some(1u32.into()) {
                        return Some((inner.clone(), env.clone()));
                    }
                }
            }
        }
    }
    None
}

/// Walk a curry ENV builder `(c (q . a1) (c (q . a2) ... 1))` -- i.e. `(4 (1 . a1) (4 (1 . a2) ... 1))`
/// -- pushing each QUOTED curry argument `ai` onto `out`.  A curried argument can itself be a
/// puzzle (an inner puzzle curried into an outer wrapper), so callers recurse into each `ai`.
/// Stops at the first non-cons / non-`c` tail (the terminal env-path `1`).
fn walk_env_curry_args(env: &Rc<SExp>, out: &mut Vec<Rc<SExp>>) {
    let mut cur = env.clone();
    loop {
        let next = match cur.borrow() {
            SExp::Cons(_, cop, cargs) if cop.to_bigint() == Some(4u32.into()) => {
                if let SExp::Cons(_, argp, rest) = cargs.borrow() {
                    if let SExp::Cons(_, q, ai) = argp.borrow() {
                        if q.to_bigint() == Some(1u32.into()) {
                            out.push(ai.clone());
                        }
                    }
                    // The next `c`-form is the FIRST element of `rest` (`(c X NEXT)`).
                    if let SExp::Cons(_, y, _) = rest.borrow() {
                        Some(y.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        match next {
            Some(n) => cur = n,
            None => break,
        }
    }
}

/// Collect candidate inner-puzzle SExps from a captured program by peeling curry wrappers:
/// every `(a (q . INNER) ENV)` contributes `INNER` (recursed for nested wrapping) and each of
/// ENV's quoted curry arguments (recursed -- an outer wrapper may curry an inner puzzle as an arg).
/// Bounded by depth, a visited-hash set, and a candidate cap so a malformed program can't blow up.
fn collect_curry_candidates(
    prog: &Rc<SExp>,
    depth: usize,
    seen: &mut HashSet<String>,
    out: &mut Vec<Rc<SExp>>,
) {
    if depth > MAX_UNCURRY_DEPTH || out.len() >= MAX_UNCURRY_CANDIDATES {
        return;
    }
    if let Some((inner, env)) = as_apply(prog) {
        let h = program_root_hash(&inner);
        if seen.insert(h) {
            out.push(inner.clone());
            collect_curry_candidates(&inner, depth + 1, seen, out);
        }
        let mut args = Vec::new();
        walk_env_curry_args(&env, &mut args);
        for ai in args {
            if out.len() >= MAX_UNCURRY_CANDIDATES {
                break;
            }
            let ha = program_root_hash(&ai);
            if seen.insert(ha) {
                out.push(ai.clone());
                collect_curry_candidates(&ai, depth + 1, seen, out);
            }
        }
    }
}

/// Given a captured program hex, return the indices (into `GlobalSourceTable::sources`) of every
/// source whose compiled root hash matches EITHER the program's own root OR an inner puzzle peeled
/// from its curry wrappers.  A match PROVES the (sub)program is that source, letting the caller
/// attribute against that source's collision-free per-source table instead of the union.
///
/// Deterministic and side-effect-free; returns an empty vec when the program is unparseable or no
/// (sub)program's root is a known source (the caller then falls back to the union table).
fn identify_sources_by_uncurry(program_hex: &str, g: &GlobalSourceTable) -> Vec<usize> {
    let mut allocator = Allocator::new();
    let empty: HashMap<String, String> = HashMap::new();
    let prog = match hex_to_modern_sexp(
        &mut allocator,
        &empty,
        Srcloc::start("*uncurry*"),
        program_hex,
    ) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };

    // Candidates = the program's OWN root (a DIRECT, un-wrapped capture is its source root) plus
    // every inner puzzle recovered by peeling curry wrappers.
    let mut candidates: Vec<Rc<SExp>> = vec![prog.clone()];
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(program_root_hash(&prog));
    collect_curry_candidates(&prog, 0, &mut seen, &mut candidates);

    let mut matched: Vec<usize> = Vec::new();
    let mut matched_set: HashSet<usize> = HashSet::new();
    for cand in &candidates {
        let h = program_root_hash(cand);
        if let Some(&idx) = g.by_root.get(&h) {
            if matched_set.insert(idx) {
                matched.push(idx);
            }
        }
    }
    matched
}

/// Build the decoration table for a source-less record: the union table OVERLAID with the
/// collision-free per-source tables of every source the record was re-identified with.  Overlaying
/// the identified source's table adds back the sibling-shared subtrees the union dropped, and --
/// because re-identification PROVES this program IS that source -- attributing those subtrees to it
/// is correct even though a sibling also owns them.  A subtree that two DISTINCT identified sources
/// disagree on (rare: two matched sources both structurally present) stays ambiguous and is dropped.
fn table_for_identified(g: &GlobalSourceTable, matched: &[usize]) -> HashMap<String, String> {
    let mut table = g.srcloc_syms.clone();
    if matched.is_empty() {
        return table;
    }
    // Fold the matched sources' per-source entries; detect intra-matched conflicts.
    let mut overlay: HashMap<String, String> = HashMap::new();
    let mut conflict: HashSet<String> = HashSet::new();
    for &idx in matched {
        if let Some((_, syms, ambiguous)) = g.sources.get(idx) {
            for (h, loc) in syms {
                match overlay.get(h) {
                    Some(prev) if prev != loc => {
                        conflict.insert(h.clone());
                    }
                    _ => {
                        overlay.insert(h.clone(), loc.clone());
                    }
                }
            }
            // A hash ambiguous WITHIN the identified source must not credit anyone: the
            // record IS this source, so the sibling loc the union may carry is wrong and the
            // source's own verdict ("cannot tell the occurrences apart") is authoritative.
            for h in ambiguous {
                conflict.insert(h.clone());
            }
        }
    }
    for (h, loc) in overlay {
        if conflict.contains(&h) {
            table.remove(&h);
        } else {
            table.insert(h, loc);
        }
    }
    for h in &conflict {
        table.remove(h);
    }
    table
}

/// Parse a `Srcloc::to_string()` value -- either `file(line):col` or
/// `file(line):col-file(line):col` -- into `(file, line)`.  Returns `None` for synthetic
/// locations (`*program*`, `*macros*`, `*sym*`, ...) which are not real source.
pub fn parse_srcloc_string(s: &str) -> Option<(String, usize)> {
    // Find the first "(<digits>):" -- the file is everything before its '('.
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // Try to read digits then ')' then ':'.
            let mut j = i + 1;
            let start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && j < bytes.len() && bytes[j] == b')' {
                let file = &s[..i];
                if file.is_empty() || file.starts_with('*') {
                    return None;
                }
                let line: usize = s[start..j].parse().ok()?;
                return Some((file.to_string(), line));
            }
        }
        i += 1;
    }
    None
}

/// Fold a set of srcloc strings into per-file line sets.
fn lines_by_file(srclocs: &BTreeSet<String>) -> BTreeMap<String, BTreeSet<usize>> {
    let mut out: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for s in srclocs {
        if let Some((file, line)) = parse_srcloc_string(s) {
            out.entry(file).or_default().insert(line);
        }
    }
    out
}

/// Render LCOV text for one covered-line selector against the reachable set.
fn render_lcov(
    report: &CoverageReport,
    covered: &dyn Fn(&FileReport) -> BTreeSet<usize>,
) -> String {
    let mut s = String::new();
    for (file, fr) in &report.files {
        let cov = covered(fr);
        s.push_str("TN:\n");
        s.push_str(&format!("SF:{file}\n"));
        for line in &fr.reachable {
            let hit = if cov.contains(line) { 1 } else { 0 };
            s.push_str(&format!("DA:{line},{hit}\n"));
        }
        s.push_str(&format!("LF:{}\n", fr.reachable.len()));
        s.push_str(&format!("LH:{}\n", cov.len()));
        s.push_str("end_of_record\n");
    }
    s
}

fn pct(covered: usize, total: usize) -> f64 {
    if total == 0 {
        100.0
    } else {
        100.0 * covered as f64 / total as f64
    }
}

/// Render the `--summary` table: per-file union coverage, a per-layer total, and the
/// cold worklist (reachable lines no record hit).
fn render_summary(report: &CoverageReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("Layers observed: {}\n", {
        let v: Vec<&str> = report.layers.iter().map(|x| x.as_str()).collect();
        if v.is_empty() {
            "(none)".to_string()
        } else {
            v.join(", ")
        }
    }));
    s.push_str(&format!(
        "{:<40} {:>10} {:>8} {:>6}\n",
        "FILE", "UNION", "TOTAL", "EXCL"
    ));

    let (mut tu, mut tt, mut tx) = (0usize, 0usize, 0usize);
    for (file, fr) in &report.files {
        let total = fr.reachable.len();
        let union = fr.covered_union.len();
        tu += union;
        tt += total;
        tx += fr.unmeasurable;
        s.push_str(&format!(
            "{:<40} {:>9.1}% {:>8} {:>6}\n",
            short_path(file),
            pct(union, total),
            total,
            fr.unmeasurable
        ));
    }
    s.push_str(&format!(
        "{:<40} {:>9.1}% {:>8} {:>6}\n",
        "TOTAL",
        pct(tu, tt),
        tt,
        tx
    ));

    // Per-layer total coverage.
    if !report.layers.is_empty() {
        let empty = BTreeSet::new();
        s.push_str("\nBY LAYER:\n");
        for layer in &report.layers {
            let (mut c, mut t) = (0usize, 0usize);
            for fr in report.files.values() {
                c += fr.covered_by_layer.get(layer).unwrap_or(&empty).len();
                t += fr.reachable.len();
            }
            s.push_str(&format!("  {:<24} {:>6.1}%\n", layer, pct(c, t)));
        }
    }

    // Cold: reachable but covered by nothing.
    s.push_str("\nCOLD (reachable, covered by no record):\n");
    let mut any_cold = false;
    for (file, fr) in &report.files {
        let cold: Vec<usize> = fr
            .reachable
            .iter()
            .filter(|l| !fr.covered_union.contains(l))
            .copied()
            .collect();
        if !cold.is_empty() {
            any_cold = true;
            let lines: Vec<String> = cold.iter().map(|l| l.to_string()).collect();
            s.push_str(&format!(
                "  {}: lines {}\n",
                short_path(file),
                lines.join(", ")
            ));
        }
    }
    if !any_cold {
        s.push_str("  (none)\n");
    }
    s
}

fn short_path(p: &str) -> String {
    Path::new(p)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| p.to_string())
}

/// Resolve a record's `source` against the `--source` root.
fn resolve_source(root: &str, source: &str) -> String {
    let p = Path::new(source);
    if p.is_absolute() {
        source.to_string()
    } else {
        let mut base = PathBuf::from(root);
        base.push(source);
        base.to_string_lossy().to_string()
    }
}

/// Load and parse the JSONL corpus.
fn load_corpus(path: &str) -> Result<Vec<CorpusRecord>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("could not read corpus {path}: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: CorpusRecord =
            serde_json::from_str(trimmed).map_err(|e| format!("corpus line {}: {e}", n + 1))?;
        out.push(rec);
    }
    Ok(out)
}

/// Per-file coverage: reachable lines, the union covered set, and per-layer covered sets.
#[derive(Debug, Clone, Default)]
pub struct FileReport {
    pub reachable: BTreeSet<usize>,
    pub covered_union: BTreeSet<usize>,
    pub covered_by_layer: BTreeMap<String, BTreeSet<usize>>,
    /// Source lines carried by the compile that the matched corpus artifacts cannot express
    /// (no shared subtree) -- outside the measurable denominator, shown as EXCL.
    pub unmeasurable: usize,
}

/// Whole-corpus coverage: per-file reports plus the set of layers seen.
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    pub files: BTreeMap<String, FileReport>,
    pub layers: BTreeSet<String>,
    /// Non-fatal warnings (e.g. a corpus `program` hex whose treehash did not match the
    /// source we compiled -- coverage for it is best-effort).
    pub warnings: Vec<String>,
    /// Distinct executed locations on the source-LESS (global-table) path that resolved to no
    /// source file -- program nodes the global table could not attribute (they carry the
    /// `*unmapped*` fallback).  Informational only: these are simply uncovered-by-us, not an
    /// error.  Empty when every source-less program node mapped to some compiled source.
    pub unattributed: BTreeSet<String>,
    /// 1-based corpus record numbers whose executed nodes included unattributed locations.
    pub unattributed_records: BTreeSet<usize>,
}

/// Core: run the whole corpus and return aggregated, layer-aware coverage.  Separated from
/// CLI/IO so it is unit-testable.
pub fn run_corpus(
    records: &[CorpusRecord],
    source_root: &str,
    includes: &[String],
    force_optimize: bool,
) -> Result<CoverageReport, String> {
    // Compile each distinct resolved source once (records that carry an explicit `source`).
    let mut compiled: HashMap<String, CompiledSource> = HashMap::new();
    // executed srcloc strings, per layer.
    let mut executed_by_layer: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut layers: BTreeSet<String> = BTreeSet::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut mismatched_sources: BTreeSet<String> = BTreeSet::new();
    // Union of matched artifact hashes per source path (the measurable-denominator filter).
    let mut artifact_by_file: HashMap<String, HashSet<String>> = HashMap::new();

    // The GLOBAL attribution table is built lazily -- only if some record lacks a `source`.
    // A corpus that supplies `source` on every record pays nothing for the global machinery.
    let mut global: Option<GlobalSourceTable> = None;
    let mut unattributed: BTreeSet<String> = BTreeSet::new();
    let mut unattributed_records: BTreeSet<usize> = BTreeSet::new();

    let n_records = records.len();
    for (ri, rec) in records.iter().enumerate() {
        eprintln!("[clvm_cov] stepping record {}/{}", ri + 1, n_records);
        // A `source` that is present and non-empty selects the original per-source path; a
        // missing or empty `source` selects the global-table path.
        let explicit_source = rec
            .source
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());

        let executed = if let Some(source) = explicit_source {
            let resolved = resolve_source(source_root, source);
            if !compiled.contains_key(&resolved) {
                let mut inc = vec![source_root.to_string()];
                inc.extend(includes.iter().cloned());
                let cs = compile_source(&resolved, &inc, force_optimize)?;
                compiled.insert(resolved.clone(), cs);
            }
            let cs = compiled.get(&resolved).unwrap();
            let (executed, matched, artifact) = run_record(
                &rec.program,
                &rec.args,
                &cs.srcloc_syms,
                Some(&cs.root_hashes),
                "*program*",
            )?;
            if matched {
                artifact_by_file
                    .entry(resolved.clone())
                    .or_default()
                    .extend(artifact);
            }
            if !matched && mismatched_sources.insert(resolved.clone()) {
                warnings.push(format!(
                    "corpus program hex for {resolved} does not match the source compiled \
                     srcloc-preserving (treehash differs, likely optimized) -- line mapping for \
                     this source is best-effort"
                ));
            }
            executed
        } else {
            // Source-less record: build the global table on first sight, then attribute this
            // program's nodes against it.  There is no single expected root, so `run_record`
            // skips the treehash check; unresolved nodes carry the `*unmapped*` fallback.
            if global.is_none() {
                let g = build_global_table(source_root, includes, force_optimize);
                for (path, err) in &g.skipped {
                    warnings.push(format!(
                        "skipping source {path}: failed to compile for the global attribution \
                         table ({err}) -- programs whose nodes came from it will be unattributed"
                    ));
                }
                global = Some(g);
            }
            let g = global.as_ref().unwrap();
            // Re-identify the captured program with a specific source by UNCURRYING it and matching
            // an inner puzzle's root hash to a known source.  When identified, decorate against that
            // source's collision-free per-source table so a curried program attributes a body it
            // shares with sibling sources -- which the union table drops as ambiguous.  Unidentified
            // => the union table, so nothing regresses for programs that were already attributing.
            let matched = identify_sources_by_uncurry(&rec.program, g);
            let table = table_for_identified(g, &matched);
            let (executed, _matched, artifact) =
                run_record(&rec.program, &rec.args, &table, None, "*unmapped*")?;
            for &idx in &matched {
                if let Some((path, _, _)) = g.sources.get(idx) {
                    artifact_by_file
                        .entry(path.clone())
                        .or_default()
                        .extend(artifact.iter().cloned());
                }
            }
            let mut any_unmapped = false;
            for loc in &executed {
                if loc.starts_with("*unmapped*") {
                    unattributed.insert(loc.clone());
                    any_unmapped = true;
                }
            }
            if any_unmapped {
                unattributed_records.insert(ri + 1);
            }
            executed
        };

        layers.insert(rec.layer.clone());
        executed_by_layer
            .entry(rec.layer.clone())
            .or_default()
            .extend(executed);
    }

    // The measurable denominator.  For each compiled source, the lines its srcloc table can
    // CREDIT; when one or more corpus artifacts were positively matched to the source (by
    // root), only table entries whose subtree hash occurs in some matched artifact count -- a
    // line the producer's actual bytes cannot express is not measurable by this corpus, so it
    // is reported per-file under EXCL instead of posing as permanently cold (a classic
    // producer's stage_2 bytes diverge from the modern srcloc-bearing recompile; only the
    // shared subtrees are observable).  A source no artifact matched keeps its full table.
    let mut denom_tables: BTreeMap<String, &HashMap<String, String>> = BTreeMap::new();
    for (path, cs) in &compiled {
        denom_tables.insert(path.clone(), &cs.srcloc_syms);
    }
    if let Some(g) = global.as_ref() {
        for (path, syms, _) in &g.sources {
            denom_tables.entry(path.clone()).or_insert(syms);
        }
    }
    let mut reachable_srclocs: BTreeSet<String> = BTreeSet::new();
    let mut all_srclocs: BTreeSet<String> = BTreeSet::new();
    for (path, table) in &denom_tables {
        let artifact = artifact_by_file.get(path);
        for (h, loc) in table.iter() {
            if parse_srcloc_string(loc).is_none() {
                continue;
            }
            all_srclocs.insert(loc.clone());
            if artifact.map(|a| a.contains(h)).unwrap_or(true) {
                reachable_srclocs.insert(loc.clone());
            }
        }
    }
    let reach_lines = lines_by_file(&reachable_srclocs);
    let all_lines = lines_by_file(&all_srclocs);

    // Per-layer covered lines per file, and the union.
    let mut union_srclocs: BTreeSet<String> = BTreeSet::new();
    let mut layer_lines: BTreeMap<String, BTreeMap<String, BTreeSet<usize>>> = BTreeMap::new();
    for (layer, execset) in &executed_by_layer {
        union_srclocs.extend(execset.iter().cloned());
        layer_lines.insert(layer.clone(), lines_by_file(execset));
    }
    let union_lines = lines_by_file(&union_srclocs);

    let mut files: BTreeMap<String, FileReport> = BTreeMap::new();
    for (file, alines) in &all_lines {
        let empty = BTreeSet::new();
        let rlines = reach_lines.get(file).cloned().unwrap_or_default();
        let unmeasurable = alines.difference(&rlines).count();
        let ul = union_lines.get(file).unwrap_or(&empty);
        let covered_union: BTreeSet<usize> = rlines.intersection(ul).copied().collect();

        let mut covered_by_layer: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for layer in &layers {
            let ll = layer_lines
                .get(layer)
                .and_then(|m| m.get(file))
                .unwrap_or(&empty);
            covered_by_layer.insert(layer.clone(), rlines.intersection(ll).copied().collect());
        }

        files.insert(
            file.clone(),
            FileReport {
                reachable: rlines,
                covered_union,
                covered_by_layer,
                unmeasurable,
            },
        );
    }

    Ok(CoverageReport {
        files,
        layers,
        warnings,
        unattributed,
        unattributed_records,
    })
}

/// Binary entry point.  Returns a process exit code.
pub fn clvm_cov(args: &[String]) -> i32 {
    let parsed = match parse_args(args) {
        Ok(p) => p,
        Err(msg) => {
            // --help prints to stdout with success; real errors go to stderr.
            if msg.starts_with("clvm_cov --") || msg == USAGE {
                println!("{msg}");
                return 0;
            }
            eprintln!("{msg}");
            return 2;
        }
    };

    let records = match load_corpus(&parsed.corpus) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if records.is_empty() {
        eprintln!("corpus {} contained no records", parsed.corpus);
        return 1;
    }

    let report = match run_corpus(
        &records,
        &parsed.source_root,
        &parsed.includes,
        parsed.force_optimize,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    for w in &report.warnings {
        eprintln!("warning: {w}");
    }
    if !report.unattributed.is_empty() {
        let mut recs: Vec<String> = report
            .unattributed_records
            .iter()
            .take(10)
            .map(|r| r.to_string())
            .collect();
        if report.unattributed_records.len() > 10 {
            recs.push(format!("... {} total", report.unattributed_records.len()));
        }
        eprintln!(
            "note: {} distinct executed program location(s) from source-less records mapped to \
             no compiled source (unattributed) -- uncovered-by-us, not an error (corpus \
             record(s): {})",
            report.unattributed.len(),
            recs.join(", ")
        );
    }

    if let Some(out) = &parsed.out {
        // Union LCOV.
        let union = render_lcov(&report, &|fr| fr.covered_union.clone());
        if let Err(e) = fs::write(out, union) {
            eprintln!("could not write {out}: {e}");
            return 1;
        }
        println!("wrote union LCOV to {out}");

        // Per-layer LCOVs alongside: <stem>.<layer>.<ext>.
        for layer in &report.layers {
            let path = layer_lcov_path(out, layer);
            let lcov = render_lcov(&report, &|fr| {
                fr.covered_by_layer.get(layer).cloned().unwrap_or_default()
            });
            if let Err(e) = fs::write(&path, lcov) {
                eprintln!("could not write {path}: {e}");
                return 1;
            }
            println!("wrote {layer} LCOV to {path}");
        }
    }

    if parsed.summary || parsed.out.is_none() {
        print!("{}", render_summary(&report));
    }

    0
}

/// Given `dir/coverage.lcov` and layer `unit`, produce `dir/coverage.unit.lcov`.
fn layer_lcov_path(out: &str, layer: &str) -> String {
    let p = Path::new(out);
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "coverage".to_string());
    let ext = p
        .extension()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "lcov".to_string());
    let file = format!("{stem}.{layer}.{ext}");
    match p.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(file).to_string_lossy().to_string(),
        _ => file,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classic::clvm::sexp::sexp_as_bin;
    use crate::compiler::clvm::convert_to_clvm_rs;
    use crate::compiler::sexp::SExp;

    /// Compile a source string to a temp file and return (temp_dir, path).
    fn write_source(dir: &tempfile::TempDir, name: &str, body: &str) -> String {
        let path = dir.path().join(name);
        fs::write(&path, body).unwrap();
        path.to_string_lossy().to_string()
    }

    /// Compile `source_path` and serialize the compiled program to hex, so the test drives
    /// the exact corpus path (hex in, srcloc out) rather than a shortcut.
    fn compile_to_hex(source_path: &str, includes: &[String]) -> (String, HashMap<String, String>) {
        let mut allocator = Allocator::new();
        let content = fs::read_to_string(source_path).unwrap();
        let mut parsed_args: HashMap<String, ArgumentValue> = HashMap::new();
        parsed_args.insert(
            "path_or_code".to_string(),
            ArgumentValue::ArgString(Some(source_path.to_string()), content),
        );
        let include_vals: Vec<ArgumentValue> = includes
            .iter()
            .map(|d| ArgumentValue::ArgString(Some(d.clone()), d.clone()))
            .collect();
        parsed_args.insert("include".to_string(), ArgumentValue::ArgArray(include_vals));
        let input = RunAndCompileInputData::new(&mut allocator, &parsed_args).unwrap();
        let mut name_syms = HashMap::new();
        let res = input
            .compile_modern(&mut allocator, &mut name_syms)
            .unwrap();
        let res_sexp: &SExp = res.borrow();
        // Serialize through the real CLVM serializer (length-prefix framing), matching what
        // an on-chain producer would emit -- `SExp::encode` is an internal form cldb cannot
        // read back.
        let node = convert_to_clvm_rs(&mut allocator, res.clone()).unwrap();
        let hex = sexp_as_bin(&mut allocator, node).hex();
        let mut srcloc_syms = HashMap::new();
        build_symbol_table_mut(&mut srcloc_syms, res_sexp);
        (hex, srcloc_syms)
    }

    // A two-branch puzzle with each arm on its own line so line granularity can tell them
    // apart:
    //   line 1: (mod (X)
    //   line 2:   (if X
    //   line 3:     (q . 111)   <- then arm
    //   line 4:     (q . 222)   <- else arm
    const TWO_BRANCH: &str = "(mod (X)\n  (if X\n    (q . 111)\n    (q . 222)\n  )\n)\n";

    // args = (1) -> "ff0180" ; args = (0) -> "ff8080"
    const ARGS_ONE: &str = "ff0180";
    const ARGS_ZERO: &str = "ff8080";

    // Regression: a multi-export module must record the root of EVERY export, not just the
    // first -- otherwise a source-less capture of a non-first export cannot be identified by
    // root and mis-attributes.  `modtest2.clsp` exports two functions.
    #[test]
    fn multi_export_records_every_export_root() {
        let src = "resources/tests/module/modtest2.clsp";
        let includes = vec!["resources/tests/module".to_string()];
        let cs = compile_source(src, &includes, false).unwrap();
        assert!(
            cs.root_hashes.len() >= 2,
            "both export roots must be recorded, got {:?}",
            cs.root_hashes
        );
        assert_ne!(
            cs.root_hashes[0], cs.root_hashes[1],
            "distinct exports have distinct roots"
        );
    }

    fn rec(program: &str, args: &str, layer: &str) -> CorpusRecord {
        CorpusRecord {
            program: program.to_string(),
            args: args.to_string(),
            source: Some("two_branch.clsp".to_string()),
            layer: layer.to_string(),
        }
    }
    fn line_covered(report: &CoverageReport, file: &str, line: usize) -> bool {
        report
            .files
            .get(file)
            .map(|fc| fc.covered_union.contains(&line))
            .unwrap_or(false)
    }
    fn line_reachable(report: &CoverageReport, file: &str, line: usize) -> bool {
        report
            .files
            .get(file)
            .map(|fc| fc.reachable.contains(&line))
            .unwrap_or(false)
    }
    fn line_covered_by(report: &CoverageReport, file: &str, layer: &str, line: usize) -> bool {
        report
            .files
            .get(file)
            .and_then(|fc| fc.covered_by_layer.get(layer))
            .map(|s| s.contains(&line))
            .unwrap_or(false)
    }

    #[test]
    fn cold_branch_detected_when_only_one_arm_runs() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "two_branch.clsp", TWO_BRANCH);
        let includes = vec![dir.path().to_string_lossy().to_string()];
        let (program_hex, _syms) = compile_to_hex(&src, &includes);

        // Only exercise X=1: the (q . 111) arm on line 3 runs, the (q . 222) arm on line 4
        // must be COLD.
        let records = vec![rec(&program_hex, ARGS_ONE, "e2e")];
        let report = run_corpus(&records, &dir.path().to_string_lossy(), &[], false).unwrap();

        // Both arms are reachable (they exist in the compiled program).
        assert!(
            line_reachable(&report, &src, 3),
            "line 3 (then arm) should be reachable; cov={report:?}"
        );
        assert!(
            line_reachable(&report, &src, 4),
            "line 4 (else arm) should be reachable; cov={report:?}"
        );
        // Line 3 executed, line 4 cold.
        assert!(
            line_covered(&report, &src, 3),
            "line 3 (then arm) should be COVERED with X=1; cov={report:?}"
        );
        assert!(
            !line_covered(&report, &src, 4),
            "line 4 (else arm) should be COLD with X=1 only; cov={report:?}"
        );
    }

    #[test]
    fn both_arms_covered_when_both_inputs_run() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "two_branch.clsp", TWO_BRANCH);
        let includes = vec![dir.path().to_string_lossy().to_string()];
        let (program_hex, _syms) = compile_to_hex(&src, &includes);

        let records = vec![
            rec(&program_hex, ARGS_ONE, "e2e"),
            rec(&program_hex, ARGS_ZERO, "e2e"),
        ];
        let report = run_corpus(&records, &dir.path().to_string_lossy(), &[], false).unwrap();

        assert!(
            line_covered(&report, &src, 3),
            "line 3 should be covered; cov={report:?}"
        );
        assert!(
            line_covered(&report, &src, 4),
            "line 4 should now be covered too; cov={report:?}"
        );

        // 100% of the arm lines are covered; assert no arm line is cold.
        let fc = report.files.get(&src).unwrap();
        let cold: Vec<usize> = fc
            .reachable
            .iter()
            .filter(|l| (**l == 3 || **l == 4) && !fc.covered_union.contains(l))
            .copied()
            .collect();
        assert!(cold.is_empty(), "no arm line should be cold; cold={cold:?}");
    }

    #[test]
    fn per_layer_coverage_is_tracked_separately() {
        // Layer `e2e` exercises X=1 (line 3); layer `unit` fills X=0 (line 4). Each line is
        // attributed to the layer that hit it, and the union covers both.
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "two_branch.clsp", TWO_BRANCH);
        let includes = vec![dir.path().to_string_lossy().to_string()];
        let (program_hex, _syms) = compile_to_hex(&src, &includes);

        let records = vec![
            rec(&program_hex, ARGS_ONE, "e2e"),
            rec(&program_hex, ARGS_ZERO, "unit"),
        ];
        let report = run_corpus(&records, &dir.path().to_string_lossy(), &[], false).unwrap();

        assert!(line_covered(&report, &src, 3));
        assert!(line_covered(&report, &src, 4));
        assert!(line_covered_by(&report, &src, "e2e", 3));
        assert!(!line_covered_by(&report, &src, "e2e", 4));
        assert!(line_covered_by(&report, &src, "unit", 4));

        let summary = render_summary(&report);
        assert!(
            summary.contains("BY LAYER"),
            "summary lists per-layer totals:\n{summary}"
        );
    }

    #[test]
    fn missing_layer_defaults_to_unknown() {
        let json = r#"{"program":"ff","args":"80","source":"p.clsp"}"#;
        let r: CorpusRecord = serde_json::from_str(json).unwrap();
        assert_eq!(r.layer, "unknown");
    }

    #[test]
    fn parse_srcloc_string_forms() {
        assert_eq!(
            parse_srcloc_string("foo/bar.clsp(3):10"),
            Some(("foo/bar.clsp".to_string(), 3))
        );
        assert_eq!(
            parse_srcloc_string("foo/bar.clsp(4):10-foo/bar.clsp(4):13"),
            Some(("foo/bar.clsp".to_string(), 4))
        );
        assert_eq!(parse_srcloc_string("*program*(1):1"), None);
        assert_eq!(parse_srcloc_string("*macros*(2):27"), None);
        assert_eq!(parse_srcloc_string("no-loc-here"), None);
    }

    // A *standard-cl-25* source whose branches are (non-inline) defun CALLS. cl-25 (stepping 25 >
    // 22) forces the optimizer on (comp_input.rs:189-193), so the compiled hex is byte-identical to
    // an optimized producer's captured hex.
    //   line 3: (defun arm-a (v) (+ v 1))   <- then-arm body
    //   line 4: (defun arm-b (v) (* v 2))   <- else-arm body
    //   line 7:       (arm-a v)             <- then call site
    //   line 8:       (arm-b v)             <- else call site
    const CL25_DEFUN_BRANCH: &str = "(mod (X)\n  (include *standard-cl-25*)\n  (defun arm-a (v) \
(+ v 1))\n  (defun arm-b (v) (* v 2))\n  (defun route (flag v)\n    (if flag\n      (arm-a v)\n\
      (arm-b v)\n    )\n  )\n  (route X X)\n)\n";

    #[test]
    fn cl25_optimized_defun_branch_keeps_line_fidelity() {
        // This is THE production granularity check: cl-25 + forced optimizer, defun-structured
        // branches.  Running only X=1 must leave the untaken function's BODY line (4) COLD while
        // the taken function's body line (3) is covered.
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "cl25.clsp", CL25_DEFUN_BRANCH);
        let root = dir.path().to_string_lossy().to_string();
        let includes = vec![root.clone()];
        // compile_to_hex auto-detects the *standard-cl-25* sigil and thus emits OPTIMIZED hex,
        // exactly like the producer.
        let (program_hex, _syms) = compile_to_hex(&src, &includes);

        // X=1 only: route(1,1) -> arm-a; arm-b body (line 4) never runs.
        let records = vec![CorpusRecord {
            program: program_hex.clone(),
            args: ARGS_ONE.to_string(),
            source: Some("cl25.clsp".to_string()),
            layer: "e2e".to_string(),
        }];
        let report = run_corpus(&records, &root, &[], false).unwrap();

        // The recompiled source must byte-match the (optimized) corpus hex -> no warning.
        assert!(
            report.warnings.is_empty(),
            "cl-25 recompile should treehash-match the optimized hex; warnings={:?}",
            report.warnings
        );
        assert!(
            line_reachable(&report, &src, 3) && line_reachable(&report, &src, 4),
            "both defun bodies (lines 3,4) should be reachable; cov={report:?}"
        );
        assert!(
            line_covered(&report, &src, 3),
            "arm-a body (line 3) should be COVERED with X=1; cov={report:?}"
        );
        assert!(
            !line_covered(&report, &src, 4),
            "arm-b body (line 4) should be COLD with X=1 only -- optimizer preserved line \
             fidelity through the defun call; cov={report:?}"
        );

        // Adding X=0 covers arm-b's body line too.
        let both = vec![
            CorpusRecord {
                program: program_hex.clone(),
                args: ARGS_ONE.to_string(),
                source: Some("cl25.clsp".to_string()),
                layer: "e2e".to_string(),
            },
            CorpusRecord {
                program: program_hex.clone(),
                args: ARGS_ZERO.to_string(),
                source: Some("cl25.clsp".to_string()),
                layer: "unit".to_string(),
            },
        ];
        let report2 = run_corpus(&both, &root, &[], false).unwrap();
        assert!(
            line_covered(&report2, &src, 4),
            "arm-b body (line 4) should be covered once X=0 runs; cov={report2:?}"
        );
        // ...attributed to the `unit` layer, not `e2e`.
        assert!(
            line_covered_by(&report2, &src, "unit", 4)
                && !line_covered_by(&report2, &src, "e2e", 4),
            "line 4 belongs to the unit layer only; cov={report2:?}"
        );
    }

    #[test]
    fn source_field_omitted_deserializes_to_none() {
        // A runtime capture has no `source` at all.
        let json = r#"{"program":"ff","args":"80","layer":"e2e"}"#;
        let r: CorpusRecord = serde_json::from_str(json).unwrap();
        assert!(r.source.is_none(), "omitted source must be None");
        assert_eq!(r.layer, "e2e");
        // An explicit empty string is treated as source-less by run_corpus.
        let json2 = r#"{"program":"ff","args":"80","source":"","layer":"e2e"}"#;
        let r2: CorpusRecord = serde_json::from_str(json2).unwrap();
        assert_eq!(r2.source.as_deref(), Some(""));
    }

    // Two tiny, DISTINCT cl-25 sources for the source-less-attribution gate.
    //
    // Engineered so that per-source and global attribution are line-IDENTICAL -- which requires
    // two things, both of which mirror real coverage hygiene:
    //
    //  (1) No covered line may depend SOLELY on a subtree shared between the two sources.  A
    //      shared subtree (down to a bare env-path atom) is ambiguous, so the global table drops
    //      it and the decorator inherits the enclosing UNIQUE ancestor's srcloc.  To keep every
    //      covered line pinned by a unique node, B has a different env layout (an extra `b-pad`
    //      defun shifts `b-hit`/`b-cold` to different environment paths than `a-hit`/`a-cold`)
    //      and a different top wrapper (`(* 2 ...)`), so B's executed lines carry B-unique nodes.
    //  (2) The UNTAKEN branch body must be GENUINELY cold -- i.e. no atom evaluated on the taken
    //      path may map onto it.  The cold arm therefore reads the SECOND parameter `n`, and
    //      nothing on the executed path ever evaluates `n`; the taken path only touches `m`.
    //      (Without this, per-source `build_symbol_table_mut` last-wins-maps the shared `n` atom
    //      onto the cold line, spuriously "covering" it -- an artifact the global path can't and
    //      shouldn't reproduce.)
    //
    // Each keeps a defun-structured then/else so the untaken function body is a real COLD
    // reachable line (line fidelity through the cl-25 optimizer -- the same property the
    // cl25_optimized_defun_branch_keeps_line_fidelity test pins).  Real corpora that share
    // library subtrees (sha256tree.clib, curry glue) are attributed best-effort; that is out of
    // this gate's scope.
    const GATE_A: &str =
        "(mod (X)\n  (include *standard-cl-25*)\n  (defun a-hit (m n) (+ m 7))\n  \
(defun a-cold (m n) (+ n 9))\n  (if X (a-hit X X) (a-cold X X))\n)\n";
    const GATE_B: &str =
        "(mod (Y)\n  (include *standard-cl-25*)\n  (defun b-pad (m n) (+ m 1))\n  \
(defun b-hit (m n) (* m 5))\n  (defun b-cold (m n) (* n 3))\n  (* 2 (if (b-pad Y Y) (b-hit Y Y) \
(b-cold Y Y)))\n)\n";

    /// (covered_union, cold) line sets for one file in a report.
    fn cov_and_cold(report: &CoverageReport, file: &str) -> (BTreeSet<usize>, BTreeSet<usize>) {
        let fr = report.files.get(file).cloned().unwrap_or_default();
        let cold: BTreeSet<usize> = fr
            .reachable
            .iter()
            .filter(|l| !fr.covered_union.contains(l))
            .copied()
            .collect();
        (fr.covered_union, cold)
    }

    // ---- Red-canary: a CURRIED capture attributes its sibling-shared body to its own source ----
    //
    // A and B are TWO sources that share a byte-identical `shared` defun body (line 4) but differ at
    // the top (main adds 1 vs 2), so their COMPILED ROOTS differ -- like template siblings that share
    // a body.  Because the `shared` subtree is common to both, the UNION attribution table drops it
    // as ambiguous.  We then CURRY A -- `(a (q . COMPILED_A) 1)` -- so the
    // captured program's own root is NOT any source root; only UNCURRYING recovers the inner root
    // that equals A's compiled root and re-identifies the source.
    //
    // RED before the fix: with only the union table, A's shared body (line 4) is dropped -> COLD,
    // so `assert line 4 covered` fails.  GREEN after: uncurry identifies A, decorates against A's
    // collision-free per-source table, and line 4 lights -- attributed to A, never B.
    const CANARY_A: &str = "(mod (X)\n  (include *standard-cl-25*)\n  (defun shared (v)\n    \
(+ (* v v) 7)\n  )\n  (+ 1 (shared X))\n)\n";
    const CANARY_B: &str = "(mod (X)\n  (include *standard-cl-25*)\n  (defun shared (v)\n    \
(+ (* v v) 7)\n  )\n  (+ 2 (shared X))\n)\n";

    /// Compile `source_path` and serialize the compiled program WRAPPED in a curry/apply layer
    /// `(a (q . COMPILED) 1)`, so the emitted hex's own root differs from the source root and can
    /// only be re-identified by uncurrying.  Executing it applies COMPILED to the same env, so the
    /// body runs identically to the un-wrapped program.
    fn compile_to_curried_hex(source_path: &str, includes: &[String]) -> String {
        let mut allocator = Allocator::new();
        let content = fs::read_to_string(source_path).unwrap();
        let mut parsed_args: HashMap<String, ArgumentValue> = HashMap::new();
        parsed_args.insert(
            "path_or_code".to_string(),
            ArgumentValue::ArgString(Some(source_path.to_string()), content),
        );
        let include_vals: Vec<ArgumentValue> = includes
            .iter()
            .map(|d| ArgumentValue::ArgString(Some(d.clone()), d.clone()))
            .collect();
        parsed_args.insert("include".to_string(), ArgumentValue::ArgArray(include_vals));
        let input = RunAndCompileInputData::new(&mut allocator, &parsed_args).unwrap();
        let mut name_syms = HashMap::new();
        let compiled = input
            .compile_modern(&mut allocator, &mut name_syms)
            .unwrap();

        // Wrap: (a (q . COMPILED) 1) == (2 (1 . COMPILED) 1).
        let l = Srcloc::start("*canary-curry*");
        let two = Rc::new(SExp::Atom(l.clone(), vec![2]));
        let one = Rc::new(SExp::Atom(l.clone(), vec![1]));
        let qbody = Rc::new(SExp::Cons(l.clone(), one.clone(), compiled.clone()));
        let args_list = Rc::new(SExp::Cons(
            l.clone(),
            qbody,
            Rc::new(SExp::Cons(
                l.clone(),
                one.clone(),
                Rc::new(SExp::Nil(l.clone())),
            )),
        ));
        let wrapper = Rc::new(SExp::Cons(l.clone(), two, args_list));

        let node = convert_to_clvm_rs(&mut allocator, wrapper).unwrap();
        sexp_as_bin(&mut allocator, node).hex()
    }

    #[test]
    fn curried_capture_attributes_shared_body_to_its_source() {
        let dir = tempfile::tempdir().unwrap();
        let src_a = write_source(&dir, "a.clsp", CANARY_A);
        let src_b = write_source(&dir, "b.clsp", CANARY_B);
        let root = dir.path().to_string_lossy().to_string();
        let includes = vec![root.clone()];

        // Curry A (so only uncurry can re-identify it) and capture it SOURCE-LESS with X=3.
        let curried_a = compile_to_curried_hex(&src_a, &includes);
        let records = vec![CorpusRecord {
            program: curried_a,
            args: "ff0380".to_string(), // (3)
            source: None,
            layer: "e2e".to_string(),
        }];
        // b.clsp must be present so its shared body collides A's in the union table (the drop the
        // fix has to overcome). `--include root` discovers both.
        let report = run_corpus(&records, &root, &includes, false).unwrap();

        // Sanity: the shared body line is REACHABLE in A (denominator present).
        assert!(
            line_reachable(&report, &src_a, 4),
            "a.clsp line 4 (shared body) must be reachable; cov={report:?}"
        );
        // THE TEETH: the sibling-shared body (line 4), which the union table drops, must attribute
        // to A once the curried capture is uncurried and re-identified.  RED before the fix.
        assert!(
            line_covered(&report, &src_a, 4),
            "a.clsp line 4 (shared body, dropped by the union) must be COVERED after uncurry \
             re-identification; cov={report:?}"
        );
        // A's UNIQUE main line covers with or without the fix (control that the run really ran).
        assert!(
            line_covered(&report, &src_a, 6),
            "a.clsp line 6 (A's unique main) should be covered; cov={report:?}"
        );
        // Discrimination: the shared subtree is attributed to A, not to the sibling B.
        assert!(
            !line_covered(&report, &src_b, 4),
            "b.clsp line 4 must NOT be covered -- the capture is A, not B; cov={report:?}"
        );
    }

    // THE GATE: a corpus of SOURCE-LESS records must produce the SAME per-source covered/cold
    // line sets as the SAME corpus with an explicit per-record `source`.  That equivalence is
    // what proves the global-table attribution path is correct.
    #[test]
    fn sourceless_attribution_matches_explicit_source() {
        let dir = tempfile::tempdir().unwrap();
        let src_a = write_source(&dir, "a.clsp", GATE_A);
        let src_b = write_source(&dir, "b.clsp", GATE_B);
        let root = dir.path().to_string_lossy().to_string();
        let includes = vec![root.clone()];
        let (hex_a, _sa) = compile_to_hex(&src_a, &includes);
        let (hex_b, _sb) = compile_to_hex(&src_b, &includes);

        // Exercise the TAKEN arm of each (X=1 -> a-hit; Y=1 -> b-hit): the taken body is covered,
        // the untaken body is a cold reachable line.
        let with_source = vec![
            CorpusRecord {
                program: hex_a.clone(),
                args: ARGS_ONE.to_string(),
                source: Some("a.clsp".to_string()),
                layer: "e2e".to_string(),
            },
            CorpusRecord {
                program: hex_b.clone(),
                args: ARGS_ONE.to_string(),
                source: Some("b.clsp".to_string()),
                layer: "e2e".to_string(),
            },
        ];
        // Identical corpus with NO source field at all.
        let sourceless = vec![
            CorpusRecord {
                program: hex_a.clone(),
                args: ARGS_ONE.to_string(),
                source: None,
                layer: "e2e".to_string(),
            },
            CorpusRecord {
                program: hex_b.clone(),
                args: ARGS_ONE.to_string(),
                source: None,
                layer: "e2e".to_string(),
            },
        ];

        let expl = run_corpus(&with_source, &root, &includes, false).unwrap();
        let auto = run_corpus(&sourceless, &root, &includes, false).unwrap();

        // The equivalence gate: per-source covered AND cold sets are identical with and without
        // an explicit `source`.
        for src in [&src_a, &src_b] {
            let (ce, colde) = cov_and_cold(&expl, src);
            let (ca, colda) = cov_and_cold(&auto, src);
            assert_eq!(
                ce, ca,
                "covered lines for {src} must match with/without source\n  explicit={ce:?}\n  \
                 auto={ca:?}"
            );
            assert_eq!(
                colde, colda,
                "cold lines for {src} must match with/without source\n  explicit={colde:?}\n  \
                 auto={colda:?}"
            );
        }

        // Red-canary teeth: the equivalence is only meaningful if coverage is NON-TRIVIAL and
        // NON-VACUOUS -- each source must have BOTH a covered line and a cold line on the auto
        // path.  If the global union were broken (a program decorated against a missing/foreign
        // table) the covered set would collapse to empty; if attribution leaked across sources
        // the sets would diverge and the equality asserts above would already fire.
        for src in [&src_a, &src_b] {
            let (cov, cold) = cov_and_cold(&auto, src);
            assert!(
                !cov.is_empty(),
                "auto path must cover at least one line of {src} (empty => attribution broken); \
                 report={auto:?}"
            );
            assert!(
                !cold.is_empty(),
                "auto path must leave at least one reachable line of {src} COLD (the untaken \
                 defun body); report={auto:?}"
            );
        }
        // And no program node should have fallen through to the unattributed bucket for these
        // collision-free sources.
        assert!(
            auto.unattributed.is_empty(),
            "no node should be unattributed for these two fully-compiled sources; unattributed={:?}",
            auto.unattributed
        );
    }

    // ---- classic-producer identification (stage_2 bytes vs the modern srcloc artifact) ----
    //
    // A CLASSIC (no-sigil) producer -- `run` / `cdv clsp build` -- emits stage_2 bytes that do
    // NOT byte-match the modern compiler's srcloc-bearing artifact for the same source, so a
    // captured classic program used to be unidentifiable: its root matched no registered root,
    // sibling-shared subtrees stayed union-dropped, and the artifact filter never engaged.
    //
    // A and B share a byte-identical helper body (line 2); each has a unique main (line 3).
    const CLASSIC_A: &str = "(mod (X)\n  (defun helper (v) (+ v 101))\n  (+ 1 (helper X))\n)\n";
    const CLASSIC_B: &str = "(mod (X)\n  (defun helper (v) (+ v 101))\n  (* 3 (helper X))\n)\n";

    /// Compile `source_path` like a CLASSIC producer (the stage_2 pipeline: `run`,
    /// `cdv clsp build` -- no srclocs) and return the serialized hex: the bytes an on-chain
    /// capture of this source actually has.
    fn compile_classic_producer_hex(source_path: &str, includes: &[String]) -> String {
        let mut allocator = Allocator::new();
        let mut sym: HashMap<String, String> = HashMap::new();
        let content = fs::read_to_string(source_path).unwrap();
        let opts: Rc<dyn CompilerOpts> =
            Rc::new(DefaultCompilerOpts::new(source_path)).set_search_paths(includes);
        let node = compile_clvm_text_maybe_opt(
            &mut allocator,
            false,
            opts,
            &mut sym,
            &content,
            source_path,
            false,
        )
        .unwrap();
        sexp_as_bin(&mut allocator, node).hex()
    }

    /// Wrap serialized program hex in a standard curry/apply layer `(a (q . P) 1)` so its own
    /// root matches no source root and only uncurrying can re-identify it.
    fn curry_wrap_hex(program_hex: &str) -> String {
        let mut allocator = Allocator::new();
        let empty: HashMap<String, String> = HashMap::new();
        let prog = hex_to_modern_sexp(&mut allocator, &empty, Srcloc::start("*wrap*"), program_hex)
            .unwrap();
        let l = Srcloc::start("*wrap*");
        let two = Rc::new(SExp::Atom(l.clone(), vec![2]));
        let one = Rc::new(SExp::Atom(l.clone(), vec![1]));
        let qbody = Rc::new(SExp::Cons(l.clone(), one.clone(), prog));
        let args_list = Rc::new(SExp::Cons(
            l.clone(),
            qbody,
            Rc::new(SExp::Cons(
                l.clone(),
                one.clone(),
                Rc::new(SExp::Nil(l.clone())),
            )),
        ));
        let wrapper = Rc::new(SExp::Cons(l.clone(), two, args_list));
        let node = convert_to_clvm_rs(&mut allocator, wrapper).unwrap();
        sexp_as_bin(&mut allocator, node).hex()
    }

    #[test]
    fn classic_producer_capture_identifies_and_attributes() {
        let dir = tempfile::tempdir().unwrap();
        let src_a = write_source(&dir, "classic_a.clsp", CLASSIC_A);
        let src_b = write_source(&dir, "classic_b.clsp", CLASSIC_B);
        let root = dir.path().to_string_lossy().to_string();
        let includes = vec![root.clone()];

        // The producer's bytes: classic stage_2 compile of A, curry-wrapped, captured
        // source-less (the shape an on-chain capture has).
        let classic_hex = compile_classic_producer_hex(&src_a, &includes);
        let records = vec![CorpusRecord {
            program: curry_wrap_hex(&classic_hex),
            args: "ff0580".to_string(), // (5)
            source: None,
            layer: "e2e".to_string(),
        }];
        let report = run_corpus(&records, &root, &includes, false).unwrap();

        // Identification teeth: the helper body A shares with B (dropped from the union as
        // ambiguous) attributes to A -- possible only because A's CLASSIC root re-identified
        // the capture and A's per-source table was overlaid.
        assert!(
            line_reachable(&report, &src_a, 2),
            "a's helper body (line 2) must be measurable; cov={report:?}"
        );
        assert!(
            line_covered(&report, &src_a, 2),
            "a's helper body (line 2) must be covered via classic-root identification; \
             cov={report:?}"
        );
        // ...and to A only, never the sibling that also owns those bytes.
        assert!(
            !line_covered(&report, &src_b, 2),
            "b's helper line must NOT be covered -- the capture is A; cov={report:?}"
        );
        // Denominator honesty: measurable (TOTAL) + excluded (EXCL) partitions A's compiled
        // lines; nothing was silently lost.
        let fr = report.files.get(&src_a).unwrap();
        assert!(
            !fr.reachable.is_empty(),
            "a must have measurable lines; cov={report:?}"
        );
    }

    // ---- intra-source shared subexpression must not credit the untaken line ----
    //
    // pick-cold (line 3) and pick-hot (line 4) contain a byte-identical `(* v 3)`.  Running
    // only pick-hot used to credit line 3 as well: the table kept ONE location per subtree
    // hash (first-wins), so the taken occurrence stepped and lit the untaken line -- a false
    // positive there and a false negative on its own line.  An ambiguous-by-hash subtree now
    // credits nothing; each line is carried by its unique enclosing expression.
    const SHARED_SUBEXPR: &str = "(mod (X)\n  (include *standard-cl-25*)\n  (defun pick-cold \
(v) (- (* v 3) 2))\n  (defun pick-hot (v) (+ (* v 3) 1))\n  (if X (pick-hot X) (pick-cold X))\n)\n";

    #[test]
    fn intra_source_shared_subexpr_cannot_credit_the_untaken_line() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "shared_subexpr.clsp", SHARED_SUBEXPR);
        let root = dir.path().to_string_lossy().to_string();
        let includes = vec![root.clone()];
        let (hex, _) = compile_to_hex(&src, &includes);
        let records = vec![CorpusRecord {
            program: hex,
            args: ARGS_ONE.to_string(),
            source: Some("shared_subexpr.clsp".to_string()),
            layer: "e2e".to_string(),
        }];
        let report = run_corpus(&records, &root, &[], false).unwrap();
        assert!(
            line_reachable(&report, &src, 3) && line_reachable(&report, &src, 4),
            "both defun bodies must be measurable; cov={report:?}"
        );
        assert!(
            line_covered(&report, &src, 4),
            "pick-hot's line must be covered (unique wrapper node); cov={report:?}"
        );
        assert!(
            !line_covered(&report, &src, 3),
            "the shared (* v 3) must not credit pick-cold's line; cov={report:?}"
        );
    }

    // ---- --optimize on a classic source must refuse, not report a vacuous 100% ----
    #[test]
    fn optimize_with_classic_source_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "classic.clsp", CLASSIC_A);
        let includes = vec![dir.path().to_string_lossy().to_string()];
        let err = compile_source(&src, &includes, true).err().unwrap();
        assert!(
            err.contains("--optimize") && err.contains("classic"),
            "refusal must say what and why; err={err}"
        );
        // The guard is dialect-independent: --optimize routes even a cl-25 source through the
        // classic finalizer, which strips srclocs -- refused too, not silently 100%.
        let modern = write_source(&dir, "modern.clsp", CANARY_A);
        let err2 = compile_source(&modern, &includes, true).err().unwrap();
        assert!(
            err2.contains("collapsed"),
            "collapse refusal must be reported; err={err2}"
        );
        // Without --optimize the cl-25 source keeps a dense, line-bearing srcloc table.
        let cs = match compile_source(&modern, &includes, false) {
            Ok(cs) => cs,
            Err(e) => panic!("cl-25 without --optimize must compile: {e}"),
        };
        assert!(
            cs.reachable.len() > 1,
            "cl-25 srcloc table must be dense; reachable={}",
            cs.reachable.len()
        );
    }

    // ---- the unattributed note must say WHICH record failed to attribute ----
    #[test]
    fn unattributed_note_identifies_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_source(&dir, "known.clsp", CANARY_A);
        let root = dir.path().to_string_lossy().to_string();
        let includes = vec![root.clone()];
        let (known_hex, _) = compile_to_hex(&src, &includes);

        // A foreign program no --source compile produced: nothing decorates, every executed
        // node is unattributed.
        let foreign_dir = tempfile::tempdir().unwrap();
        let foreign = write_source(
            &foreign_dir,
            "foreign.clsp",
            "(mod (X)\n  (include *standard-cl-25*)\n  (* X 7919)\n)\n",
        );
        let f_includes = vec![foreign_dir.path().to_string_lossy().to_string()];
        let (foreign_hex, _) = compile_to_hex(&foreign, &f_includes);

        let records = vec![
            CorpusRecord {
                program: known_hex,
                args: "ff0380".to_string(), // (3)
                source: None,
                layer: "e2e".to_string(),
            },
            CorpusRecord {
                program: foreign_hex,
                args: "ff0380".to_string(),
                source: None,
                layer: "e2e".to_string(),
            },
        ];
        let report = run_corpus(&records, &root, &includes, false).unwrap();
        assert!(
            report.unattributed_records.contains(&2),
            "record 2 (the foreign program) must be named; report={report:?}"
        );
        assert!(
            !report.unattributed_records.contains(&1),
            "record 1 attributes cleanly and must not be named; report={report:?}"
        );
    }
}
