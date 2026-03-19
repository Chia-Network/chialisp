# Rue HIR Translation Plan (for `run_test_let_star_3_deep_rue`)

## Context

Branch examined: `origin/20260318-alt-cg`.

New test:

- `src/tests/classic/run.rs` -> `run_test_let_star_3_deep_rue`

This test uses the new sigil include:

- `(include *standard-cl-rue1*)`

which routes code generation to the alternate Rue path (`stepping = 1000000`).

Current failure observed on that branch:

- panic at `src/compiler/compiler.rs` in `intern_expr_hir` (`todo!()`).

## Goal

Extend the partial translation path from Chialisp `CompileForm`/`BodyForm` into Rue HIR, then use Rue lowering/codegen to emit equivalent CLVM so the test:

- `run_test_let_star_3_deep_rue`

passes.

## Key blockers in `src/compiler/compiler.rs`

The following stubs currently block end-to-end translation:

- `intern_expr_hir`
- `impl Into<rue_diagnostic::SrcLoc> for Srcloc`
- `param_names_and_paths_`
- `param_names_and_paths`
- `create_param_helper`
- `intern_helper_hir` (multiple `todo!()` sites)
- Rue branch in `compile_from_compileform` (final lowering/codegen `todo!()`)

## Phased implementation plan

### Phase 1: Make expression translation usable for the target test

1. Convert Rue translation helpers to return `Result<_, CompileErr>` where needed.
2. Replace panic stubs with real translation or explicit unsupported-form errors.
3. Implement `intern_expr_hir` for the subset needed by the deep let* test:
   - `BodyForm::Value`:
     - integers/atoms
     - variable references to args/helpers
     - `@` / `@*env*` handling used by let-hoisted forms
   - `BodyForm::Quoted`:
     - recursive `SExp` literal conversion into Rue HIR literals/pairs
   - `BodyForm::Call`:
     - primitive mappings required by test path (`+`, `*`, and any structural ops introduced by desugaring)
     - helper call translation (`Hir::FunctionCall`)
   - Other variants can initially return explicit unsupported errors (no panic).
4. Implement `Srcloc` -> `rue_diagnostic::SrcLoc` conversion with stable file/span mapping.

### Phase 2: Helper/function interning and symbol resolution

5. Complete `intern_helper_hir` for `HelperForm::Defun`:
   - register function symbols in scope before/while translating bodies
   - support inline defuns first (most relevant for desugared let* path)
   - keep non-inline arg-tree destructuring either implemented or as explicit temporary error
6. Ensure symbol tables/scopes are populated so references resolve during expression interning.
7. Update `intern_hir` to produce a proper Rue main symbol entrypoint (not only a raw expression id), because Rue lowering is symbol-oriented.

### Phase 3: Wire Rue lowering + CLVM emission

8. Replace Rue `todo!()` in `compile_from_compileform` with real pipeline:
   - build dependency graph
   - lower HIR -> LIR
   - optional LIR optimization
   - `rue_lir::codegen` to CLVM node
   - convert CLVM node back to project `SExp` (`convert_from_clvm_rs`) for return
9. Add direct crate imports/dependencies required by this path (`rue-lir`, `rue-options`, `id-arena`) if not already available in compile unit.

### Phase 4: Test loop and expansion

10. Run:

    - `cargo test run_test_let_star_3_deep_rue -- --nocapture`

11. Iterate on concrete failures in order (name resolution, env path semantics, missing op mappings).
12. Once green, run nearby regression checks to ensure default/classic path remains unaffected.

## Implementation notes

- Prioritize replacing all `todo!()` panics in the Rue path with deterministic behavior or structured errors.
- Keep initial scope focused on the deep let* test path; broaden coverage after this target is green.
- The parameter helper functions (`param_names_and_paths*`, `create_param_helper`) are likely the next major expansion for non-inline destructuring support.
