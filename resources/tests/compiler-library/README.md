# Compiler library regressions

These source fixtures and the tests in `src/tests/compiler/library` were copied
from [chia-gaming at 6bc98161](https://github.com/Chia-Network/chia-gaming/tree/6bc98161f106acd96ab888f9bc1f5a951b883e35).
They are a fixed compiler regression corpus, not a separately maintained standard
library. No Gaming checkout, generated puzzle, or downstream dependency is needed.
The normal Cargo test suite recompiles the sources with the compiler under test
and executes their exported programs. Module outputs stay in memory.

The corpus covers module imports, captured lambdas, recursive list algorithms,
macro expansion and hygiene, balanced trees, `&rest`, safe tree-path selection,
compile-time rejection, and constructor runtime costs. The `shapes` and `pack`
cases preserve Gaming's library assertions; Gaming retains those tests and
libraries because its games use them.

The standalone smoke tests, map/enumerate fixture, and unused prepend, range,
reverse, sort, and permutations fixtures move here from Gaming. Old bare includes
are converted to module imports to use the current library sources. The list
fixtures now have asserted results in Rust, including empty lists, duplicates,
permutation counts, and sorting every permutation.

Game/referee integration tests remain Gaming's responsibility. Its poker hand
fixtures and game libraries are not removed by this migration. Existing historical
compiler fixtures elsewhere in this repository also remain unchanged.

Run `cargo test tests::compiler::library` (or add `--release`).
