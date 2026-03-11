# chialisp Changelog

## 0.1.42
### Fixed
- Subtle differences with python representation of disassmebled strings.
- Bug in CSE that would lift common subexpressions above their containing
  forms.

## 0.1.41
### Changed
- Addition of assemble, disassemble and compile from string to python api.
### Fixed
- Issue in parsing lambdas supporting the LSP.

## 0.1.34
### Fixed
- Fixed chialisp compilation issues


## 0.1.33
### Changed
- Set macOS deployment target to 10.14
- Ensure we flush streams in case the runtime system doesn't get a chance
### Fixed
- Fix erroneous detection of recursion when two similar inline siblings


## 0.1.32
Skipped

## 0.1.31 Chia CLVM Tools Rust 2023-04-17

### Added

- defconst was added.
- hierarchial debug was added.
- clvm command linetools: supported more command line features in both compiler front-ends.

## 0.1.35 

- embed-file was added.
- &rest arguments.
- new bls and sec256 operators.

## 0.1.36

- modern lambda added
- updated some internal data strucutres and call interfaces to support env variable renaming at during closure generation / lambda capture, or any step during transformation.

## 0.1.37

- First npm publish with a Program-like object reminiscent of
  chia.types.blockchain_format.program.Program

## 0.1.38

- Uncurry fix, typescript type improvements for npm personality.

## 0.1.39

- Support conversion from Uint8Array to IProgram in wasm.

## 0.1.40

- New language sigil ```*standard-cl-23*``` is here which is the same language as that introduced by ```*strict-cl-21*``` but has much better optimization.
- Syntactic tree lists using #(...) syntax in cl21 and cl23 (matt-o-how).
