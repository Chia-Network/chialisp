# Structural CLVM debug metadata, version 1

This document specifies the metadata file format independently of any compiler,
programming language, or serialization library. It describes version 1, introduced
in [Chialisp #579](https://github.com/Chia-Network/chialisp/pull/579).
**MUST**, **SHOULD**, and **MAY** describe requirements, recommendations, and
permitted choices, respectively.

A metadata file accompanies a serialized CLVM program. It contains source text,
names, parameter information, and a *shadow tree*: the program's pairs and atom
bytes with annotations attached to individual occurrences. It does not change
program execution. Missing annotations are valid; the format does not promise
that every runtime frame has a unique name or location.

## Binary encoding and notation

The file MUST contain exactly one ordinary CLVM serialization, with no trailing
bytes. Atoms contain bytes; pairs contain two CLVM values. In this document,
`(A B C)` means a proper list ending in the empty atom, and `(A . B)` means a pair.
Every record and table described below MUST be a proper list. Quoted text denotes
UTF-8 atom bytes, not a particular source-language string syntax.

The ordinary CLVM byte encoding is:

| Value | Encoding (hexadecimal bytes) |
| --- | --- |
| Pair | `ff`, followed by the left value, then the right value |
| Empty atom | `80` |
| One-byte atom whose byte is below `80` | That byte itself |
| Other atom of length `n < 0x40` | One length byte `0x80 \| n`, then the atom bytes |
| Atom of length `n < 0x2000` | Two-byte big-endian integer `0xc000 \| n`, then the atom bytes |
| Atom of length `n < 0x100000` | Three-byte big-endian integer `0xe00000 \| n`, then the atom bytes |
| Atom of length `n < 0x8000000` | Four-byte big-endian integer `0xf0000000 \| n`, then the atom bytes |
| Atom of length `n < 0x400000000` | Five-byte big-endian integer `0xf800000000 \| n`, then the atom bytes |

Writers MUST use the shortest applicable encoding. Readers MAY also accept
nonminimal atom-length prefixes. Back-reference encodings are not part of this
format. There is no unique encoding for equivalent metadata: table order and
ignored extensions can differ.

An **unsigned integer** is an atom containing the shortest nonnegative signed
big-endian CLVM integer: zero is the empty atom; a positive value has a leading
`00` byte only when needed to keep its high bit clear. Thus 127, 128, and 256
have payloads `7f`, `0080`, and `0100`. A payload of `00`, a negative value, or
redundant leading zero bytes MUST be rejected in an integer field. These rules
do not apply to arbitrary program atom bytes.

All tags, versions, indices, coordinates, flags, lengths, and parameter paths
below are unsigned integers. The format does not depend on a host's machine-word
size. Implementations MAY reject values, sizes, or nesting depths beyond their
resource limits; they MUST NOT truncate or wrap integers. The Chialisp reader
stores most numeric fields in a host-sized unsigned integer (32 bits on wasm32,
64 bits on typical native builds), while parameter paths retain arbitrary width.

## Top-level record

```text
("CHIALISP_DEBUG" VERSION COORDINATES PROGRAM_SHA256
  FILES STRINGS SPANS FUNCTIONS SHADOW_TREE)
```

| Field | Meaning |
| --- | --- |
| Magic | The 14 ASCII bytes `CHIALISP_DEBUG` |
| `VERSION` | `1`; the compatibility version of this metadata file |
| `COORDINATES` | `(1 8)`, specifying the coordinate convention below |
| `PROGRAM_SHA256` | Exactly 32 raw bytes, SHA-256 of the associated serialized program bytes |
| `FILES` | List of source file records |
| `STRINGS` | List of UTF-8 atoms used for names and labels |
| `SPANS` | List of source span records |
| `FUNCTIONS` | List of function records |
| `SHADOW_TREE` | One annotated program tree |

The digest is **not** a CLVM tree hash or hexadecimal text. It identifies exact
serialized bytes; serializations of the same tree can have different digests.
The digest provides an association, not authentication.

Tables are indexed in list order. Tables MAY be empty or contain duplicate
entries; readers MUST NOT require sorted or deduplicated tables. Every used index
MUST refer to an existing entry. Required references (`FILE`, `FUNCTION_NAME`,
`NAME`) are **zero-based**. Only the optional references on shadow nodes use
**index + 1**, with zero meaning absent. This distinction is part of the format.

## Source files and spans

A file record is `(PATH SOURCE)`. Both fields MUST be valid UTF-8 atoms. `SOURCE`
contains the source text, including its original line endings; `PATH` is a display
identifier and need not name a readable local file. Readers do not need filesystem
access to display the source. No Unicode normalization is implied.

A span record is `(FILE START_LINE START_COLUMN END_LINE END_COLUMN)`. `FILE`
indexes `FILES`. All four coordinates MUST be positive. The end is exclusive.
Producers SHOULD supply ordered coordinates within the embedded source; readers
MUST tolerate unavailable or out-of-range excerpts without requiring source
reconstruction. The Chialisp decoder validates positivity and the file reference,
but does not validate span ordering or source bounds.

Coordinates follow these exact rules:

- Lines start at 1; LF starts a new line. A CR immediately preceding LF is part
  of the line ending and is omitted when displaying an excerpt.
- Columns start at 1. Each Unicode scalar other than a tab advances the column
  by 1. These are neither UTF-8 byte offsets nor grapheme or terminal-cell counts:
  combining characters and wide characters each count as one scalar.
- A tab at column `c` advances to `8 * (floor(c / 8) + 1)`. A leading tab moves
  from column 1 to 8; a tab at column 8 moves to 16.

For example, in `λ\tX\r\n`, `λ` starts at column 1 and `X` at column 8.
The span for `X` is `(FILE 1 8 1 9)`.

## Functions, parameters, and environments

A function record is `(FUNCTION_NAME LEFT_ENV PARAMETER_TREE)`.
`FUNCTION_NAME` indexes `STRINGS`. `LEFT_ENV` MUST be 0 or 1. A value of 1 means
the function environment is `(CAPTURED_LEXICAL_ENV . DECLARED_ARGUMENTS)`:
the first environment element is compiler-internal and is omitted when displaying
declared arguments. A value of 0 means the parameter pattern describes the whole
argument environment.

Parameter trees describe formal argument patterns:

| Record | Meaning |
| --- | --- |
| `(0)` | Empty parameter pattern |
| `(1 NAME PATH CONSTRAINT)` | Named parameter; `NAME` indexes `STRINGS` |
| `(2 LEFT RIGHT)` | Pair of parameter patterns, supporting destructuring and rest arguments |

`PATH` MUST be a positive canonical CLVM integer atom. It selects the parameter
from the full environment of the uncurried function, including any captured
lexical environment. Starting at that environment, read the path's bits from
least significant upward: 0 selects the left child, 1 selects the right child;
stop at the final leading 1. Thus 1 is the whole environment, 2 its left child,
3 its right child, and 5 the left child of its right child. Paths can exceed
machine-word width.

For a single parameter list `(X)`, the pattern is
`(2 (1 X_INDEX 2 CONSTRAINT) (0))`. With `LEFT_ENV = 1`, the pattern `(X . REST)`
uses paths 5 for `X` and 7 for `REST`: the paths include the captured environment,
but the pattern describes only declared arguments. Producers SHOULD keep paths
consistent with the pattern and `LEFT_ENV`; readers need not prove that consistency.

Curried values precede runtime arguments when reconstructing the base function's
environment. A debugger can retain whether each value was bound by a curry or
supplied at runtime. Parameter patterns and constraints are diagnostic metadata:
a malformed call or failing program can supply values that do not fit them, and
readers MUST NOT assume that such values satisfy the advertised constraints.

### Parameter constraints

| Record | Diagnostic interpretation |
| --- | --- |
| `(0)` | Unknown; no constraint |
| `(1)` | Atom interpreted as a signed CLVM integer |
| `(2)` | Atom |
| `(3 LENGTH)` | Atom of exactly `LENGTH` bytes |
| `(4)` | Pair |
| `(5)` | Proper list, including the empty list |
| `(6 (MEMBER ...))` | Union of alternatives |

Writers MUST encode unions as at least two distinct, non-union, known members.
Unknown is not a canonical union member. Members MUST be ordered by numeric tag
(1 through 5), then by increasing `LENGTH` for tag 3. This is a wire rule,
independent of a language's enum declaration order. No subsumption rule removes
overlapping alternatives such as atom and fixed-length bytes.

Readers MUST interpret an unknown constraint tag as unknown and ignore its
remaining fields. For a union, readers MUST decode every member and require at
least two members. Malformed known members still cause rejection. If any member
decodes to unknown, the entire union is unknown; ordering and flattening checks
are then unnecessary. Otherwise readers MUST reject nested unions, duplicate
members, and incorrect ordering.

## Shadow tree and program verification

```text
atom: (0 SPAN LABEL FUNCTION VALUE)
pair: (1 SPAN LABEL FUNCTION LEFT RIGHT)
```

`SPAN`, `LABEL`, and `FUNCTION` are optional references to `SPANS`, `STRINGS`,
and `FUNCTIONS`, respectively, using index + 1. `VALUE` is an atom with the exact
program atom bytes, including any noncanonical integer representation. `LEFT`
and `RIGHT` are shadow nodes. An empty atom and an atom containing byte `00` are
different program values and MUST remain distinguishable.

Removing the annotation fields and replacing each shadow node with its atom or
pair reconstructs the annotated program tree. Labels identify individual nodes;
a function reference marks an executable form, and a span identifies its source
location. Annotations are optional and need not appear on every child. Different
occurrences of identical subtrees MAY carry different annotations. The file does
not specify a runtime tie-breaking algorithm for such occurrences.

**Exact verification** against an associated program requires both:

1. SHA-256 of the supplied serialized bytes equals `PROGRAM_SHA256`.
2. Decoding those bytes yields exactly the shadow tree's pairs and atom bytes.

Decoding a metadata file alone does not perform these checks. A matching digest
alone is insufficient to verify the structural mapping.

Applications MAY associate a file with a recognized wrapper around the annotated
program, retaining the base program's shadow tree and hashing the wrapped
artifact. This is not exact verification. A consumer supporting such associations
must separately check the wrapper relationship before using the annotations.
For example, the Chialisp symbolizer recognizes canonical curry forms:

```text
(2 (1 . BASE_PROGRAM) ENVIRONMENT_EXPRESSION)
ENVIRONMENT_EXPRESSION = 1 | () | (4 (1 . BOUND_VALUE) ENVIRONMENT_EXPRESSION)
```

The tail `1` forwards the runtime environment; `()` discards it. Wrappers can be
nested. A file associated this way still decodes as version 1, but the exact
verification operation correctly fails when its shadow describes only the base.
Symbolization of frames and subprograms is a separate operation from artifact
verification.

## Compatibility and validation

Version numbers apply to the whole metadata file. Readers MUST check the magic
and version before interpreting version-specific fields and MUST reject an
unsupported version. They need not attempt to interpret a later version's schema.
Parsing enough CLVM framing to obtain the version is still necessary.

All records permit additional trailing fields. Readers MUST ignore those fields,
including at the top level, in coordinates, files, spans, functions, parameter
nodes, constraints, and shadow nodes. Additional table entries are entries, not
extension fields. Ignored values may be arbitrary well-formed CLVM values;
readers need not preserve them when writing metadata again.

Unknown constraint tags are handled as described above. Unknown parameter or
shadow-node tags MUST be rejected: their structural meaning cannot be skipped.
Missing required fields, wrong atom/list kinds, invalid UTF-8 in text fields,
invalid integer encodings, out-of-range references, a non-boolean `LEFT_ENV`, an
incorrect hash length, or an unsupported coordinate convention MUST be rejected.

Additive, ignorable fields and constraint hints can keep version 1. A change to
required fields, their meaning, or structural tags that an existing reader cannot
safely interpret requires a new compatibility version. No version negotiation or
migration protocol is implied.

## Conformance vectors

[debug_metadata_v1.json](resources/tests/debug_metadata_v1.json) contains fixed
vectors constructed independently of the Rust encoder. JSON is only the test
container; the metadata files themselves are CLVM bytes in `metadata_hex`.
Hex strings denote raw bytes without a prefix. Integer vectors use atom payloads,
while constraint, program, environment, and metadata vectors use full CLVM
serialization. Decimal integer expectations are strings to avoid JSON precision
limits.

The vectors cover empty annotations, raw byte-zero atoms, source and function
references, captured environments with rest parameters, wrapper association,
integer boundaries, and constraint compatibility. `canonical_hex` records the
constraint encoding after unknown extensions have been discarded. Tests check
fixed bytes and independently specified meanings, not only writer/reader round
trips. Resource-limited implementations may reject integer values beyond their
supported range while preserving the same encoding rules.
