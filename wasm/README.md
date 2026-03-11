Build
-----

Clone GitHub repository
```bash
git clone https://github.com/Chia-Network/chialisp
cd chialisp/wasm
```

Use `wasm-pack` to build the wasm `pkg` file used with npm. Install it with:

```bash
cargo install wasm-pack
```

Then build with

```bash
# Make sure you're at <chialisp root>/wasm
wasm-pack build --release --target=nodejs
```

Test
-----
Prerequisite:
- NodeJS >= 18
- Wasm files built by `wasm-pack` command exist at `<chialisp root>/wasm/pkg/`

```bash
# Make sure you're at <chialisp root>/wasm
node ./tests/index.js
```

Program
===

Program is exported by ```chialisp``` and contains a ```to``` function
among a few others.  Its use is very like Program.to in the python code and
similar to chiaminejp's ```clvm_tools``` library.  It produces a value that
can be used together with other such values, can be curried (and uncurried)
converted to the hex representation and run.

```Program.to(javascript_value)```

Converts javascript values to SExp objects which can be used together and run
as CLVM programs or used as program arguments.  This conversion follows simple
conventions that were established in ```clvm_tools```.

- There's a tuple object returned by the ```t``` function (2 arguments) which
produces a cons.

- javascript arrays are treated as linear proper lists.  Each element appears
as the first of a cons with the rest of the converted list as its tail.  The
list is terminated by a nil.

- an object which has a serialize method treats the result of ```o.serialize()```
as an array-like object which specifies the byte values of the object's atom
representation.  This covers bls primitives such as G1Element and G2Element.

- javascript numbers, bignums, strings and bools are treated as atoms.

- javascript objects which contain an array-like ```pair``` member are treated
the same as tuple objects above.

```Program.from_hex(hex_str)```

Converts a string of pairs of hex digits into the CLVM deserialized form of the
object.

```Program.null()```

Returns a null object.

The returned objects have these methods:

SExp methods
===

```SExp.toString()```

Convert the object to its hex representation.

```SExp.as_pair()```

If it is a cons, return a tuple-compatible object containing a ```pair``` array
with 2 elements, otherwise null.

```SExp.listp()```

Return true if the object is a cons.

```SExp.nullp()```

Return true if the object is falsey.

```SExp.as_int()```

Returns a javascript number that fits within the 32-bit integers representing the object's atom value, or throw.

```SExp.as_bigint()```

Returns a javascript big number representing the value of the given atom or throw.

```SExp.first()```

If the object is a cons, return its first or left component, or throw if not.

```SExp.rest()```

If the object is a cons, return its rest or right component, or throw if not.

```SExp.cons(other)```

Creates an SExp which is a cons of this sexp and other.

```SExp.run(env)```

Runs the indicated SExp as a program with the given environment.

```SExp.as_bin()```

Serialize the object into an array of byte values.

```SExp.list_len()```

Give the number of conses one needs to traverse until reaching a non-cons rest.
For a proper list, this gives the list's length.

```SExp.as_javascript()```

Return a javascript value that allows the given SExp to be inspected via
javascript.

```SExp.curry(a, b, c ...)```

Given a number of positional arguments, build a curried application that provides
values for the left arguments of some runnable CLVM code, giving code that can
be correctly called with fewer arguments.  This is common for providing values to
the upper case parameters of chialisp programs, such as coin puzzles.

```SExp.uncurry(program) -> [inner_program, [args...]]```
```SExp.uncurry_error(program)```

Uncurry returns an array with the inner program and the retrievable arguments
separated out, or the original program and null.  uncurry_error throws instead
of returning a value if the object wasn't a curried program.

