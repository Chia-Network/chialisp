chialisp
=
![GitHub](https://img.shields.io/github/license/Chia-Network/chialisp?logo=Github)
[![Coverage Status](https://coveralls.io/repos/github/Chia-Network/chialisp/badge.svg?branch=base)](https://coveralls.io/github/Chia-Network/clvm_tools_rs?branch=base)
![Build Crate](https://github.com/Chia-Network/chialisp/actions/workflows/build-crate.yml/badge.svg)
![Build Wheels](https://github.com/Chia-Network/chialisp/actions/workflows/build-test.yml/badge.svg)

![PyPI](https://img.shields.io/pypi/v/chialisp?logo=pypi)
[![Crates.io](https://img.shields.io/crates/v/chialisp.svg)](https://crates.io/crates/clvm_tools_rs)

Theory of operation of the modern compiler: ./HOW_CHIALISP_IS_COMPILED.md
-
This repo can be installed via cargo

    cargo install chialisp

or via pip

    pip install chialisp@git+https://github.com/Chia-Network/clvm_tools_rs.git@e17412032aa7d3b8b1d1f931893fb5802eee626a

Note: `pip` installs a subset of the tools installed by `cargo`, including `brun`, `run`, `opc` and `opd`.


The most current version of the language is in the nightly branch:

    [nightly](https://github.com/Chia-Network/chialisp/tree/nightly)

To install from a specific branch:

    cargo install --no-default-features --git 'https://github.com/Chia-Network/chialisp' --branch nightly
    
To install a git checkout into your current python environment (must be in some kind of venv or conda environment):

    git clone https://github.com/Chia-Network/chialisp
    cd chialisp
    maturin develop

Install from PYPI:

    pip install -i https://pypi.chia.net/nightlies/ chialisp
    
Most people still compile chialisp via python.  One way to set up compilation
in that way is like this:

    import json
    from chialisp import compile_clvm

    def compile_module_with_symbols(include_paths,source):
        path_obj = Path(source)
        file_path = path_obj.parent
        file_stem = path_obj.stem
        target_file = file_path / (file_stem + ".clvm.hex")
        sym_file = file_path / (file_stem + ".sym")
        compile_result = compile_clvm(source, str(target_file.absolute()), include_paths, True)
        symbols = compile_result['symbols']
        if len(symbols) != 0:
            with open(str(sym_file.absolute()),'w') as symfile:
                symfile.write(json.dumps(symbols))

The command line tools provided:

    - run -- Compiles CLVM code from chialisp

    Most commonly, you'll compile chialisp like this:

      ./target/debug/run -O -i include_dir chialisp.clsp
    
    'run' outputs the code resulting from compiling the program, or an error.
    
    - repl -- Accepts chialisp forms and expressions and produces results
              interactively.
              
    Run like:
    
      ./target/debug/repl
      
    Example session:
    
    >>> (defmacro assert items
       (if (r items)
           (list if (f items) (c assert (r items)) (q . (x)))
         (f items)
         )
       )
    (q)
    >>> (assert 1 1 "hello")
    (q . hello)
    >>> (assert 1 0 "bye")
    failed: CompileErr(Srcloc { file: "*macros*", line: 2, col: 26, until: Some(Until { line: 2, col: 82 }) }, "clvm raise in (8) (())")
    >>> 

    - cldb -- Stepwise run chialisp programs with program readable yaml output.
    
      ./target/debug/cldb '(mod (X) (x X))' '(4)'
      ---
      - Arguments: (() (4))
        Operator: "4"
        Operator-Location: "*command*(1):11"
        Result-Location: "*command*(1):11"
        Row: "0"
        Value: (() 4)
      - Env: "4"
        Env-Args: ()
        Operator: "2"
        Operator-Location: "*command*(1):11"
        Result-Location: "*command*(1):13"
        Row: "1"
        Value: "4"
      - Arguments: (4)
        Failure: clvm raise in (8 5) (() 4)
        Failure-Location: "*command*(1):11"
        Operator: "8"
        Operator-Location: "*command*(1):13"

    - brun -- Runs a "binary" program.  Instead of serving as a chialisp
      compiler, instead runs clvm programs.
    
    As 'brun' from the python code:
    
    $ ./target/debug/run '(mod (X) (defun fact (N X) (if (> 2 X) N (fact (* X N) (- X 1)))) (fact 1 X))'
    (a (q 2 2 (c 2 (c (q . 1) (c 5 ())))) (c (q 2 (i (> (q . 2) 11) (q . 5) (q 2 2 (c 2 (c (* 11 5) (c (- 11 (q . 1)) ()))))) 1) 1))
    $ ./target/debug/brun '(a (q 2 2 (c 2 (c (q . 1) (c 5 ())))) (c (q 2 (i (> (q . 2) 11) (q . 5) (q 2 2 (c 2 (c (* 11 5) (c (- 11 (q . 1)) ()))))) 1) 1))' '(5)'
    120
    
    - opc -- crush clvm s-expression form to hex.
    
    As 'opc' from the python code.
    
    opc '(a (q 2 2 (c 2 (c (q . 1) (c 5 ())))) (c (q 2 (i (> (q . 2) 11) (q . 5) (q 2 2 (c 2 (c (* 11 5) (c (- 11 (q . 1)) ()))))) 1) 1))'
    ff02ffff01ff02ff02ffff04ff02ffff04ffff0101ffff04ff05ff8080808080ffff04ffff01ff02ffff03ffff15ffff0102ff0b80ffff0105ffff01ff02ff02ffff04ff02ffff04ffff12ff0bff0580ffff04ffff11ff0bffff010180ff808080808080ff0180ff018080
    
    - opd -- disassemble hex to s-expression form.
    
    As 'opd' from the python code.
    
    opd 'ff02ffff01ff02ff02ffff04ff02ffff04ffff0101ffff04ff05ff8080808080ffff04ffff01ff02ffff03ffff15ffff0102ff0b80ffff0105ffff01ff02ff02ffff04ff02ffff04ffff12ff0bff0580ffff04ffff11ff0bffff010180ff808080808080ff0180ff018080'
    (a (q 2 2 (c 2 (c (q . 1) (c 5 ())))) (c (q 2 (i (> (q . 2) 11) (q . 5) (q 2 2 (c 2 (c (* 11 5) (c (- 11 (q . 1)) ()))))) 1) 1))

History
=

This is a second-hand port of chia's [clvm tools](https://github.com/Chia-Network/clvm_tools/) to rust via the work of
ChiaMineJP porting to typescript.  This would have been a lot harder to
get to where it is without prior work mapping out the types of various
semi-dynamic things (thanks, ChiaMineJP).

Some reasons for doing this are:

 - Chia switched the clvm implementation to rust: [clvm_rs](https://github.com/Chia-Network/clvm_rs), and this code may both pick up speed and track clvm better being in the same language.
 
 - I wrote a new compiler with a simpler, less intricate structure that should be easier to improve and verify in the future in ocaml: [ochialisp](https://github.com/prozacchiwawa/ochialisp).

 - Also it's faster even in this unoptimized form.

All acceptance tests i've brought over so far work, and more are being added.
As of now, I'm not aware of anything that shouldn't be authentic when running
these command line tools from clvm_tools in their equivalents in this repository

 - opc
 
 - opd
 
 - run
 
 - brun

 - repl
 
argparse was ported to javascript and I believe I have faithfully reproduced it
as it is used in cmds, so command line parsing should work similarly in all three
versions.

The directory structure is expected to be:

    src/classic  <-- any ported code with heritage pointing back to
                     the original chia repo.
                    
    src/compiler <-- a newer compiler (ochialisp) with a simpler
                     structure.  Select new style compilation by
                     including a `(include *standard-cl-21*)`
                     form in your toplevel `mod` form.

Mac M1
===

Use ```cargo build --no-default-features``` due to differences in how mac m1 and
other platforms handle python extensions.

Use with chia-blockchain
===

    # Activate your venv, then
    $ maturin develop --release

