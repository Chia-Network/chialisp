import binascii
import chialisp
from pathlib import Path

classic_code = """
(mod (N X)
  (defun F (N X) (if N (sha256 (F (- N 1) X)) X))
  (F N X)
  )
"""
expected_classic = "ff02ffff01ff02ff02ffff04ff02ffff04ff05ffff04ff0bff8080808080ffff04ffff01ff02ffff03ff05ffff01ff0bffff02ff02ffff04ff02ffff04ffff11ff05ffff010180ffff04ff0bff808080808080ffff010b80ff0180ff018080"

cl23_code = """
(mod (N X)
  (include *standard-cl-23*)
  (defun F (N X) (if N (sha256 (F (- N 1) X)) X))
  (F N X)
  )
"""
expected_cl23 = "ff02ffff01ff02ff02ffff04ff02ffff04ff05ffff04ff0bff8080808080ffff04ffff01ff02ffff03ff05ffff01ff0bffff02ff02ffff04ff02ffff04ffff11ff05ffff010180ffff04ff0bff808080808080ffff010b80ff0180ff018080"

expected_deinline = "ff02ffff01ff02ff02ffff04ff02ffff04ff03ffff04ffff10ff05ffff01830f424080ff8080808080ffff04ffff01ff10ff0bff0bff0bff0bff0bff0b80ff018080"

classic_error = "(mod (X) (xxx X))"
cl23_error = "(mod (X) (include *standard-cl-23*) (+ X X1))"

compiled_code = chialisp.compile(
    classic_code,
    ["."],
    True
)
# Classic outputs symbols to the filesystem down all routes, which likely should
# be fixed.
assert compiled_code["output"] == expected_classic

compiled_code = chialisp.compile(
    classic_code,
    ["."]
)
assert compiled_code == expected_classic

# Verify modern compilation
compiled_code = chialisp.compile(
    cl23_code,
    ["."],
    True
)
symbols = compiled_code["symbols"]
assert symbols["__chia__main_arguments"] == "(N X)"
assert symbols["30960d7f2ddc7188a6428a11d39a13ff70d308e6cc571ffb6ed5ec8dbe4376c0_arguments"] == "(N X)"
assert symbols["30960d7f2ddc7188a6428a11d39a13ff70d308e6cc571ffb6ed5ec8dbe4376c0"] == "F"
assert compiled_code["output"] == expected_cl23

# Check compilation with a path
test_path = Path(__file__).parent

output_file = "simple_deinline_case_23.hex"
compiled_code = chialisp.compile_clvm(
    str(test_path / "simple_deinline_case_23.clsp"),
    output_file,
    ["."],
    True
)
assert compiled_code["symbols"]["d623cecd87575189eb1518b50cecc8944a51aa6f4bb4cf6419f70e4aa34f5a20"].startswith("letbinding")
assert compiled_code["output"] == output_file
assert open(output_file).read().strip() == expected_deinline

# Check dependency output
game_referee = Path(__file__).parent / "game-referee-in-cl23"
dependencies = chialisp.check_dependencies(
    str(game_referee / "test_reverse.clsp"),
    [str(game_referee)]
)
assert len(dependencies) == 1
assert Path(dependencies[0]) == game_referee / 'reverse.clinc'

# Better error reporting
try:
    chialisp.compile(classic_error, [])
    assert False
except Exception as e:
    assert e.args[0] == "error can't compile (\"xxx\" 88), unknown operator compiling (\"xxx\" 88)"

try:
    chialisp.compile(cl23_error, [])
    assert False
except Exception as e:
    assert e.args[0] == "*inline*(1):42-*inline*(1):44: Unbound use of X1 as a variable name"
