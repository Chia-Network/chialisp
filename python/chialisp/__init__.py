import sys
from typing import TypedDict

from chialisp._chialisp import (
    CldbError,
    CompError,
    PythonRunStep,
    binutils as _binutils,
    call_tool,
    check_dependencies,
    compile,
    compile_clvm,
    compile_debug,
    compose_run_function,
    get_version,
    launch_tool,
    start_clvm_program,
)

sys.modules[f"{__name__}.binutils"] = _binutils
binutils = _binutils


class DebugCompileArtifact(TypedDict):
    export_name: str | None
    program: bytes
    debug: bytes
    symbols: dict[str, str]


class DebugCompileResult(TypedDict):
    artifacts: list[DebugCompileArtifact]


__all__ = [
    "CldbError",
    "CompError",
    "DebugCompileArtifact",
    "DebugCompileResult",
    "PythonRunStep",
    "binutils",
    "call_tool",
    "check_dependencies",
    "compile",
    "compile_clvm",
    "compile_debug",
    "compose_run_function",
    "get_version",
    "launch_tool",
    "start_clvm_program",
]
