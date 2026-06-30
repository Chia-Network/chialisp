from collections.abc import Callable
from typing import Any, Literal, TypedDict, final, overload

class CldbError(Exception): ...
class CompError(Exception): ...

@final
class PythonRunStep:
    def is_ended(self) -> bool: ...
    def step(self) -> dict[str, str] | None: ...
    def drop(self) -> None: ...

class _CompileResultWithSymbols(TypedDict):
    output: str
    symbols: dict[str, str]

def get_version() -> str: ...

@overload
def compile_clvm(
    input_path: Any,
    output_path: str,
    search_paths: list[str] = ...,
    export_symbols: None = ...,
) -> str: ...
@overload
def compile_clvm(
    input_path: Any,
    output_path: str,
    search_paths: list[str] = ...,
    export_symbols: Literal[True] = ...,
) -> _CompileResultWithSymbols: ...

@overload
def compile(
    source: str,
    search_paths: list[str] = ...,
    export_symbols: None = ...,
) -> str: ...
@overload
def compile(
    source: str,
    search_paths: list[str] = ...,
    export_symbols: Literal[True] = ...,
) -> _CompileResultWithSymbols: ...

def check_dependencies(input_path: Any, search_paths: list[str] = ...) -> list[str]: ...
def start_clvm_program(
    hex_prog: str,
    hex_args: str,
    symbol_table: dict[str, str] | None,
    overrides: dict[str, Callable[[Any], Any]] | None = ...,
    run_options: dict[str, Any] | None = ...,
) -> PythonRunStep: ...
def launch_tool(tool_name: str, args: list[str], default_stage: int = 2) -> bytes: ...
def call_tool(tool_name: str, args: list[str]) -> bytes: ...
def compose_run_function(
    hex_prog: str,
    symbol_table: dict[str, str],
    function_name: str,
) -> str: ...

from . import binutils as binutils

__all__ = [
    "CldbError",
    "CompError",
    "PythonRunStep",
    "binutils",
    "call_tool",
    "check_dependencies",
    "compile",
    "compile_clvm",
    "compose_run_function",
    "get_version",
    "launch_tool",
    "start_clvm_program",
]
