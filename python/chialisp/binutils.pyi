def assemble_generic(
    cons: object,
    from_bytes: object,
    args: str,
) -> object: ...
def disassemble_generic(program_bytes: bytes) -> str: ...

__all__ = ["assemble_generic", "disassemble_generic"]
