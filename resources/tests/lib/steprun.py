from typing import List
import binascii
import json
import os
from pathlib import Path
from chialisp import start_clvm_program, compose_run_function, compile_clvm
from clvm_tools.binutils import assemble
from clvm_rs import Program

def compile_module_with_symbols(include_paths: List[Path], source: Path):
    path_obj = Path(source)
    file_path = path_obj.parent
    file_stem = path_obj.stem
    target_file = file_path / (file_stem + ".clvm.hex")
    sym_file = file_path / (file_stem + ".sym")
    # print(f"compile_clvm({path_obj.absolute()}, {str(target_file.absolute().as_posix())}, {include_paths}, True)")
    compile_result = compile_clvm(
        str(path_obj.resolve()), str(target_file.absolute()), [str(p) for p in include_paths], True
    )
    print(f"Writing to {target_file} {compile_result}")
    symbols = compile_result["symbols"]
    if len(symbols) != 0:
        with open(str(sym_file.absolute()), "w") as symfile:
            symfile.write(json.dumps(symbols))


def run_until_end(p):
    last = None
    location = None

    while not p.is_ended():
        step_result = p.step()
        if step_result is not None:
            last = step_result
            if 'Print' in last:
                print(f"{last['Print']}")

    return last

def diag_run_clvm(program, args, symbols, options=None):
    if options is None:
        options = {}
    hex_form_of_program = binascii.hexlify(bytes(program)).decode('utf8')
    hex_form_of_args = binascii.hexlify(bytes(args)).decode('utf8')
    symbols = json.loads(open(symbols).read())
    p = start_clvm_program(hex_form_of_program, hex_form_of_args, symbols, None, options)
    report = run_until_end(p)
    if 'Failure' in report:
        print(report['Failure'])
    else:
        return assemble(report['Final'])

if __name__ == '__main__':
    # smoke test
    import sys
    import argparse
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--print', action='store_true', default=True)
    parser.add_argument('program')
    parser.add_argument('env')
    parser.add_argument('symbols')
    args = parser.parse_args()

    program = Program.fromhex(open(args.program).read())
    env = Program.fromhex(open(args.env).read())
    options = { 'print': args.print }
    diag_run_clvm(program, env, args.symbols, options)
