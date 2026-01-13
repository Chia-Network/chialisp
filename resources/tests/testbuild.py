import os
from pathlib import Path
import binascii
import json
from clvm_tools.binutils import assemble, disassemble
from chialisp import start_clvm_program, compose_run_function, compile_clvm
from chia_rs import run_chia_program, tree_hash

def compute_output_file_names(source):
    path_obj = Path(source)
    file_path = path_obj.parent
    file_stem = path_obj.stem
    return (file_path / (file_stem + ".clvm.hex"), file_path / (file_stem + ".sym"))

def get_program_hash(hexfile):
    return tree_hash(binascii.unhexlify(open(hexfile).read().strip()))

def compile_module_with_symbols(include_paths,source):
    (hex_file, sym_file) = compute_output_file_names(source)
    compile_result = compile_clvm(source, str(hex_file.absolute()), include_paths, True)
    symbols = compile_result['symbols']
    if len(symbols) != 0:
        with open(str(sym_file.absolute()),'w') as symfile:
            symfile.write(json.dumps(symbols))

def compile_programs(programs):
    for p in programs:
        hex_file, sym_file = compute_output_file_names(p[0])
        prev_hex_file = str(hex_file.absolute()) + '.reference'
        prev_sym_file = str(sym_file.absolute()) + '.reference'
        prev_symbols = json.loads(open(prev_sym_file).read())
        prev_hash = get_program_hash(prev_hex_file)

        try:
            os.unlink(hex_file)
        except:
            pass

        compile_module_with_symbols(p[1], p[0])
        curr_hash = get_program_hash(hex_file)
        curr_symbols = json.loads(open(sym_file).read())

        if prev_hash != curr_hash:
            print(f'compiling {p[0]} got a different hash')
            assert prev_hash == curr_hash

        # Things might be added, but we should produce the same basic
        # outline.
        for hash_match, data in prev_symbols.items():
            # Be lenient about line number info from generated sources.
            if '*' in data:
                continue

            assert hash_match in curr_symbols
            assert curr_symbols[hash_match] == data
