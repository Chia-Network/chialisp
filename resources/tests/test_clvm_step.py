#!/usr/bin/env python3
import os
from pathlib import Path
import json
from chialisp import start_clvm_program, compose_run_function

def do_nothing(p):
    pass

def run_until_end(p,printing=do_nothing,spam=True):
    last = None
    location = None

    while not p.is_ended():
        step_result = p.step()
        if step_result is not None:
            last = step_result
            if 'Print' in last:
                printing(last['Print'])
            if 'Operator-Location' in last:
                location = last['Operator-Location']
            if spam:
                print(json.dumps(step_result))

    return (last, location)

def simple_test():
    p = start_clvm_program('ff02ffff01ff02ff02ffff04ff02ffff04ff05ff80808080ffff04ffff01ff02ffff03ffff09ff05ffff010180ffff01ff0101ffff01ff12ff05ffff02ff02ffff04ff02ffff04ffff11ff05ffff010180ff808080808080ff0180ff018080', 'ff0580', {"de3687023fa0a095d65396f59415a859dd46fc84ed00504bf4c9724fca08c9de":"factorial"})

    last, location = run_until_end(p)

    assert int(last['Final']) == 120
    assert location.startswith('factorial')

def complex_test():
    mypath = Path(os.path.abspath(__file__))
    testpath = mypath.parent.joinpath('steprun')
    symbols = json.loads(open(str(testpath.joinpath('fact.sym'))).read())

    def fact_base_override(env):
        print('fact_base_override')
        return 99

    p = start_clvm_program(open(str(testpath.joinpath('fact.clvm.hex'))).read(), 'ff0580', symbols, {
        "fact-base": fact_base_override
    })

    last, location = run_until_end(p)

    assert int(last['Final']) == 11880

def test_with_printing():
    mypath = Path(os.path.abspath(__file__))
    testpath = mypath.parent.joinpath('mandelbrot')
    symbols = json.loads(open(str(testpath.joinpath('mandelbrot.sym'))).read())

    p = start_clvm_program(open(str(testpath.joinpath('mandelbrot.clvm.hex'))).read(), 'ff82ff40ff8180ff82ff70ff81a0ff0880', symbols, {})

    print_outputs = []
    def do_print(p):
        print(p)
        print_outputs.append(p)

    last, location = run_until_end(p,printing=do_print,spam=False)

    expected_outcome = "||567AAC|68DEEE|78EEEE|78BEEE"
    want_prints = [
        "((\"escape-at\" -152 -104) 14)",
        "((\"escape-at\" -160 -104) 14)",
        "((\"escape-at\" -168 -104) 14)",
        "((\"escape-at\" -176 -104) 11)",
        "((\"escape-at\" -184 -104) 8)",
        "((\"escape-at\" -192 -104) 7)",
        "((\"escape-at\" -152 -112) 14)",
        "((\"escape-at\" -160 -112) 14)",
        "((\"escape-at\" -168 -112) 14)",
        "((\"escape-at\" -176 -112) 14)",
        "((\"escape-at\" -184 -112) 8)",
        "((\"escape-at\" -192 -112) 7)",
        "((\"escape-at\" -152 -120) 14)",
        "((\"escape-at\" -160 -120) 14)",
        "((\"escape-at\" -168 -120) 14)",
        "((\"escape-at\" -176 -120) 13)",
        "((\"escape-at\" -184 -120) 8)",
        "((\"escape-at\" -192 -120) 6)",
        "((\"escape-at\" -152 -128) 12)",
        "((\"escape-at\" -160 -128) 10)",
        "((\"escape-at\" -168 -128) 10)",
        "((\"escape-at\" -176 -128) 7)",
        "((\"escape-at\" -184 -128) 6)",
        "((\"escape-at\" -192 -128) 5)",
        f"(\"result\" \"{expected_outcome}\")"
    ]

    assert print_outputs == want_prints
    to_hex = hex(int(last['Final']))
    have_outcome = bytes.fromhex(to_hex[2:]).decode('utf8')
    assert have_outcome == expected_outcome

def single_function_test():
    mypath = Path(os.path.abspath(__file__))
    testpath = mypath.parent.joinpath('steprun')
    symbols = json.loads(open(str(testpath.joinpath('twofun.sym'))).read())

    start_program = open(str(testpath.joinpath('twofun.clvm.hex'))).read();
    torun = compose_run_function(start_program, symbols, "second-fun");
    p = start_clvm_program(torun, 'ff0780', symbols)

    last, location = run_until_end(p)

    assert int(last['Final']) == 12

if __name__ == '__main__':
    simple_test()
    complex_test()
    single_function_test()
    test_with_printing()
