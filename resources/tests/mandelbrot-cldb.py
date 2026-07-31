#!/usr/bin/env python

import subprocess
import argparse
import yaml
import io

parser = argparse.ArgumentParser()
parser.add_argument('-sigil', default=None)
args = parser.parse_args()

infile = './mandelbrot/mandelbrot.clsp'
if args.sigil:
    mb = open(infile).read()
    infile = f'{infile}.sigil'
    with open(infile, 'w') as outf:
        outf.write(mb.replace('*standard-cl-21*', args.sigil))

command = ['../../target/debug/cldb', '-p', infile, '(-192 -128 -144 -96 8)']
want_output = f"""---
- Print: ((escape-at -152 -104) 14)
- Print: ((escape-at -160 -104) 14)
- Print: ((escape-at -168 -104) 14)
- Print: ((escape-at -176 -104) 11)
- Print: ((escape-at -184 -104) 8)
- Print: ((escape-at -192 -104) 7)
- Print: ((escape-at -152 -112) 14)
- Print: ((escape-at -160 -112) 14)
- Print: ((escape-at -168 -112) 14)
- Print: ((escape-at -176 -112) 14)
- Print: ((escape-at -184 -112) 8)
- Print: ((escape-at -192 -112) 7)
- Print: ((escape-at -152 -120) 14)
- Print: ((escape-at -160 -120) 14)
- Print: ((escape-at -168 -120) 14)
- Print: ((escape-at -176 -120) 13)
- Print: ((escape-at -184 -120) 8)
- Print: ((escape-at -192 -120) 6)
- Print: ((escape-at -152 -128) 12)
- Print: ((escape-at -160 -128) 10)
- Print: ((escape-at -168 -128) 10)
- Print: ((escape-at -176 -128) 7)
- Print: ((escape-at -184 -128) 6)
- Print: ((escape-at -192 -128) 5)
- Print: "(\\"result\\" \\"||567AAC|68DEEE|78EEEE|78BEEE\\")"
- Final: "3356114000950459963475899699747220812557867594760040767593731831711045"
  Final-Location: "{infile}(8):43"
"""

have_output = subprocess.check_output(command).decode('utf8')
parsed_have = yaml.safe_load(io.StringIO(have_output))
parsed_want = yaml.safe_load(io.StringIO(want_output))

def dequote(kv):
    result = {}
    for k in kv.keys():
        result[k] = kv[k].replace('"','')
    return result

if args.sigil:
    if 'Final-Location' in parsed_have[-1]:
        del parsed_have[-1]['Final-Location']
    del parsed_want[-1]['Final-Location']
    parsed_have = list(map(dequote, parsed_have))
    parsed_want = list(map(dequote, parsed_want))

assert parsed_have == parsed_want
