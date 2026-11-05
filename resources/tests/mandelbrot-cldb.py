#!/usr/bin/env python

import subprocess

command = ['../../target/debug/cldb', '-p', './mandelbrot/mandelbrot.clsp', '(-192 -128 -144 -96 8)']
# The result comes from the then-branch P (column 46), not the equal P in
# the condition (column 43). Macro expansion preserves each token's origin.
want_output = """---
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
  Final-Location: "./mandelbrot/mandelbrot.clsp(8):46"
"""

have_output = subprocess.check_output(command).decode('utf8')
assert have_output == want_output
