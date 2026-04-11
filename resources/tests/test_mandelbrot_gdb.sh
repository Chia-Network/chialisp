#!/bin/sh

gdb-multiarch --ex "file mandelbrot.elf" --ex "source support/gdb_print_sexp.py" --ex "dir resources/tests/mandelbrot" --ex "target remote :9001"

