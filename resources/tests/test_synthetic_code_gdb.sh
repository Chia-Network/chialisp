#!/bin/sh

gdb-multiarch --ex "file tsc.elf" --ex "source support/gdb_print_sexp.py" --ex "dir resources/tests" --ex "target remote :9001"

