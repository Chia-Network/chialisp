#!/bin/sh

gdb-multiarch --ex "file sdc.elf" --ex "source support/gdb_print_sexp.py" --ex "dir resources/tests" --ex "target remote :9001"

