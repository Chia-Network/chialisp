#!/bin/sh

gdb-multiarch --ex "set confirm off" --ex "file sdc.elf" --ex "source support/gdb_print_sexp.py" --ex "dir resources/tests" --ex "target remote :9001" --ex "continue" --ex "quit 0" 2>&1
