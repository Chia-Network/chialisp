#!/bin/sh

exec gdb-multiarch --ex "file tap.elf" --ex "source ./support/gdb_print_sexp.py" --ex "target remote :9001"
