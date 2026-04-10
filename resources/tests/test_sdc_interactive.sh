#!/bin/sh

gdb-multiarch \
  --ex "set pagination off" \
  --ex "set confirm off" \
  --ex "file sdc.elf" \
  --ex "source support/gdb_print_sexp.py" \
  --ex "dir resources/tests" \
  --ex "target extended-remote :9001" \
  --ex "run" \
  --ex "run" \
  --ex "quit" 2>&1
