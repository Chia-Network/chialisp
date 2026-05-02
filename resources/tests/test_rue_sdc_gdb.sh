#!/bin/sh

ADDR=127.0.0.1:9001
if [ "x$1" != "x" ] ; then
	ADDR="$1"
fi

gdb-multiarch --batch --ex "set confirm off" --ex "set architecture arm" --ex "file rue_sdc.elf" --ex "source support/gdb_print_sexp.py" --ex "handle SIGUSR1 noprint nostop pass" --ex "dir resources/tests" --ex "target remote ${ADDR}" --ex "break *(factorial + 28)" --ex "continue" --ex "print num" --ex "continue" --ex "print num" --ex "continue" --ex "print num" --ex "continue" --ex "print num" --ex "continue" --ex "print num" --ex "continue" --ex "print num" --ex "delete breakpoints" --ex "continue" 2>&1
