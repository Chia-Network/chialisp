#!/bin/sh

exec ./target/debug/armtx -o tap.elf ./resources/tests/test_assign_path_opt.clsp '((1 1 1 1 1 (1 (2 3 4 5))))'
