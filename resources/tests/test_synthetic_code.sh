#!/bin/sh

./target/debug/armtx -o tsc.elf ./resources/tests/test_synthetic_code.clsp '(2 (q . 2) (q . 3) (q . 4) ())'
