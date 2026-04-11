#!/bin/sh

./target/debug/armtx -o mandelbrot.elf ./resources/tests/mandelbrot/mandelbrot.clsp '(100 100 200 200 20)'
