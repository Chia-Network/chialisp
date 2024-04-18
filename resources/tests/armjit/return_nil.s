  .text
  .globl _start
_start:
  sub r0, r0, r0
  add r0, r0, #1
  swi 0

