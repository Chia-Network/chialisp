  .text
  .globl _start
  .align 4
_start:
  adr r0, testcons
  swi 0

  .align 4
  .globl testcons
testcons:
  .long hi_string
  .long there_string

  .align 4
  .globl hi_string
hi_string:
  .long 5
  .ascii "hi"

  .align 4
  .globl there_string
there_string:
  .long 11
  .ascii "there"
