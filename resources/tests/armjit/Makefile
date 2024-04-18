ARMGCC ?= arm-non-eabi-gcc

.SUFFIXES: .s .elf

.s.elf:
	arm-none-eabi-gcc -nostartfiles -nodefaultlibs -Ttext=0x1000 -o $@ $^

all: \
	return_nil.elf \
	return_cons.elf

return_nil.elf: return_nil.s

return_cons.elf: return_cons.s
