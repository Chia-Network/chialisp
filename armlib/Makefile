all: armlib

%.o: %.c
	arm-none-eabi-gcc -Iinclude -c $^

clean:
	rm -f *.o armlib

armlib: entry.o support.o sha256.o
	arm-none-eabi-ld -Tlink.ld -o $@ $^
