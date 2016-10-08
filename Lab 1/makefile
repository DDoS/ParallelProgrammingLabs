UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	# Because "gcc" on macOS is actually Clang which doesn't support "-fopenmp"
	# Define the C compiler to actually be GCC (which must be installed)
    CC = gcc-6
else
	CC = gcc
endif

CCFLAGS = -std=c99

OPEN_MP = -fopenmp

default: rectify

lodepng.o: lodepng.c lodepng.h
	$(CC) $(CCFLAGS) -c lodepng.c

rectify.o: rectify.c lodepng.h
	$(CC) $(CCFLAGS) $(OPEN_MP) -c rectify.c

rectify: lodepng.o rectify.o
	$(CC) $(CCFLAGS) $(OPEN_MP) -o rectify lodepng.o rectify.o

test_equality: test_equality.c lodepng.o
	$(CC) $(CCFLAGS) -o test_equality test_equality.c lodepng.o

clean:
	rm -f lodepng.o
	rm -f rectify.o
	rm -f rectify
	rm -f test_equality
