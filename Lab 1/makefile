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

default: rectify pool convolve test_equality

# compile lodepng
lodepng.o: lodepng.c lodepng.h
	$(CC) $(CCFLAGS) -c lodepng.c

# compile transform without the implementation
transform.o: transform.c lodepng.h
	$(CC) $(CCFLAGS) -c transform.c

# compile the implementations of transform
rectify.o: rectify.c
	$(CC) $(CCFLAGS) $(OPEN_MP) -c rectify.c

pool.o: pool.c
	$(CC) $(CCFLAGS) $(OPEN_MP) -c pool.c

convolve.o: convolve.c
	$(CC) $(CCFLAGS) $(OPEN_MP) -c convolve.c

# create and executable for each transform
rectify: rectify.o transform.o lodepng.o
	$(CC) $(CCFLAGS) $(OPEN_MP) -o rectify lodepng.o transform.o rectify.o

pool: pool.o transform.o lodepng.o
	$(CC) $(CCFLAGS) $(OPEN_MP) -o pool lodepng.o transform.o pool.o

convolve: convolve.o transform.o lodepng.o
	$(CC) $(CCFLAGS) $(OPEN_MP) -o convolve lodepng.o transform.o convolve.o

# compile the equality test
test_equality: test_equality.c lodepng.o
	$(CC) $(CCFLAGS) -o test_equality test_equality.c lodepng.o

# clean the compilation output
clean:
	rm -f *.o
	rm -f rectify
	rm -f pool
	rm -f convolve
	rm -f test_equality
