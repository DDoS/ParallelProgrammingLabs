CC = nvcc
CCFLAGS =

.PHONY: cleanObjs clean

default: rectify pool convolve

# compile lodepng
lodepng.o: lodepng.cu lodepng.h
	$(CC) $(CCFLAGS) -c lodepng.cu

# compile transform without the implementation
transform.o: transform.cu transform.h lodepng.h
	$(CC) $(CCFLAGS) -c transform.cu

# compile the implementations of transform
rectify.o: rectify.cu transform.h
	$(CC) $(CCFLAGS) -c rectify.cu

pool.o: pool.cu transform.h
	$(CC) $(CCFLAGS) -c pool.cu

convolve.o: convolve.cu transform.h wm.h
	$(CC) $(CCFLAGS) -c convolve.cu

# create and executable for each transform
rectify: rectify.o transform.o lodepng.o
	$(CC) $(CCFLAGS) -o rectify lodepng.o transform.o rectify.o

pool: pool.o transform.o lodepng.o
	$(CC) $(CCFLAGS) -o pool lodepng.o transform.o pool.o

convolve: convolve.o transform.o lodepng.o
	$(CC) $(CCFLAGS) -o convolve lodepng.o transform.o convolve.o

cleanObjs:
	rm -f *.o

clean: cleanObjs
	rm -f rectify
	rm -f pool
	rm -f convolve
