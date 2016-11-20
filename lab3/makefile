CC = nvcc
CCFLAGS =

.PHONY: cleanObjs clean

default: rectify pool convolve
	${MAKE} grid_4_4
	${MAKE} grid_512_512

# compile lodepng
lodepng.o: lodepng.cu lodepng.h
	$(CC) $(CCFLAGS) -c lodepng.cu

# compile transform without the implementation
transform.o: transform.cu transform.h common.h lodepng.h
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

grid_4_4: CCFLAGS += -DN=4
grid_4_4: grid.cu common.h
	$(CC) $(CCFLAGS) -o grid_4_4 grid.cu

grid_512_512: CCFLAGS += -DN=512
grid_512_512: grid.cu common.h
	$(CC) $(CCFLAGS) -o grid_512_512 grid.cu

cleanObjs:
	rm -f *.o

clean: cleanObjs
	rm -f rectify
	rm -f pool
	rm -f convolve
	rm -f grid_4_4
	rm -f grid_512_512
