CC = mpicc
CCFLAGS = -std=c99

grid.o: grid.c constant.h node.h
	$(CC) $(CCFLAGS) -c grid.c

# clean the compilation output
clean:
	rm -f *.o
	rm -f grid_4_4
	rm -f grid_512_512
