CC = mpicc
CCFLAGS = -std=c99

grid.o: grid.c grid.h constant.h
	$(CC) $(CCFLAGS) -c grid.c

main.o: main.c grid.h
	$(CC) $(CCFLAGS) -c main.c

grid_4_4: main.o grid.o
	$(CC) $(CCFLAGS) -o grid_4_4 main.o grid.o -lm

# clean the compilation output
clean:
	rm -f *.o
	rm -f grid_4_4
	rm -f grid_512_512
