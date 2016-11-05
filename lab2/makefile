CC := mpicc
CCFLAGS := -std=c99

.PHONY: cleanObjs clean

default:
	${MAKE} clean
	${MAKE} grid_4_4
	${MAKE} cleanObjs
	${MAKE} grid_512_512

grid_4_4: CCFLAGS += -DN=4
grid_512_512: CCFLAGS += -DN=512

grid.o: grid.c grid.h constant.h
	$(CC) $(CCFLAGS) -c grid.c

main.o: main.c grid.h
	$(CC) $(CCFLAGS) -c main.c

grid_4_4: main.o grid.o
	$(CC) $(CCFLAGS) -o grid_4_4 main.o grid.o -lm

grid_512_512: main.o grid.o
	$(CC) $(CCFLAGS) -o grid_512_512 main.o grid.o -lm

cleanObjs:
	rm -f *.o

clean: cleanObjs
	rm -f grid_4_4
	rm -f grid_512_512
