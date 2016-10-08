C = gcc

lodepng.o: lodepng.c lodepng.h
	$(C) -c lodepng.c

rectify: lodepng.o rectify.c
	$(C) -o rectify lodepng.o rectify.c

clean:
	rm -f lodepng.o
	rm -f rectify
