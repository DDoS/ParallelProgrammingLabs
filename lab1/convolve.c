#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "wm.h"

#define WEIGHTED_MATRIX_WIDTH  3
#define WEIGHTED_MATRIX_HEIGHT  3

void transform(unsigned char **image, unsigned *width, unsigned *height, unsigned threadCount) {
	unsigned char *imageIn = *image;
    unsigned widthIn = *width;
    unsigned heightIn = *height;
	
	//Calculate the output width and height
	unsigned widthOut = *width - (WEIGHTED_MATRIX_WIDTH - 1);
    unsigned heightOut = *height - (WEIGHTED_MATRIX_HEIGHT - 1);
	
	//Calculate the number of sub matrix in the main matrix
	unsigned totalSubMatrices = widthOut * heightOut;
	
	//Create output image
	unsigned char *imageOut = malloc(sizeof(unsigned char) * totalSubMatrices * 4);
    if (imageOut == NULL) {
        printf("Could not allocate new image\n");
        exit(-1);
    }
	
    // Parallelize the for loop
    #pragma omp parallel for num_threads(threadCount)
	for (unsigned subMatrixIndex = 0; subMatrixIndex < totalSubMatrices; subMatrixIndex++) {
        
		unsigned xSubMatrix = subMatrixIndex % widthOut;
        unsigned ySubMatrix = subMatrixIndex / widthOut;
		
		unsigned char *currentSubMatrix = imageIn + (xSubMatrix * WEIGHTED_MATRIX_WIDTH +  ySubMatrix * widthIn) * 4;
		int addition = 0;
		
        for (unsigned x = 0; x < WEIGHTED_MATRIX_WIDTH; x++) {
			for (unsigned y = 0; y < WEIGHTED_MATRIX_HEIGHT; y++) {
				addition += *(currentSubMatrix + (x * 4)) * w[x][y];
			}
        }
		
		if (addition <= 0){
			addition = 0;
		}
		else if (addition >= 255){
			addition = 255;
		}
		
		unsigned char additionConverted = addition + '0';
		
		*(imageOut + xSubMatrix + ySubMatrix * widthOut)  = additionConverted;
		
    }
    // Delete the input image
    free(imageIn);
    // Replace it with the output one
    *image = imageOut;
    *width = widthOut;
    *height = heightOut;
}
