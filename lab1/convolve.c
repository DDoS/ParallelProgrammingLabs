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
	unsigned widthOut = widthIn - (WEIGHTED_MATRIX_WIDTH - 1);
    unsigned heightOut = heightIn - (WEIGHTED_MATRIX_HEIGHT - 1);
	
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
		unsigned char addition[4] = {0,0,0,255};
		
        for (unsigned y = 0; y < WEIGHTED_MATRIX_WIDTH * 4; y+=4) {
			unsigned char *line = currentSubMatrix + y * widthIn;
            for (unsigned x = 0; x < WEIGHTED_MATRIX_WIDTH * 4; x+=4) {
				unsigned char *pixel = line + x;
                addition[0] += pixel[0] + w[y/4][x/4];
                addition[1] += pixel[1] + w[y/4][x/4];
                addition[2] += pixel[2] + w[y/4][x/4];
			}
        }	

        for (int pixelIndex = 0; pixelIndex < 4; pixelIndex++){
		    if (addition[pixelIndex] <= 0){
			    addition[pixelIndex] = 0;
		    }
		    else if (addition[pixelIndex] >= 255){
			    addition[pixelIndex] = 255;
            }
        }
		
        unsigned char* pixel = imageOut + subMatrixIndex * 4;
		pixel[0] = addition[0];
        pixel[1] = addition[1];
        pixel[2] = addition[2];
        pixel[3] = addition[3];
		
    }
    // Delete the input image
    free(imageIn);
    // Replace it with the output one
    *image = imageOut;
    *width = widthOut;
    *height = heightOut;
}
