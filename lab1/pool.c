#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "transform.h"

#define POOL_SIZE 2

void transform(unsigned char **image, unsigned *width, unsigned *height, unsigned threadCount) {
    // Original image
    unsigned char *imageIn = *image;
    unsigned widthIn = *width;
    unsigned heightIn = *height;
    // Calculate the pooled image size
    unsigned widthOut = widthIn / POOL_SIZE;
    unsigned heightOut = heightIn / POOL_SIZE;
    // There is one pool per pixel in the output image
    unsigned poolCount = widthOut * heightOut;
    // Allocate the output image
    unsigned char *imageOut = malloc(sizeof(unsigned char) * poolCount * 4);
    if (imageOut == NULL) {
        printf("Could not allocate new image\n");
        exit(-1);
    }
    // Parallelize the for loop
    #pragma omp parallel for num_threads(threadCount)
    for (unsigned poolIndex = 0; poolIndex < poolCount; poolIndex++) {
        // Calculate the pool coordinates
        unsigned xPool = poolIndex % widthOut;
        unsigned yPool = poolIndex / widthOut;
        // Calculate the pool top left pixel in the input image
        unsigned char *pool = imageIn + (yPool * POOL_SIZE * widthIn + xPool * POOL_SIZE) * 4;
        // Calculate the max values in the pool in the original image
        unsigned char maxR = 0, maxG = 0, maxB = 0;
        for (unsigned y = 0; y < POOL_SIZE * 4; y += 4) {
            unsigned char *line = pool + y * widthIn;
            for (unsigned x = 0; x < POOL_SIZE * 4; x += 4) {
                unsigned char *pixel = line + x;
                if (pixel[0] > maxR) {
                    maxR = pixel[0];
                }
                if (pixel[1] > maxG) {
                    maxG = pixel[1];
                }
                if (pixel[2] > maxB) {
                    maxB = pixel[2];
                }
            }
        }
        // Set the max in the new image pool
        unsigned char* pixel = imageOut + poolIndex * 4;
        pixel[0] = maxR;
        pixel[1] = maxG;
        pixel[2] = maxB;
        pixel[3] = 255;
    }
    // Delete the input image
    free(imageIn);
    // Replace it with the output one
    *image = imageOut;
    *width = widthOut;
    *height = heightOut;
}
