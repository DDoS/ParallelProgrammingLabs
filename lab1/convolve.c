#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "wm.h"

unsigned char toUnsignedCharSaturated(float v) {
    if (v < 0) {
        return 0;
    }
    if (v > 0xFF) {
        return 0xFF;
    }
    return (char) v;
}

void transform(unsigned char **image, unsigned *width, unsigned *height, unsigned threadCount) {
    // Original image
    unsigned char *imageIn = *image;
    unsigned widthIn = *width;
    unsigned heightIn = *height;
    // Calculate the output width and height, this is integer math so "i / 2 * 2" isn't redundant
    unsigned padding = WEIGHT_MATRIX_SIZE / 2;
    unsigned widthOut = widthIn - padding * 2;
    unsigned heightOut = heightIn - padding * 2;
    // There is one sum per pixel in the output image
    unsigned sumCount = widthOut * heightOut;
    // Allocate the output image
    unsigned char *imageOut = malloc(sizeof(unsigned char) * sumCount * 4);
    if (imageOut == NULL) {
        printf("Could not allocate new image\n");
        exit(-1);
    }
    // Parallelize the for loop
    #pragma omp parallel for num_threads(threadCount)
    for (unsigned sumIndex = 0; sumIndex < sumCount; sumIndex++) {
        // Calculate the sum coordinates in the output image
        unsigned xSum = sumIndex % widthOut;
        unsigned ySum = sumIndex / widthOut;
        // Calculate the sum top left pixel in the input image
        unsigned char *sum = imageIn + (xSum + ySum * widthIn) * 4;
        // Calculate the weighted component sums in the original image
        float sumR = 0, sumG = 0, sumB = 0;
        for (unsigned y = 0; y < WEIGHT_MATRIX_SIZE; y++) {
            unsigned char *line = sum + y * widthIn * 4;
            float *weightLine = weightMatrix[y];
            for (unsigned x = 0; x < WEIGHT_MATRIX_SIZE; x++) {
                unsigned char *pixel = line + x * 4;
                float weight = weightLine[x];
                sumR += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumB += pixel[2] * weight;
            }
        }
        // Calculate the output image pixel
        unsigned char* pixel = imageOut + sumIndex * 4;
        // Set the weighted sums in the output image
        pixel[0] = toUnsignedCharSaturated(sumR);
        pixel[1] = toUnsignedCharSaturated(sumG);
        pixel[2] = toUnsignedCharSaturated(sumB);
        pixel[3] = 0xFF;
    }
    // Delete the input image
    free(imageIn);
    // Replace it with the output one
    *image = imageOut;
    *width = widthOut;
    *height = heightOut;
}
