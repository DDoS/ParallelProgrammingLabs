#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "lodepng.h"

void rectify(unsigned char *image, unsigned width, unsigned height, unsigned threadCount) {
    #pragma omp parallel for num_threads(threadCount)
    for (unsigned i = 0; i < width * height * 4; i += 4) {
        unsigned char *p = image + i;
        // Modify each component of each pixel, except alpha
        for (unsigned j = 0; j < 3; j++) {
            unsigned char *c = p + j;
            // The center should be 128, not 127
            // [0, 255] - 128 = [-128, 127] = [char.MIN_VALUE, char.MAX_VALUE]
            // But we use it here since that's what the comparison image used
            if (*c < 127) {
                *c = 127;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Check for the command line argument
    if (argc != 4) {
        printf("Expected 3 arguments\n");
        return -1;
    }
    // The first argument is the program name, skip it
    // The next is the input PNG file
    char *inputName = argv[1];
    // The next is the output file name
    char *outputName = argv[2];
    // The last is the thread count
    unsigned threadCount = strtoul(argv[3], NULL, 10);
    if (threadCount <= 0) {
        // This also occurs if the string is not a number
        printf("Thread count is 0 or not a number\n");
        return -1;
    }
    // Now load the input PNG file
    unsigned char *image;
    unsigned width, height;
    unsigned readError = lodepng_decode32_file(&image, &width, &height, inputName);
    if (readError) {
        printf("Error when loading the input image: %s\n", lodepng_error_text(readError));
    }
    // Apply rectification
    rectify(image, width, height, threadCount);
    // Save the results
    unsigned outputError = lodepng_encode32_file(outputName, image, width, height);
    if (outputError) {
        printf("Error when saving the output image: %s\n", lodepng_error_text(outputError));
    }
    // Delete the image
    free(image);
    return 0;
}
