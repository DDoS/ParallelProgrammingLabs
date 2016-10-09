#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"
#include "transform.h"

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
    // Apply transformation
    transform(&image, &width, &height, threadCount);
    // Save the results
    unsigned outputError = lodepng_encode32_file(outputName, image, width, height);
    if (outputError) {
        printf("Error when saving the output image: %s\n", lodepng_error_text(outputError));
    }
    // Delete the image
    free(image);
    return 0;
}
