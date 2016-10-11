#ifdef __MACH__
#define MACH_TIMING
#endif // __MACH__

#ifdef __unix__
#define POSIX_TIMING
#define _POSIX_C_SOURCE 199309L
#endif // __unix__

#include <stdio.h>
#include <stdlib.h>

#ifdef MACH_TIMING
#include <mach/mach_time.h>
#elif defined(POSIX_TIMING)
#include <time.h>
#endif // MACH_TIMING

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
    // Get the start time
#ifdef MACH_TIMING
    uint64_t start = mach_absolute_time();
#elif defined(POSIX_TIMING)
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif // MACH_TIMING
    // Apply transformation
    transform(&image, &width, &height, threadCount);
    // Get the end time and delta in seconds
#ifdef MACH_TIMING
    uint64_t end = mach_absolute_time();
    uint64_t elapsed = end - start;
    mach_timebase_info_data_t timebaseInfo;
    mach_timebase_info(&timebaseInfo);
    double elapsedNano = elapsed * timebaseInfo.numer / timebaseInfo.denom;
#elif defined(POSIX_TIMING)
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsedNano = 1e9 * end.tv_sec + end.tv_nsec - 1e9 * start.tv_sec - start.tv_nsec;
#endif // MACH_TIMING
    // Print the time taken
    printf("took about %.5f seconds\n", 1e-9 * elapsedNano);
    // Save the results
    unsigned outputError = lodepng_encode32_file(outputName, image, width, height);
    if (outputError) {
        printf("Error when saving the output image: %s\n", lodepng_error_text(outputError));
    }
    // Delete the image
    free(image);
    return 0;
}
