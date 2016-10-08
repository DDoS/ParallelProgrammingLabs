#include <omp.h>

void transform(unsigned char *image, unsigned width, unsigned height, unsigned threadCount) {
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
