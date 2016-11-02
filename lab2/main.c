#include <stdio.h>

#include "grid.h"

int main(int argc, char *argv[]) {
    unsigned sum;
    for (unsigned i = 0; i < 6; i++) {
        Block block = createBlock(6, i);
        printf("%d %d %d %d %d %d %d %d\n", block.i, block.j, block.rows, block.cols, block.boundaries[0], block.boundaries[1], block.boundaries[2], block.boundaries[3]);
        sum += block.rows * block.cols;
    }
    printf("%d\n", sum);
    return 0;
}
