#include <stdio.h>

#include "grid.h"

int main(int argc, char *argv[]) {
    unsigned processCount = 1;
    unsigned sum;
    for (unsigned i = 0; i < processCount; i++) {
        Block block = createBlock(processCount, i);
        printf("%d %d %d %d %d %d %d %d\n", block.i, block.j, block.rows, block.cols,
                block.aboveNodes != NULL, block.rightNodes != NULL, block.belowNodes != NULL, block.leftNodes != NULL);
        sum += block.rows * block.cols;
    }
    printf("%d\n", sum);
    return 0;
}
