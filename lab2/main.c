#include <stdio.h>

#include "grid.h"

void printGrid(Block* block) {
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
            Node *node = nodes + i + j * rows;
            printf("(%d,%d): %0.6f ", i, j, node->u);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    Block block = createBlock(1, 0);
    block.nodes[2 + 2 * block.rows].u1 += 1;
    updateBlock(&block);
    printGrid(&block);
    printf("\n");
    updateBlock(&block);
    printGrid(&block);
    return 0;
}
