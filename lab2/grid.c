#include <stdlib.h>
#include <math.h>

#include "grid.h"
#include "constant.h"

void calculateBlockLayout(unsigned blocks, unsigned *rows, unsigned *cols) {
    // Calculate the hypothetical best division of blocks: the square root
    unsigned s = sqrt(blocks);
    // Find the next closest multiple of the block count
    while (blocks % s != 0) {
        s--;
    }
    // It will be the row count
    *rows = s;
    // We get the column count from the row count
    *cols = blocks / s;
}

Block createBlock(unsigned blocks, unsigned process) {
    // First calculate the number of rows and columns
    unsigned rows;
    unsigned cols;
    calculateBlockLayout(blocks, &rows, &cols);
    // Now calculate the size of the block for the process
    unsigned blockRows = N / rows;
    unsigned blockCols = N / cols;
    // Then calculate the block coordinates, in block space
    unsigned i = process % rows;
    unsigned j = process / rows;
    // If the process is the last row, assign it any extra rows
    if (i + 1 == rows) {
        blockRows += N % rows;
    }
    // If the process is the last column, assign it any extra columns
    if (j + 1 == cols) {
        blockCols += N % cols;
    }
    // Allocate nodes for the block
    Node *nodes = malloc(blockRows * blockCols);
    // Create the grid block
    Block block = {
        .i = i, .j = j, .rows = blockRows, .cols = blockCols, .nodes = nodes, .comNodes = NULL
    };
    // If the block shares a boundary, then we add extra nodes
    // to represent the overlap with another process
    unsigned comNodeCount = 0;
    if (i + 1 < rows) {
        // Need to communicate with above
        comNodeCount += blockRows;
        block.boundaries[0] = 1;
    } else {
        block.boundaries[0] = 0;
    }
    if (j + 1 < cols) {
        // Need to communicate with right
        comNodeCount += blockCols;
        block.boundaries[1] = 1;
    } else {
        block.boundaries[1] = 0;
    }
    if (i > 0) {
        // Need to communicate with below
        comNodeCount += blockRows;
        block.boundaries[2] = 1;
    } else {
        block.boundaries[2] = 0;
    }
    if (j > 0) {
        // Need to communicate with left
        comNodeCount += blockCols;
        block.boundaries[3] = 1;
    } else {
        block.boundaries[3] = 0;
    }
    if (comNodeCount > 0) {
        block.comNodes = malloc(comNodeCount);
    }
    return block;
}

/*
    The node at "n" and has coordinates (i, j).
    It is surrounded, as depicted bellow, by nodes "a", "r", "b" and "l".
    If the node is not in the middle, 0 is used for non-existing values.

        a
        |        j
    l - n - r    ^
        |        |
        b        + -- > i
*/
void update(unsigned i, unsigned j, Node *n,  Node *a,  Node *r,  Node *b,  Node *l) {
    if (i == 0) {
        if (j == 0) {
            // Corner case
            n->u = G * r->u;
            return;
        }
        if (j == N - 1) {
            // Corner case
            n->u = G * b->u;
            return;
        }
        // Side case
        n->u = G * r->u;
        return;
    }
    if (i == N - 1) {
        if (j == 0) {
            // Corner case
            n->u = G * l->u;
            return;
        }
        if (j == N - 1) {
            // Corner case
            n->u = G * b->u;
            return;
        }
        // Side case
        n->u = G * l->u;
        return;
    }
    if (j == 0) {
        // Side case
        n->u = G * a->u;
        return;
    }
    if (j == N - 1) {
        // Side case
        n->u = G * b->u;
        return;
    }
    // Middle case
    n->u = (RHO * (l->u1 + r->u1 + b->u1 + a->u1 - 4 * n->u1) + 2 * n->u1 - (1 - ETA) * n->u2) / (1 + ETA);
}
