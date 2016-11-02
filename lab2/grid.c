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
    if (process >= blocks) {
        exit(-1);
    }
    // First calculate the number of rows and columns
    unsigned rows;
    unsigned cols;
    calculateBlockLayout(blocks, &rows, &cols);
    // Now calculate the size of the block for the process
    unsigned blockRows = N / rows;
    unsigned blockCols = N / cols;
    // Then calculate the block coordinates, in block space
    unsigned bi = process % rows;
    unsigned bj = process / rows;
    // Convert the coordinates to node space
    unsigned ni = bi * blockRows;
    unsigned nj = bj * blockCols;
    // If the process is the last row, assign it any extra rows
    if (bi + 1 == rows) {
        blockRows += N % rows;
    }
    // If the process is the last column, assign it any extra columns
    if (bj + 1 == cols) {
        blockCols += N % cols;
    }
    // Allocate nodes for the block
    Node *nodes = calloc(blockRows * blockCols, sizeof(Node));
    // Create the grid block
    Block block = {
        .i = ni, .j = nj, .rows = blockRows, .cols = blockCols, .nodes = nodes,
        .aboveNodes = NULL, .rightNodes = NULL, .belowNodes = NULL, .leftNodes = NULL
    };
    // If the block shares a boundary, then we add extra nodes
    // to represent the overlap with another process
    unsigned comNodeCount = 0;
    if (bi + 1 < rows) {
        // Need to communicate with above
        block.aboveNodes = calloc(blockCols, sizeof(Node));
    }
    if (bj + 1 < cols) {
        // Need to communicate with right
        block.rightNodes = calloc(blockRows, sizeof(Node));
    }
    if (bi > 0) {
        // Need to communicate with below
        block.belowNodes = calloc(blockCols, sizeof(Node));
    }
    if (bj > 0) {
        // Need to communicate with left
        block.leftNodes = calloc(blockRows, sizeof(Node));
    }
    return block;
}

/*
    The node at "n" and has coordinates (i, j).
    It is surrounded, as depicted below, by nodes "a", "r", "b" and "l".
    If the node is not in the middle, NULL is used for non-existing neighbours.

        a
        |        j
    l - n - r    ^
        |        |
        b        + -- > i
*/
void updateNode(unsigned i, unsigned j, Node *n,  Node *a,  Node *r,  Node *b,  Node *l) {
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

/*
    Performs one block update, only for the middle nodes.
*/
void updateBlockGridMiddle(Block *block) {
    unsigned ni = block->i;
    unsigned nj = block->j;
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    // Start with the nodes in the middle of the block
    for (unsigned jj = 1; jj < cols - 1; jj++) {
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the upper edge nodes, if we have nodes above
    Node *aboveNodes = block->aboveNodes;
    if (aboveNodes != NULL) {
        unsigned ii = rows - 1;
        for (unsigned jj = 1; jj < cols - 1; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = aboveNodes + jj;
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the right edge nodes, if we have nodes on the right
    Node *rightNodes = block->rightNodes;
    if (rightNodes != NULL) {
        unsigned jj = cols - 1;
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = rightNodes + ii;
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the lower edge nodes, if we have nodes bellow
    Node *belowNodes = block->belowNodes;
    if (belowNodes != NULL) {
        unsigned ii = 0;
        for (unsigned jj = 1; jj < cols - 1; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = belowNodes + jj;
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the left edge nodes, if we have nodes on the left
    Node *leftNodes = block->leftNodes;
    if (leftNodes != NULL) {
        unsigned jj = 0;
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = leftNodes + ii;
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the upper right corner node, if we have nodes above and on the right
    if (aboveNodes != NULL && rightNodes != NULL) {
        unsigned ii = rows - 1;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = aboveNodes + jj;
        Node *r = rightNodes + ii;
        Node *b = nodes + (ii + (jj - 1) * rows);
        Node *l = nodes + ((ii - 1) + jj * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // Update the bottom right corner node, if we have nodes below and on the right
    if (belowNodes != NULL && rightNodes != NULL) {
        unsigned ii = 0;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + (ii + (jj + 1) * rows);
        Node *r = rightNodes + ii;
        Node *b = belowNodes + jj;
        Node *l = nodes + ((ii - 1) + jj * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // Update the bottom left corner node, if we have nodes below and on the left
    if (belowNodes != NULL && leftNodes != NULL) {
        unsigned ii = 0;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + (ii + (jj + 1) * rows);
        Node *r = nodes + ((ii + 1) + jj * rows);
        Node *b = belowNodes + jj;
        Node *l = leftNodes + ii;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // Update the upper left corner node, if we have nodes above and on the left
    if (aboveNodes != NULL && leftNodes != NULL) {
        unsigned ii = rows - 1;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = aboveNodes + jj;
        Node *r = nodes + ((ii + 1) + jj * rows);
        Node *b = nodes + (ii + (jj - 1) * rows);
        Node *l = leftNodes + ii;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
}

void updateBlockGridEdge(Block* block) {
    unsigned ni = block->i;
    unsigned nj = block->j;
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    // The lack of nodes aboves means this is the upper edge
    Node *aboveNodes = block->aboveNodes;
    if (aboveNodes == NULL) {
        unsigned ii = rows - 1;
        for (unsigned jj = 1; jj < cols - 1; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = NULL;
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes on the right means this is the right edge
    Node *rightNodes = block->rightNodes;
    if (rightNodes == NULL) {
        unsigned jj = rows - 1;
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = NULL;
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes bellow means this is the lower edge
    Node *belowNodes = block->belowNodes;
    if (belowNodes == NULL) {
        unsigned ii = 0;
        for (unsigned jj = 1; jj < cols - 1; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = NULL;
            Node *l = nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes on the left means this is the left edge
    Node *leftNodes = block->leftNodes;
    if (leftNodes == NULL) {
        unsigned jj = 0;
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + (ii + (jj + 1) * rows);
            Node *r = nodes + ((ii + 1) + jj * rows);
            Node *b = nodes + (ii + (jj - 1) * rows);
            Node *l = NULL;
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes on the upper and right edges means this is the upper right corner
    if (aboveNodes != NULL && rightNodes != NULL) {
        unsigned ii = rows - 1;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = NULL;
        Node *r = NULL;
        Node *b = nodes + (ii + (jj - 1) * rows);
        Node *l = nodes + ((ii - 1) + jj * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // The lack of nodes on the bottom and right edges means this is the lower right corner
    if (belowNodes != NULL && rightNodes != NULL) {
        unsigned ii = 0;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + (ii + (jj + 1) * rows);
        Node *r = NULL;
        Node *b = NULL;
        Node *l = nodes + ((ii - 1) + jj * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // The lack of nodes on the bottom and left edges means this is the lower left corner
    if (belowNodes != NULL && leftNodes != NULL) {
        unsigned ii = 0;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + (ii + (jj + 1) * rows);
        Node *r = nodes + ((ii + 1) + jj * rows);
        Node *b = NULL;
        Node *l = NULL;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // The lack of nodes on the upper and left edges means this is the upper left corner
    if (aboveNodes != NULL && leftNodes != NULL) {
        unsigned ii = rows - 1;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = NULL;
        Node *r = nodes + ((ii + 1) + jj * rows);
        Node *b = nodes + (ii + (jj - 1) * rows);
        Node *l = NULL;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
}

void updateNodeValueAge(Node* nodes, unsigned count) {
    for (unsigned i = 0; i < count; i++) {
        Node *n = nodes + i;
        n->u2 = n->u1;
        n->u1 = n->u;
    }
}

void updateBlockValueAge(Block* block) {
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    updateNodeValueAge(block->nodes, rows * cols);
    Node *aboveNodes = block->aboveNodes;
    if (aboveNodes != NULL) {
        updateNodeValueAge(aboveNodes, cols);
    }
    Node *rightNodes = block->rightNodes;
    if (rightNodes != NULL) {
        updateNodeValueAge(rightNodes, rows);
    }
    Node *belowNodes = block->belowNodes;
    if (belowNodes != NULL) {
        updateNodeValueAge(belowNodes, cols);
    }
    Node *leftNodes = block->leftNodes;
    if (leftNodes != NULL) {
        updateNodeValueAge(leftNodes, rows);
    }
}

void updateBlock(Block* block) {
    updateBlockGridMiddle(block);
    // TODO: Communication goes here
    updateBlockGridEdge(block);
    updateBlockValueAge(block);
}
