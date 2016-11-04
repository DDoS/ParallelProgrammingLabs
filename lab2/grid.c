#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include "grid.h"
#include "constant.h"

#define ABOVE_TAG 0
#define RIGHT_TAG 1
#define BELOW_TAG 2
#define LEFT_TAG 3

/*
    Calculates the number of rows and columns in the partition from the desired number of blocks
*/
Partition createPartition(unsigned processCount) {
    // Calculate the hypothetical best division of blocks: the square root
    unsigned s = sqrt(processCount);
    // Find the next closest multiple of the block count
    while (processCount % s != 0) {
        s--;
    }
    // It will be the row count
    unsigned rows = s;
    // We get the column count from the row count
    unsigned cols = processCount / s;
    Partition partition = {rows, cols};
    return partition;
}

/*
    Creates a block, which is the data assigned to a single process
    The parameters are the number of blocks and the process rank
*/
Block createBlock(Partition* partition, unsigned index) {
    // Get the number of rows and columns in the block partition
    unsigned rows = partition->rows;
    unsigned cols = partition->cols;
    // Check that the process index is in bounds
    if (index >= rows * cols) {
        exit(-1);
    }
    // Now calculate the size of the block for the process
    unsigned blockRows = N / rows;
    unsigned blockCols = N / cols;
    // Then calculate the block coordinates, in block space
    unsigned bi = index % rows;
    unsigned bj = index / rows;
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
        .index = index, .i = ni, .j = nj, .rows = blockRows, .cols = blockCols, .nodes = nodes,
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
        |        i
    l - n - r    ^
        |        |
        b        + -- > j
*/
void updateNode(unsigned i, unsigned j, Node *n,  Node *a,  Node *r,  Node *b,  Node *l) {
    if (j == 0) {
        if (i == 0) {
            // Corner case
            n->u = G * r->u;
            return;
        }
        if (i == N - 1) {
            // Corner case
            n->u = G * b->u;
            return;
        }
        // Side case
        n->u = G * r->u;
        return;
    }
    if (j == N - 1) {
        if (i == 0) {
            // Corner case
            n->u = G * l->u;
            return;
        }
        if (i == N - 1) {
            // Corner case
            n->u = G * b->u;
            return;
        }
        // Side case
        n->u = G * l->u;
        return;
    }
    if (i == 0) {
        // Side case
        n->u = G * a->u;
        return;
    }
    if (i == N - 1) {
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
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = nodes + ((ii - 1) + jj * rows);
            Node *l = nodes + (ii + (jj - 1) * rows);
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
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = nodes + ((ii - 1) + jj * rows);
            Node *l = nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the right edge nodes, if we have nodes on the right
    Node *rightNodes = block->rightNodes;
    if (rightNodes != NULL) {
        unsigned jj = cols - 1;
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = rightNodes + ii;
            Node *b = nodes + ((ii - 1) + jj * rows);
            Node *l = nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the lower edge nodes, if we have nodes bellow
    Node *belowNodes = block->belowNodes;
    if (belowNodes != NULL) {
        unsigned ii = 0;
        for (unsigned jj = 1; jj < cols - 1; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = belowNodes + jj;
            Node *l = nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // Update the left edge nodes, if we have nodes on the left
    Node *leftNodes = block->leftNodes;
    if (leftNodes != NULL) {
        unsigned jj = 0;
        for (unsigned ii = 1; ii < rows - 1; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = nodes + ((ii - 1) + jj * rows);
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
        Node *b = nodes + ((ii - 1) + jj * rows);
        Node *l = nodes + (ii + (jj - 1) * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // Update the bottom right corner node, if we have nodes below and on the right
    if (belowNodes != NULL && rightNodes != NULL) {
        unsigned ii = 0;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + ((ii + 1) + jj * rows);
        Node *r = rightNodes + ii;
        Node *b = belowNodes + jj;
        Node *l = nodes + (ii + (jj - 1) * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // Update the bottom left corner node, if we have nodes below and on the left
    if (belowNodes != NULL && leftNodes != NULL) {
        unsigned ii = 0;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + ((ii + 1) + jj * rows);
        Node *r = nodes + (ii + (jj + 1) * rows);
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
        Node *r = nodes + (ii + (jj + 1) * rows);
        Node *b = nodes + ((ii - 1) + jj * rows);
        Node *l = leftNodes + ii;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
}

/*
    Performs one block update, only for the edge nodes
*/
void updateBlockGridEdge(Block *block) {
    unsigned ni = block->i;
    unsigned nj = block->j;
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    Node *aboveNodes = block->aboveNodes;
    Node *rightNodes = block->rightNodes;
    Node *belowNodes = block->belowNodes;
    Node *leftNodes = block->leftNodes;
    // Find the start and end of the rows and columns for middle nodes
    unsigned startRow = belowNodes == NULL ? 1 : 0;
    unsigned endRow = aboveNodes == NULL ? rows - 1 : rows;
    unsigned startCol = leftNodes == NULL ? 1 : 0;
    unsigned endCol = rightNodes == NULL ? cols - 1 : cols;
    // The lack of nodes aboves means this is the upper edge
    if (aboveNodes == NULL) {
        unsigned ii = rows - 1;
        for (unsigned jj = startCol; jj < endCol; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = NULL;
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = nodes + ((ii - 1) + jj * rows);
            Node *l = nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes on the right means this is the right edge
    if (rightNodes == NULL) {
        unsigned jj = cols - 1;
        for (unsigned ii = startRow; ii < endRow; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = NULL;
            Node *b = nodes + ((ii - 1) + jj * rows);
            Node *l = nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes bellow means this is the lower edge
    if (belowNodes == NULL) {
        unsigned ii = 0;
        for (unsigned jj = startCol; jj < endCol; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = NULL;
            Node *l = nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes on the left means this is the left edge
    if (leftNodes == NULL) {
        unsigned jj = 0;
        for (unsigned ii = startRow; ii < endRow; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = nodes + ((ii + 1) + jj * rows);
            Node *r = nodes + (ii + (jj + 1) * rows);
            Node *b = nodes + ((ii - 1) + jj * rows);
            Node *l = NULL;
            updateNode(ni + ii, nj + jj, n, a, r, b, l);
        }
    }
    // The lack of nodes on the upper and right edges means this is the upper right corner
    if (aboveNodes == NULL && rightNodes == NULL) {
        unsigned ii = rows - 1;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = NULL;
        Node *r = NULL;
        Node *b = nodes + ((ii - 1) + jj * rows);
        Node *l = nodes + (ii + (jj - 1) * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // The lack of nodes on the bottom and right edges means this is the lower right corner
    if (belowNodes == NULL && rightNodes == NULL) {
        unsigned ii = 0;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + ((ii + 1) + jj * rows);
        Node *r = NULL;
        Node *b = NULL;
        Node *l = nodes + (ii + (jj - 1) * rows);
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // The lack of nodes on the bottom and left edges means this is the lower left corner
    if (belowNodes == NULL && leftNodes == NULL) {
        unsigned ii = 0;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = nodes + ((ii + 1) + jj * rows);
        Node *r = nodes + (ii + (jj + 1) * rows);
        Node *b = NULL;
        Node *l = NULL;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
    // The lack of nodes on the upper and left edges means this is the upper left corner
    if (aboveNodes == NULL && leftNodes == NULL) {
        unsigned ii = rows - 1;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *a = NULL;
        Node *r = nodes + (ii + (jj + 1) * rows);
        Node *b = nodes + ((ii - 1) + jj * rows);
        Node *l = NULL;
        updateNode(ni + ii, nj + jj, n, a, r, b, l);
    }
}

/*
    Updates the age of a node value by moving them to older "u" variables
*/
void updateNodeValueAge(Node *nodes, unsigned count) {
    for (unsigned i = 0; i < count; i++) {
        Node *n = nodes + i;
        n->u2 = n->u1;
        n->u1 = n->u;
    }
}

/*
    Updates the age of the node values by moving them to older "u" variables
*/
void updateBlockValueAge(Block *block) {
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    // Update the values of the main nodes
    updateNodeValueAge(block->nodes, rows * cols);
    // Do the same for the boundaries
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

void communicateBoundaryNodes(Partition *partition, Block *block) {
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    // Calculate the block position in the partition
    unsigned partitionRows = partition->rows;
    unsigned index = block->index;
    unsigned bi = index % partitionRows;
    unsigned bj = index / partitionRows;
    // Use a custom data types for rows so we don't have to copy to temp storage
    static MPI_Datatype rowType;
    static int rowTypeCached = 0;
    if (!rowTypeCached) {
        // "cols" elements, of size "sizeof(Node)" bytes , every "sizeof(Node) * cols" bytes
        MPI_Type_vector(cols, sizeof(Node), sizeof(Node) * cols, MPI_CHAR, &rowType);
        MPI_Type_commit(&rowType);
        rowTypeCached = 1;
    }
    // Receive requests, so we can wait for reception
    MPI_Request receiveRequests[4];
    unsigned requestCount = 0;
    // Exchange on upper boundary, if it exists
    Node *aboveNodes = block->aboveNodes;
    if (aboveNodes != NULL) {
        // Calculate the index of the process above
        unsigned indexAbove = (bi + 1) + bj * rows;
        // Send above and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes + (rows - 1), 1, rowType,
                indexAbove, ABOVE_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
        // Receive data from above, and keep the output request to be waited on
        MPI_Irecv(aboveNodes, sizeof(Node) * cols, MPI_CHAR,
                indexAbove, BELOW_TAG, MPI_COMM_WORLD, &receiveRequests[requestCount++]);
    }
    // Exchange on right boundary, if it exists
    Node *rightNodes = block->rightNodes;
    if (rightNodes != NULL) {
        // Calculate the index of the process to the right
        unsigned indexRight = bi + (bj + 1) * partitionRows;
        // Send to the right and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes + rows * (cols - 1), sizeof(Node) * rows, MPI_CHAR,
                indexRight, RIGHT_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
        // Receive data from the right, and keep the output request to be waited on
        MPI_Irecv(rightNodes, sizeof(Node) * rows, MPI_CHAR,
                indexRight, LEFT_TAG, MPI_COMM_WORLD, &receiveRequests[requestCount++]);
    }
    // Exchange on lower boundary, if it exists
    Node *belowNodes = block->belowNodes;
    if (belowNodes != NULL) {
        // Calculate the index of the process below
        unsigned indexBelow = (bi - 1) + bj * rows;
        // Send below and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes, 1, rowType,
                indexBelow, BELOW_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
        // Receive data from below, and keep the output request to be waited on
        MPI_Irecv(belowNodes, sizeof(Node) * cols, MPI_CHAR,
                indexBelow, ABOVE_TAG, MPI_COMM_WORLD, &receiveRequests[requestCount++]);
    }
    // Exchange on left boundary, if it exists
    Node *leftNodes = block->leftNodes;
    if (leftNodes != NULL) {
        // Calculate the index of the process to the left
        unsigned indexLeft = bi + (bj - 1) * partitionRows;
        // Send to the left and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes, sizeof(Node) * rows, MPI_CHAR,
                indexLeft, LEFT_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
        // Receive data from the left, and keep the output request to be waited on
        MPI_Irecv(leftNodes, sizeof(Node) * rows, MPI_CHAR,
                indexLeft, RIGHT_TAG, MPI_COMM_WORLD, &receiveRequests[requestCount++]);
    }
    // Wait for all the data to have been received before continuing
    MPI_Waitall(requestCount, receiveRequests, MPI_STATUSES_IGNORE);
}

/*
    Performs one an update of a block for one iteration
*/
void updateBlock(Partition *partition, Block *block) {
    updateBlockGridMiddle(block);
    communicateBoundaryNodes(partition, block);
    updateBlockGridEdge(block);
    updateBlockValueAge(block);
}
