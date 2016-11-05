#include <stdlib.h>
#include <math.h>
#include <stdio.h>

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
    // Check that the partition is feasable
    if (blockRows == 0 || blockCols == 0) {
        printf("Cannot divide a %dx%d grid into %d non-zero blocks\n", N, N, rows * cols);
        exit(-1);
    }
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
    Node *aboveNodes = block->aboveNodes;
    Node *rightNodes = block->rightNodes;
    Node *belowNodes = block->belowNodes;
    Node *leftNodes = block->leftNodes;
    // Specical cases when we have only one row or column
    if (rows == 1) {
        if (cols == 1) {
            // For an individual node, it's only a middle node if surrounded by other nodes on all sides
            if (aboveNodes != NULL && rightNodes != NULL && belowNodes != NULL && leftNodes != NULL) {
                updateNode(ni, nj, nodes, aboveNodes, rightNodes, belowNodes, leftNodes);
            }
            return;
        }
        // For a single row, it's only a middle node if it has nodes above an below
        if (aboveNodes != NULL && belowNodes != NULL) {
            unsigned ii = 0;
            for (unsigned jj = 1; jj < cols - 1; jj++) {
                Node *n = nodes + (ii + jj * rows);
                Node *a = aboveNodes + jj;
                Node *r = nodes + (ii + (jj + 1) * rows);
                Node *b = belowNodes + jj;
                Node *l = nodes + (ii + (jj - 1) * rows);
                updateNode(ni + ii, nj + jj, n, a, r, b, l);
            }
            // Update the right end of the row, if we have a node on the right
            if (rightNodes != NULL) {
                unsigned jj = cols - 1;
                Node *n = nodes + (ii + jj * rows);
                Node *a = aboveNodes + jj;
                Node *r = rightNodes + ii;
                Node *b = belowNodes + jj;
                Node *l = nodes + (ii + (jj - 1) * rows);
                updateNode(ni + ii, nj + jj, n, a, r, b, l);
            }
            // Update the left end of the row, if we have a node on the left
            if (leftNodes != NULL) {
                unsigned jj = 0;
                Node *n = nodes + (ii + jj * rows);
                Node *a = aboveNodes + jj;
                Node *r = nodes + (ii + (jj + 1) * rows);
                Node *b = belowNodes + jj;
                Node *l = leftNodes + ii;
                updateNode(ni + ii, nj + jj, n, a, r, b, l);
            }
        }
        return;
    }
    if (cols == 1) {
        // For a single column, it's only a middle node if it has nodes above an below
        if (rightNodes != NULL && leftNodes != NULL) {
            unsigned jj = 0;
            for (unsigned ii = 1; ii < rows - 1; ii++) {
                Node *n = nodes + (ii + jj * rows);
                Node *a = nodes + ((ii + 1) + jj * rows);
                Node *r = rightNodes + ii;
                Node *b = nodes + ((ii - 1) + jj * rows);
                Node *l = leftNodes + ii;
                updateNode(ni + ii, nj + jj, n, a, r, b, l);
            }
            // Update the upper end of the column, if we have a node above
            if (aboveNodes != NULL) {
                unsigned ii = rows - 1;
                Node *n = nodes + (ii + jj * rows);
                Node *a = aboveNodes + jj;
                Node *r = rightNodes + ii;
                Node *b = nodes + ((ii - 1) + jj * rows);
                Node *l = leftNodes + ii;
                updateNode(ni + ii, nj + jj, n, a, r, b, l);
            }
            // Update the lower end of the column, if we have a node below
            if (belowNodes != NULL) {
                unsigned ii = 0;
                Node *n = nodes + (ii + jj * rows);
                Node *a = nodes + ((ii + 1) + jj * rows);
                Node *r = rightNodes + ii;
                Node *b = belowNodes + jj;
                Node *l = leftNodes + ii;
                updateNode(ni + ii, nj + jj, n, a, r, b, l);
            }
        }
        return;
    }
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
    // Update the lower edge nodes, if we have nodes below
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
void updateBlockGridEdges(Block *block) {
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
            Node *b = rows == 1 ? belowNodes + jj : nodes + ((ii - 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, NULL, NULL, b, NULL);
        }
    }
    // The lack of nodes on the right means this is the right edge
    if (rightNodes == NULL) {
        unsigned jj = cols - 1;
        for (unsigned ii = startRow; ii < endRow; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *l = cols == 1 ? leftNodes + ii : nodes + (ii + (jj - 1) * rows);
            updateNode(ni + ii, nj + jj, n, NULL, NULL, NULL, l);
        }
    }
    // The lack of nodes below means this is the lower edge
    if (belowNodes == NULL) {
        unsigned ii = 0;
        for (unsigned jj = startCol; jj < endCol; jj++) {
            Node *n = nodes + (ii + jj * rows);
            Node *a = rows == 1 ? aboveNodes + jj : nodes + ((ii + 1) + jj * rows);
            updateNode(ni + ii, nj + jj, n, a, NULL, NULL, NULL);
        }
    }
    // The lack of nodes on the left means this is the left edge
    if (leftNodes == NULL) {
        unsigned jj = 0;
        for (unsigned ii = startRow; ii < endRow; ii++) {
            Node *n = nodes + (ii + jj * rows);
            Node *r = cols == 1 ? rightNodes + ii : nodes + (ii + (jj + 1) * rows);
            updateNode(ni + ii, nj + jj, n, NULL, r, NULL, NULL);
        }
    }
}

/*
    Performs one block update, only for the corner nodes
*/
void updateBlockGridCorners(Block *block) {
    unsigned ni = block->i;
    unsigned nj = block->j;
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    Node *aboveNodes = block->aboveNodes;
    Node *rightNodes = block->rightNodes;
    Node *belowNodes = block->belowNodes;
    Node *leftNodes = block->leftNodes;
    // The lack of nodes on the upper and right edges means this is the upper right corner
    if (aboveNodes == NULL && rightNodes == NULL) {
        unsigned ii = rows - 1;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *b = rows == 1 ? belowNodes + jj : nodes + ((ii - 1) + jj * rows);
        updateNode(ni + ii, nj + jj, n, NULL, NULL, b, NULL);
    }
    // The lack of nodes on the bottom and right edges means this is the lower right corner
    if (belowNodes == NULL && rightNodes == NULL) {
        unsigned ii = 0;
        unsigned jj = cols - 1;
        Node *n = nodes + (ii + jj * rows);
        Node *l = cols == 1 ? leftNodes + ii : nodes + (ii + (jj - 1) * rows);
        updateNode(ni + ii, nj + jj, n, NULL, NULL, NULL, l);
    }
    // The lack of nodes on the bottom and left edges means this is the lower left corner
    if (belowNodes == NULL && leftNodes == NULL) {
        unsigned ii = 0;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *r = cols == 1 ? rightNodes + ii : nodes + (ii + (jj + 1) * rows);
        updateNode(ni + ii, nj + jj, n, NULL, r, NULL, NULL);
    }
    // The lack of nodes on the upper and left edges means this is the upper left corner
    if (aboveNodes == NULL && leftNodes == NULL) {
        unsigned ii = rows - 1;
        unsigned jj = 0;
        Node *n = nodes + (ii + jj * rows);
        Node *b = rows == 1 ? belowNodes + jj : nodes + ((ii - 1) + jj * rows);
        updateNode(ni + ii, nj + jj, n, NULL, NULL, b, NULL);
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

void sendBoundaryNodes(Partition *partition, Block *block, unsigned directions) {
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    // Calculate the block position in the partition
    unsigned partitionRows = partition->rows;
    unsigned index = block->index;
    unsigned bi = index % partitionRows;
    unsigned bj = index / partitionRows;
    // Use a custom vector data type for rows so we don't have to copy to temp storage
    static MPI_Datatype rowType;
    static int rowTypeCached = 0;
    if (!rowTypeCached) {
        // "cols" elements, of size "sizeof(Node)" bytes, every "sizeof(Node) * cols" bytes
        MPI_Type_vector(cols, sizeof(Node), sizeof(Node) * cols, MPI_CHAR, &rowType);
        MPI_Type_commit(&rowType);
        rowTypeCached = 1;
    }
    // Send on upper boundary, if it exists
    Node *aboveNodes = block->aboveNodes;
    if ((directions & 0b1) && aboveNodes != NULL) {
        // Calculate the index of the process above
        unsigned indexAbove = (bi + 1) + bj * partitionRows;
        // Send above and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes + (rows - 1), 1, rowType,
                indexAbove, ABOVE_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
    }
    // Send on right boundary, if it exists
    Node *rightNodes = block->rightNodes;
    if ((directions & 0b10) && rightNodes != NULL) {
        // Calculate the index of the process to the right
        unsigned indexRight = bi + (bj + 1) * partitionRows;
        // Send to the right and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes + rows * (cols - 1), sizeof(Node) * rows, MPI_CHAR,
                indexRight, RIGHT_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
    }
    // Send on lower boundary, if it exists
    Node *belowNodes = block->belowNodes;
    if ((directions & 0b100) && belowNodes != NULL) {
        // Calculate the index of the process below
        unsigned indexBelow = (bi - 1) + bj * partitionRows;
        // Send below and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes, 1, rowType,
                indexBelow, BELOW_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
    }
    // Send on left boundary, if it exists
    Node *leftNodes = block->leftNodes;
    if ((directions & 0b1000) && leftNodes != NULL) {
        // Calculate the index of the process to the left
        unsigned indexLeft = bi + (bj - 1) * partitionRows;
        // Send to the left and ignore the output request
        MPI_Request dataRequest;
        MPI_Isend(nodes, sizeof(Node) * rows, MPI_CHAR,
                indexLeft, LEFT_TAG, MPI_COMM_WORLD, &dataRequest);
        MPI_Request_free(&dataRequest);
    }
}

void receiveBoundaryNodes(Block *block, unsigned directions) {
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    // Receive requests, so we can wait for reception
    MPI_Request receiveRequests[4];
    unsigned requestCount = 0;
    // Receive on upper boundary, if it exists
    Node *aboveNodes = block->aboveNodes;
    if ((directions & 0b1) && aboveNodes != NULL) {
        // Receive data from above, and keep the output request to be waited on
        MPI_Irecv(aboveNodes, sizeof(Node) * cols, MPI_CHAR,
                MPI_ANY_SOURCE, BELOW_TAG, MPI_COMM_WORLD, receiveRequests + requestCount);
        requestCount++;
    }
    // Receive on right boundary, if it exists
    Node *rightNodes = block->rightNodes;
    if ((directions & 0b10) && rightNodes != NULL) {
        // Receive data from the right, and keep the output request to be waited on
        MPI_Irecv(rightNodes, sizeof(Node) * rows, MPI_CHAR,
                MPI_ANY_SOURCE, LEFT_TAG, MPI_COMM_WORLD, receiveRequests + requestCount);
        requestCount++;
    }
    // Receive on lower boundary, if it exists
    Node *belowNodes = block->belowNodes;
    if ((directions & 0b100) && belowNodes != NULL) {
        // Receive data from below, and keep the output request to be waited on
        MPI_Irecv(belowNodes, sizeof(Node) * cols, MPI_CHAR,
                MPI_ANY_SOURCE, ABOVE_TAG, MPI_COMM_WORLD, receiveRequests + requestCount);
        requestCount++;
    }
    // Receive on left boundary, if it exists
    Node *leftNodes = block->leftNodes;
    if ((directions & 0b1000) && leftNodes != NULL) {
        // Receive data from the left, and keep the output request to be waited on
        MPI_Irecv(leftNodes, sizeof(Node) * rows, MPI_CHAR,
                MPI_ANY_SOURCE, RIGHT_TAG, MPI_COMM_WORLD, receiveRequests + requestCount);
        requestCount++;
    }
    // Wait for all the data to have been received before continuing
    MPI_Waitall(requestCount, receiveRequests, MPI_STATUSES_IGNORE);
}

int isIsolatedEdge(Block *block) {
    return (block->rows == 1 && (block->i == 0 || block->i == N - 1))
        || (block->cols == 1 && (block->j == 0 || block->j == N - 1));
}

int isNextToIsolateEdge(Block *block) {
    return block->i == 1 || block->i + block->rows == N - 1
        || block->j == 1 || block->j + block->cols == N - 1;
}

void sendNodesToEdges(Partition *partition, Block *block) {
    unsigned directionsToEdge = 0;
    if (block->i == 1) {
        directionsToEdge |= 0b100;
    }
    if (block->j == 1) {
        directionsToEdge |= 0b1000;
    }
    if (block->i + block->rows == N - 1) {
        directionsToEdge |= 0b1;
    }
    if (block->j + block->cols == N - 1) {
        directionsToEdge |= 0b10;
    }
    sendBoundaryNodes(partition, block, directionsToEdge);
}

void edgesReceiveNodes(Block *block) {
    unsigned directionsFromEdge = 0;
    if (block->rows == 1 && (block->i == 0 || block->i == N - 1)) {
        directionsFromEdge |= 0b101;
    }
    if (block->cols == 1 && (block->j == 0 || block->j == N - 1)) {
        directionsFromEdge |= 0b1010;
    }
    receiveBoundaryNodes(block, directionsFromEdge);
}


int isIsolatedCorner(Block *block) {
    return (block->rows == 1 || block->cols == 1) && (
        (block->i == 0 && block->j == 0)
        || (block->i == 0 && block->j + block->cols == N)
        || (block->i + block->rows == N && block->j == 0)
        || (block->i + block->rows == N && block->j + block->cols == N)
    );
}

int isNextToIsolatedCorner(Block *block) {
    return (block->j == 0 && block->i == 1)
        || (block->i == 0 && block->j == 1)
        || (block->j == 0 && block->i + block->rows == N - 1)
        || (block->i == 0 && block->j + block->cols == N - 1)
        || (block->j + block->cols == N && block->i == 1)
        || (block->i + block->rows == N && block->j == 1)
        || (block->j + block->cols == N && block->i + block->rows == N - 1)
        || (block->i + block->rows == N && block->j + block->cols == N - 1);
}

void sendNodesToCorners(Partition *partition, Block *block) {
    unsigned directionToCorner;
    if (block->i == 1) {
        directionToCorner = 0b100;
    } else if (block->j == 1) {
        directionToCorner = 0b1000;
    } else if (block->i + block->rows == N - 1) {
        directionToCorner = 0b1;
    } else if (block->j + block->cols == N - 1) {
        directionToCorner = 0b10;
    }
    sendBoundaryNodes(partition, block, directionToCorner);
}

void cornersReceiveNodes(Block *block) {
    unsigned directionsFromCorner = 0;
    if (block->rows == 1) {
        directionsFromCorner |= 0b101;
    }
    if (block->cols == 1) {
        directionsFromCorner |= 0b1010;
    }
    receiveBoundaryNodes(block, directionsFromCorner);
}

/*
    Performs one an update of a block for one iteration
*/
void updateBlock(Partition *partition, Block *block) {
    sendBoundaryNodes(partition, block, 0b1111);
    receiveBoundaryNodes(block, 0b1111);

    updateBlockGridMiddle(block);

    MPI_Barrier(MPI_COMM_WORLD);
    if (isNextToIsolateEdge(block)) {
        sendNodesToEdges(partition, block);
    }
    if (isIsolatedEdge(block)) {
        edgesReceiveNodes(block);
    }

    updateBlockGridEdges(block);

    MPI_Barrier(MPI_COMM_WORLD);
    if (isNextToIsolatedCorner(block)) {
        sendNodesToCorners(partition, block);
    }
    if (isIsolatedCorner(block)) {
        cornersReceiveNodes(block);
    }

    updateBlockGridCorners(block);

    updateBlockValueAge(block);
}
