#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include "constant.h"
#include "grid.h"

void printGrid(Block *block) {
    unsigned ni = block->i;
    unsigned nj = block->j;
    unsigned rows = block->rows;
    unsigned cols = block->cols;
    Node *nodes = block->nodes;
    for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
            Node *node = nodes + i + j * rows;
            printf("(%d,%d): %0.6f ", ni + i, nj + j, node->u);
        }
        printf("\n");
    }
}

void doProcessWork(Partition *partition, unsigned index, unsigned iterationCount) {
    // Create the partition block for the process
    Block block = createBlock(partition, index);
    // Check if this process contains the middle node, which we use for input and output
    int middleInRows = block.i <= N_HALF && block.i + block.rows > N_HALF;
    int middleInCols = block.j <= N_HALF && block.j + block.cols > N_HALF;
    int containsMiddle = middleInRows && middleInCols;
    // Simulate a hit on the drum in the middle
    if (containsMiddle) {
        // The middle is in the main nodes
        block.nodes[(N_HALF - block.i) + (N_HALF - block.j) * block.rows].u1 += 1;
    } else if (middleInRows) {
        if (block.j == N_HALF + 1) {
            // The middle is in the left boundary nodes
            block.leftNodes[N_HALF - block.i].u1 += 1;
        } else if (block.j + block.cols == N_HALF) {
            // The middle is in the right boundary nodes
            block.rightNodes[N_HALF - block.i].u1 += 1;
        }
    } else if (middleInCols) {
        if (block.i == N_HALF + 1) {
            // The middle is in the lower boundary nodes
            block.belowNodes[N_HALF - block.j].u1 += 1;
        } else if (block.i + block.rows == N_HALF) {
            // The middle is in the upper boundary nodes
            block.aboveNodes[N_HALF - block.j].u1 += 1;
        }
    }
    // Wait for all processes to be ready for an iteration
    MPI_Barrier(MPI_COMM_WORLD);
    // Do an update
    updateBlock(partition, &block);
    // Print out the result
    printGrid(&block);
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    int processCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Create a block partition for the number of processes
    Partition partition = createPartition(processCount);
    // Get the iteration count from the arguments and pass it to all processes
    unsigned iterationCount;
    signed exitCode = 0;
    if (rank == 0) {
        // Check for the command line argument
        if (argc != 2) {
            printf("Expected 1 argument\n");
            exitCode = -1;
            goto end;
        }
        // The first argument is the program name, skip it
        // The second is the iteration count
        iterationCount = strtoul(argv[1], NULL, 10);
        if (iterationCount <= 0) {
            // This also occurs if the string is not a number
            printf("Iteration count is 0 or not a number\n");
            exitCode = -1;
            goto end;
        }
    }
    // Broadcast the iteration count to all processes
    MPI_Bcast(&iterationCount, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // Do the process work
    doProcessWork(&partition, rank, iterationCount);
    // Finalize the MPI environment
    end:
    MPI_Finalize();
    return exitCode;
}
