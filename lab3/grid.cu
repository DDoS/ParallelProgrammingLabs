#include <stdio.h>

#include "common.h"

__global__ void iterate(cudaSurfaceObject_t surface) {
    // Calculate image coordinates in the output
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    // Check that we are within bounds
    if (x >= N || y >= N) {
        return;
    }
    // Read the node value
    float4 node;
    surf2Dread(&node, surface, x * 4, y);
    // Perform the middle updates
    node.x = node.y / 2;
    surf2Dwrite(node, surface, x * 4, y);
    // Barrier for completion of the middle updates

    // Perform the side updates

    // Barrier for completion of the side updates

    // Perform the corner updates
}

float* createInitialGrid() {
    // Allocate a zero'd grid
    float* grid = (float*) calloc(N * N * 4, sizeof(float));
    // Set the middle element u1 to 1
    float* middle = grid + (N * (N + 1) * 4) / 2;
    middle[1] = 1;
    return grid;
}

int main(int argc, char* argv[]) {
    // Check for the command line argument
    if (argc != 2) {
        printf("Expected 1 argument\n");
        return -1;
    }
    // The first argument is the program name, skip it
    // The second is the iteration count
    unsigned iterationCount = strtoul(argv[1], NULL, 10);
    if (iterationCount <= 0) {
        // This also occurs if the string is not a number
        printf("Iteration count is 0 or not a number\n");
        return -1;
    }
    // Select the GPU first
    if (!selectBestGPU()) {
        printf("No CUDA supporting GPU found\n");
        return -1;
    }
    // Get the recommended block and grid sizes
    dim3 dimBlock;
    dim3 dimGrid;
    if (!findBestGridAndBlockDims2D(N, N, iterate, &dimBlock, &dimGrid)) {
        printf("Could not calculate a suitable block size\n");
        return -1;
    }
    // Our simulation is made up of nodes with 4 float components (not 3 because of alignment constraints)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    // Create the initial grid on the CPU side
    float* grid = createInitialGrid();
    // Allocate a CUDA array on the GPU to hold the simulation datas
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, N, N);
    // Copy the initial grid configuration to the GPU
    unsigned gridByteSize = N * N * sizeof(float) * 4;
    cudaMemcpyToArray(cuArray, 0, 0, grid, gridByteSize, cudaMemcpyHostToDevice);
    // Create a resource description for the surface using the array
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    // Create the surface for the grid
    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);
    // Check for a CUDA error when creating the surface
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(error));
        return -1;
    }
    // Start the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Invoke kernel
    iterate<<<dimGrid, dimBlock>>>(surface);
    // Stop the time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Get the time delta in miliseconds
    float elapsedMili;
    cudaEventElapsedTime(&elapsedMili, start, stop);
    // Print the time taken
    printf("Took about %.4fms\n", elapsedMili);
    // Check for a CUDA error when finishing the job
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(error));
        return -1;
    }
    return 0;
}
