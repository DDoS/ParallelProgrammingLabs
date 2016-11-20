#include <stdio.h>

#include "common.h"

__global__ void iterate() {

}

int main(int argc, char* argv[]) {
    // Check for the command line argument
    if (argc != 1) {
        printf("Expected no arguments\n");
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
    // Check for a CUDA error when creating the texture
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
    iterate<<<dimGrid, dimBlock>>>();
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
