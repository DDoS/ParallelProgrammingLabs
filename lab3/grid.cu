#include <stdio.h>

#include "common.h"
#include "constant.h"

inline __device__ float4 readUpdated(cudaSurfaceObject_t surface, unsigned x, unsigned y) {
    // Read the node value and its neighbours
    float4 n, l, r, b, a;
    surf2Dread(&n, surface, x * sizeof(float) * 4, y);
    surf2Dread(&l, surface, (x - 1) * sizeof(float) * 4, y);
    surf2Dread(&r, surface, (x + 1) * sizeof(float) * 4, y);
    surf2Dread(&b, surface, x * sizeof(float) * 4, (y - 1));
    surf2Dread(&a, surface, x * sizeof(float) * 4, (y + 1));
    // Calculate the updated value
    n.x = (RHO * (l.y + r.y + b.y + a.y - 4 * n.y) + 2 * n.y - (1 - ETA) * n.z) / (1 + ETA);
    return n;
}

__global__ void iterate(cudaSurfaceObject_t inSurface, cudaSurfaceObject_t outSurface) {
    // Calculate image coordinates in the output
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    // Check that we are within bounds
    if (x >= N || y >= N) {
        return;
    }
    // For edges and corners, just defer to the middle node value being used
    float4 n;
    if (x == 0) {
        if (y == 0) {
            // Corner case
            surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
            float4 ra = readUpdated(inSurface, x + 1, y + 1);
            n.x = G * G * ra.x;
        } else if (y == N - 1) {
            // Corner case
            surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
            float4 rb = readUpdated(inSurface, x + 1, y - 1);
            n.x = G * G * rb.x;
        } else {
            // Side case
            surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
            float4 r = readUpdated(inSurface, x + 1, y);
            n.x = G * r.x;
        }
    } else if (x == N - 1) {
        if (y == 0) {
            // Corner case
            surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
            float4 la = readUpdated(inSurface, x - 1, y + 1);
            n.x = G * G * la.x;
        } else if (y == N - 1) {
            // Corner case
            surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
            float4 lb = readUpdated(inSurface, x - 1, y - 1);
            n.x = G * G * lb.x;
        } else {
            // Side case
            surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
            float4 l = readUpdated(inSurface, x - 1, y);
            n.x = G * l.x;
        }
    } else if (y == 0) {
        // Side case
        surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
        float4 a = readUpdated(inSurface, x, y + 1);
        n.x = G * a.x;
    } else if (y == N - 1) {
        // Side case
        surf2Dread(&n, inSurface, x * sizeof(float) * 4, y);
        float4 b = readUpdated(inSurface, x, y - 1);
        n.x = G * b.x;
    } else {
        // Middle case
        n = readUpdated(inSurface, x, y);
    }
    // Update the age of the values
    n.z = n.y;
    n.y = n.x;
    // Write back the value to the output
    surf2Dwrite(n, outSurface, x * sizeof(float) * 4, y);
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
    // Allocate two CUDA arrays on the GPU to hold the simulation input and output
    cudaArray* inArray;
    cudaMallocArray(&inArray, &channelDesc, N, N, cudaArraySurfaceLoadStore);
    cudaArray* outArray;
    cudaMallocArray(&outArray, &channelDesc, N, N, cudaArraySurfaceLoadStore);
    // Copy the initial grid configuration to the GPU input array
    unsigned gridByteSize = N * N * sizeof(float) * 4;
    cudaMemcpyToArray(inArray, 0, 0, grid, gridByteSize, cudaMemcpyHostToDevice);
    // Free the grid as we no longer need it
    free(grid);
    // Create a resource description for the surfaces
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    // Create two surfaces for the grid
    resDesc.res.array.array = inArray;
    cudaSurfaceObject_t inSurface = 0;
    cudaCreateSurfaceObject(&inSurface, &resDesc);
    resDesc.res.array.array = outArray;
    cudaSurfaceObject_t outSurface = 0;
    cudaCreateSurfaceObject(&outSurface, &resDesc);
    // Check for a CUDA error when creating the surfaces
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(error));
        return -1;
    }
    // Invoke kernel once per iteration
    for (int i = 0; i < iterationCount; i++) {
        iterate<<<dimGrid, dimBlock>>>(inSurface, outSurface);
        // Get the middle value from the GPU and print it
        float value;
        cudaMemcpyFromArray(&value, outArray, (N / 2) * sizeof(float) * 4, N / 2, sizeof(float), cudaMemcpyDeviceToHost);
        printf("%0.6f", value);
        if (i < iterationCount - 1) {
            printf(",");
        }
        printf("\n");
        // Swap the input and output (round-robin)
        cudaArray* tempArray = outArray;
        outArray = inArray;
        inArray = tempArray;
        cudaSurfaceObject_t tempSurface = outSurface;
        outSurface = inSurface;
        inSurface = tempSurface;
    }
    // Check for a CUDA error when finishing the job
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(error));
        return -1;
    }
    // Destroy surface object
    cudaDestroySurfaceObject(inSurface);
    cudaDestroySurfaceObject(outSurface);
    // Free the GPU memory
    cudaFreeArray(inArray);
    cudaFreeArray(outArray);
    return 0;
}
