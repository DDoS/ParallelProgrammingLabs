#include <stdio.h>

#include "common.h"
#include "constant.h"

void printGrid(float* grid) {
    for (unsigned y = 0; y < N; y++) {
        for (unsigned x = 0; x < N; x++) {
            float* node = grid + (y * N + x) * 4;
            printf("(%d,%d): %0.6f ", x, y, node[0]);
        }
        printf("\n");
    }
}

inline __device__ float4 readAndUpdate(cudaSurfaceObject_t surface, unsigned x, unsigned y) {
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

__global__ void iterate(cudaSurfaceObject_t surface) {
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
            surf2Dread(&n, surface, x * sizeof(float) * 4, y);
            float4 ra = readAndUpdate(surface, x + 1, y + 1);
            n.x = G * G * ra.x;
        } else if (y == N - 1) {
            // Corner case
            surf2Dread(&n, surface, x * sizeof(float) * 4, y);
            float4 rb = readAndUpdate(surface, x + 1, y - 1);
            n.x = G * G * rb.x;
        } else {
            // Side case
            surf2Dread(&n, surface, x * sizeof(float) * 4, y);
            float4 r = readAndUpdate(surface, x + 1, y);
            n.x = G * r.x;
        }
    } else if (x == N - 1) {
        if (y == 0) {
            // Corner case
            surf2Dread(&n, surface, x * sizeof(float) * 4, y);
            float4 la = readAndUpdate(surface, x - 1, y + 1);
            n.x = G * G * la.x;
        } else if (y == N - 1) {
            // Corner case
            surf2Dread(&n, surface, x * sizeof(float) * 4, y);
            float4 lb = readAndUpdate(surface, x - 1, y - 1);
            n.x = G * G * lb.x;
        } else {
            // Side case
            surf2Dread(&n, surface, x * sizeof(float) * 4, y);
            float4 l = readAndUpdate(surface, x - 1, y);
            n.x = G * l.x;
        }
    } else if (y == 0) {
        // Side case
        surf2Dread(&n, surface, x * sizeof(float) * 4, y);
        float4 a = readAndUpdate(surface, x, y + 1);
        n.x = G * a.x;
    } else if (y == N - 1) {
        // Side case
        surf2Dread(&n, surface, x * sizeof(float) * 4, y);
        float4 b = readAndUpdate(surface, x, y - 1);
        n.x = G * b.x;
    } else {
        // Middle case
        n = readAndUpdate(surface, x, y);
    }
    // Update the age of the values
    n.z = n.y;
    n.y = n.x;
    // Write back the value
    surf2Dwrite(n, surface, x * sizeof(float) * 4, y);
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
    // Allocate a CUDA array on the GPU to hold the simulation data
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, N, N, cudaArraySurfaceLoadStore);
    // Copy the initial grid configuration to the GPU array
    unsigned gridByteSize = N * N * sizeof(float) * 4;
    cudaMemcpyToArray(cuArray, 0, 0, grid, gridByteSize, cudaMemcpyHostToDevice);
    // Create a resource description for a surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    // Create two surfaces for the grid
    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);
    // Check for a CUDA error when creating the surfaces
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
    // Invoke kernel once per iteration
    for (int i = 0; i < iterationCount; i++) {
        iterate<<<dimGrid, dimBlock>>>(surface);
    }
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
    // Copy the grid from the GPU back to the CPU
    cudaMemcpyFromArray(grid, cuArray, 0, 0, gridByteSize, cudaMemcpyDeviceToHost);
    // Destroy surface object
    cudaDestroySurfaceObject(surface);
    // Free the GPU memory
    cudaFreeArray(cuArray);

    printGrid(grid);

    return 0;
}
