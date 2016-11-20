#define WARP_MULTIPLE 32

int selectBestGPU() {
    // Get the number of devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    // There should be at least one
    if (deviceCount <= 0) {
        return 0;
    }
    // Get the one with the most processors
    int maxProc = 0;
    int maxDevice = 0;
    char maxName[256];
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        if (maxProc < properties.multiProcessorCount) {
            maxProc = properties.multiProcessorCount;
            maxDevice = device;
            memcpy(maxName, properties.name, 256 * sizeof(char));
        }
    }
    // Use that device
    cudaSetDevice(maxDevice);
    return 1;
}

int getLargestPreviousSquareAndMultipleOfM(int n, int m) {
    // Round down to the next multiple of m
    n -= n % m;
    // Look for a perfect square
    double ignored;
    while (n > 0 && modf(sqrt((double) n), &ignored) != 0) {
        // Otherwise reduce to the next multiple of m
        n -= m;
    }
    return n;
}

template<class T> int findBestGridAndBlockDims2D(unsigned outputWidth, unsigned outputHeight, T kernelFunction, dim3* dimBlock, dim3* dimGrid) {
    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFunction, 0, outputWidth * outputHeight);
    // Calculate the best 2D block size from the recommended block size
    if (blockSize < WARP_MULTIPLE) {
        // This can't be a multiple of the warp size, but it can be a square
        blockSize = getLargestPreviousSquareAndMultipleOfM(blockSize, 1);
    } else {
        // Find a square and multiple of the warp size
        blockSize = getLargestPreviousSquareAndMultipleOfM(blockSize, WARP_MULTIPLE);
    }
    if (blockSize <= 0) {
        return 0;
    }
    int blockLength = sqrt(blockSize);
    // Set the 2D block and grid dimensions
    dimBlock->x = blockLength;
    dimBlock->y = blockLength;
    dimBlock->z = 1;
    dimGrid->x = (outputWidth + dimBlock->x - 1) / dimBlock->x;
    dimGrid->y = (outputHeight + dimBlock->y - 1) / dimBlock->y;
    dimGrid->z = 1;
    return 1;
}
