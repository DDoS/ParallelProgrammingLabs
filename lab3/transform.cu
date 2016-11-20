#include <stdio.h>

#include "lodepng.h"
#include "transform.h"

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
    printf("Using GPU: %s\n", maxName);
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

int main(int argc, char* argv[]) {
    // Check for the command line argument
    if (argc != 3) {
        printf("Expected 2 arguments\n");
        return -1;
    }
    // The first argument is the program name, skip it
    // The next is the input PNG file
    char* inputName = argv[1];
    // The next is the output file name
    char* outputName = argv[2];
    // Now load the input PNG file
    unsigned char* image;
    unsigned width, height;
    unsigned readError = lodepng_decode32_file(&image, &width, &height, inputName);
    if (readError) {
        printf("Error when loading the input image: %s\n", lodepng_error_text(readError));
        return -1;
    }
    // Select the GPU first
    if (!selectBestGPU()) {
        printf("No CUDA supporting GPU found\n");
        return -1;
    }
    // Get the size of the output from the input
    unsigned outputWidth = width;
    unsigned outputHeight = height;
    getOutputSize(&outputWidth, &outputHeight);
    // Get the recommended block size
    unsigned outputPixelCount = outputWidth * outputHeight;
    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, transform, 0, outputPixelCount);
    // Calculate the best 2D block size from the recommended block size
    blockSize = getLargestPreviousSquareAndMultipleOfM(blockSize, WARP_MULTIPLE);
    if (blockSize <= 0) {
        printf("Could not calculate a suitable block size\n");
        return -1;
    }
    int blockLength = sqrt(blockSize);
    // Create the 2D block and grid dimensions
    dim3 dimBlock(blockLength, blockLength);
    dim3 dimGrid((outputWidth + dimBlock.x - 1) / dimBlock.x, (outputHeight + dimBlock.y - 1) / dimBlock.y);
    // Our image is made up of four 8 bit unsigned components
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuArray;
    // Allocate a CUDA array on the GPU to hold the input image
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    // Copy the image data to the GPU
    unsigned imageByteSize = width * height * sizeof(unsigned char) * 4;
    cudaMemcpyToArray(cuArray, 0, 0, image, imageByteSize, cudaMemcpyHostToDevice);
    // Delete the input image since we no longer need it
    free(image);
    // Create a resource description for the texture using the array
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    // Specify the texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    // Create texture object
    cudaTextureObject_t texture = 0;
    cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);
    // Allocate the output of transformation on the GPU
    unsigned imageOutputByteSize = outputPixelCount * sizeof(unsigned char) * 4;
    unsigned char* output;
    cudaMalloc(&output, imageOutputByteSize);
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
    transform<<<dimGrid, dimBlock>>>(output, texture, outputWidth, outputHeight);
    // Stop the time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Get the time delta in miliseconds
    float elapsedMili;
    cudaEventElapsedTime(&elapsedMili, start, stop);
    // Print the time taken
    printf("Took about %.4fms\n", elapsedMili);
    // Allocate some some CPU side memory for the output image
    unsigned char* imageOut = (unsigned char*) malloc(imageOutputByteSize);
    // Copy the output from the GPU back into the image
    cudaMemcpy(imageOut, output, imageOutputByteSize, cudaMemcpyDeviceToHost);
    // Destroy texture object
    cudaDestroyTextureObject(texture);
    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
    // Check for a CUDA error when finishing the job
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(error));
        return -1;
    }
    // Save the results
    unsigned outputError = lodepng_encode32_file(outputName, imageOut, outputWidth, outputHeight);
    if (outputError) {
        printf("Error when saving the output image: %s\n", lodepng_error_text(outputError));
        return -1;
    }
    return 0;
}
