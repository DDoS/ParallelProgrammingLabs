#include "transform.h"

#define POOL_SIZE 2

void getOutputSize(unsigned* width, unsigned* height) {
    // Divide by the pool size
    *width /= POOL_SIZE;
    *height /= POOL_SIZE;
}

inline __host__ __device__ uint4 max(uint4 a, uint4 b) {
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

__global__ void transform(unsigned char* output, cudaTextureObject_t texture, unsigned width, unsigned height) {
    // Calculate image coordinates in the output
    unsigned xOut = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yOut = blockIdx.y * blockDim.y + threadIdx.y;
    // Check that we are within bounds
    if (xOut >= width || yOut >= height) {
        return;
    }
    // Calculate the texture coordinates in the input image
    unsigned xIn = xOut * POOL_SIZE;
    unsigned yIn = yOut * POOL_SIZE;
    // Pool in the pooling region of the input image
    uint4 pooledPixel = make_uint4(0, 0, 0, 0xFF);
    for (unsigned y = 0; y < POOL_SIZE; y++) {
        for (unsigned x = 0; x < POOL_SIZE; x++) {
            // Read the pixel data from the texture and apply the max function to the current value
            pooledPixel = max(pooledPixel, tex2D<uint4>(texture, xIn + x, yIn + y));
        }
    }
    // Calculate the output pixel address
    unsigned char* outputPixelAddress = output + (yOut * width + xOut) * 4;
    // Set the pooled pixel components
    outputPixelAddress[0] = pooledPixel.x;
    outputPixelAddress[1] = pooledPixel.y;
    outputPixelAddress[2] = pooledPixel.z;
    outputPixelAddress[3] = pooledPixel.w;
}
