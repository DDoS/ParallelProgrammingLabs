#include "transform.h"

void getOutputSize(unsigned* width, unsigned* height) {
    // Same size
}

__global__ void transform(unsigned char* output, cudaTextureObject_t texture, unsigned width, unsigned height) {
    // Calculate texture coordinates
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    // Check that we are within bounds
    if (x >= width || y >= height) {
        return;
    }
    // Read the pixel data from the texture
    uint4 pixel = tex2D<uint4>(texture, x, y);
    // Calculate the output pixel address
    unsigned char* outputPixelAddress = output + (y * width + x) * 4;
    // Set the rectified pixel components (ignoring the alpha component)
    outputPixelAddress[0] = max(pixel.x, 127);
    outputPixelAddress[1] = max(pixel.y, 127);
    outputPixelAddress[2] = max(pixel.z, 127);
    outputPixelAddress[3] = pixel.w;
}
