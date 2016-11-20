#include "wm.h"

void getOutputSize(unsigned* width, unsigned* height) {
    // Calculate the output width and height, this is integer math so "i / 2 * 2" isn't redundant
    unsigned padding = WEIGHT_MATRIX_SIZE / 2;
    *width -= padding * 2;
    *height -= padding * 2;
}

inline __device__ unsigned char toUnsignedCharSaturated(float v) {
    if (v < 0) {
        return 0;
    }
    if (v > 0xFF) {
        return 0xFF;
    }
    return (unsigned char) v;
}

inline __device__ float4 operator *(uchar4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator +=(float4& a, float4 b) {
    return a = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void transform(unsigned char* output, cudaTextureObject_t texture, unsigned width, unsigned height) {
    // Calculate image coordinates in the output
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    // Check that we are within bounds
    if (x >= width || y >= height) {
        return;
    }
    // Sum in the convolve region of the input image
    float4 sumPixel = make_float4(0, 0, 0, 0xFF);
    for (unsigned yy = 0; yy < WEIGHT_MATRIX_SIZE; yy++) {
        float *weightLine = weightMatrix[yy];
        for (unsigned xx = 0; xx < WEIGHT_MATRIX_SIZE; xx++) {
            // Read the pixel data from the texture, weight it, and add it to the current value
            sumPixel += tex2D<uchar4>(texture, x + xx, y + yy) * weightLine[xx];
        }
    }
    // Calculate the output pixel address
    unsigned char* outputPixelAddress = output + (y * width + x) * 4;
    // Set the convolvec pixel components
    outputPixelAddress[0] = toUnsignedCharSaturated(sumPixel.x);
    outputPixelAddress[1] = toUnsignedCharSaturated(sumPixel.y);
    outputPixelAddress[2] = toUnsignedCharSaturated(sumPixel.z);
    outputPixelAddress[3] = 0xFF;
}
