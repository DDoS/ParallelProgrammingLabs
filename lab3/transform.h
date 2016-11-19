// This method is implemented by rectify.cu, pool.cu and convolve.cu
__global__ void transform(unsigned char* output, cudaTextureObject_t texture, int width, int height);
