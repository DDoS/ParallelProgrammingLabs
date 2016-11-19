// These methods are implemented by rectify.cu, pool.cu and convolve.cu

void getOutputSize(unsigned* width, unsigned* height);

__global__ void transform(unsigned char* output, cudaTextureObject_t texture, unsigned width, unsigned height);
