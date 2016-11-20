#define WEIGHT_MATRIX_SIZE 3

__device__ float weightMatrix[WEIGHT_MATRIX_SIZE][WEIGHT_MATRIX_SIZE] = {
    {1, 2, -1},
    {2, 0.25, -2},
    {1, -2, -1}
};
