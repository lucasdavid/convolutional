__global__ void
mat_scale(float alpha, float *a, float *c, int rows, int columns, int depth)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y,
              j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < rows && j < columns)
        for (int k = 0; k < depth; k++)
            c[(i * columns + j)*depth + k] = alpha * a[(i * columns + j)*depth + k];
}
