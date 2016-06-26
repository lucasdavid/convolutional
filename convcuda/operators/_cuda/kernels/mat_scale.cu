__global__ void
mat_scale(float alpha, float *a, float *c, int rows, int columns)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y,
              j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < rows && j < columns)
        c[i * columns + j] = alpha * a[i * columns + j];
}
