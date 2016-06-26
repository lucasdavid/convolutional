__global__ void
mat_add(float *a, float *b, float *c, int rows, int columns)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y,
              j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < rows && j < columns)
    {
        int k = i * columns + j;
        c[k] = a[k] + b[k];
    }
}
