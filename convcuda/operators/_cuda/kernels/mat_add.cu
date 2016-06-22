__global__ void
mat_add(float *a, float *b, float *c, int rows, int columns)
{
    const int i = %(N_THREADS_0)s * blockIdx.y + threadIdx.y,
              j = %(N_THREADS_1)s * blockIdx.x + threadIdx.x;

    if (i < rows && j < columns)
    {
        int k = i * columns + j;
        c[k] = a[k] + b[k];
    }
}
