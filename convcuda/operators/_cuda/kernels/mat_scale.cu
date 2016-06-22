__global__ void
mat_scale(float alpha, float *a, float *c, int rows, int columns)
{
    const int i = %(N_THREADS_1)s * blockIdx.y + threadIdx.y,
              j = %(N_THREADS_0)s * blockIdx.x + threadIdx.x;

    if (i < rows && j < columns)
        c[i * columns + j] = alpha * a[i * columns + j];
}
