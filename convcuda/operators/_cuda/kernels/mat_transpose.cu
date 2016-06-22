__global__ void
mat_transpose(float *a, float *out, int size_x, int size_y)
{
    const int i = %(N_THREADS_0)s * blockIdx.x + threadIdx.x,
              j = %(N_THREADS_1)s * blockIdx.y + threadIdx.y;

    if (i < size_x && j < size_y)
    {
        out[j * size_y + i] = a[i * size_y + j];
    }
}
