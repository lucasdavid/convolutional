__global__ void
mat_transpose(float *a, float *out, int size_x, int size_y)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y,
              j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size_x && j < size_y)
    {
        out[j * size_y + i] = a[i * size_y + j];
    }
}
