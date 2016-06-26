__global__ void
add_bias(float *a, float *bias, float *out,
         int size_x, int size_y, int size_z)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y,
              j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size_x && j < size_y)
    {
        int k = (i * size_y + j) * size_z;

        for (int c = 0; c < size_z; c++)
            out[k+c] = a[k+c] + bias[c];
    }
}
