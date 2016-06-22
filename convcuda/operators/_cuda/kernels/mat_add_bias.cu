__global__ void
add_bias(float *a, float *bias, float *out, int size_x, int size_y, int size_z,
         int n_biases)
{
    int c;
    const int i = %(N_THREADS_0)s * blockIdx.x + threadIdx.x,
              j = %(N_THREADS_1)s * blockIdx.y + threadIdx.y;

    if (i < size_x && j < size_y)
    {
        int k = i * size_y + j * size_z;
        for (c = 0; c < n_biases; c++)
        {
            out[k+c] = a[k+c] + bias[c];
        }
    }
}
