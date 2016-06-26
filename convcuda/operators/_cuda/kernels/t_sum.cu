__global__ void
t_sum(float *a, float *out, int n_elements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_elements)
        out[0] += a[i];
}
