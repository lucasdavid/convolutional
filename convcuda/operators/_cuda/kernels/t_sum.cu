__global__ void
t_sum(float *a, float *out, int n_elements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) out[0] = 0;
    __syncthreads();

    if (i < n_elements)
        atomicAdd(out, a[i]);
}
