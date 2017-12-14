__global__ void
mat_add(float *a, float *b, float *c, int limit)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < limit)
    {
        c[i] = a[i] + b[i];
    }
}
