__global__ void mat_hadmard_k(float *dest, float *a, float *b, int size)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
