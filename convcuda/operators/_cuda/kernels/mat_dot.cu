__global__ void
mat_dot(float *a, float *b, float *c,
        int a_rows, int a_columns, int b_rows, int b_columns)
{
    const int i = %(N_THREADS_0)s * blockIdx.y + threadIdx.y,
              j = %(N_THREADS_1)s * blockIdx.x + threadIdx.x;

    if (i < a_rows && j < b_columns)
    {
        float c_at_ij = 0;
        for (int k = 0; k < a_columns; k++)
            c_at_ij += a[i * a_columns + k] * b[k * b_columns + j];
        c[i * b_columns + j] = c_at_ij;
    }
}
