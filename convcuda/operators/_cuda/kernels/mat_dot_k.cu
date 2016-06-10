__global__ void mat_dot(float *a, float *b, float *c,
                        int a_size_x, int a_size_y,
                        int b_size_x, int b_size_y) {
    const int i = threadIdx.x,
              j = threadIdx.y;
    int k;
    float c_at_ij = 0;

    if (i < a_size_x && j < b_size_y) {
        for (k = 0; k < a_size_y; k++)
            c_at_ij += a[i * a_size_y + k] * b[k * b_size_y + j];
        c[i * b_size_y + j] = c_at_ij;
    }
}
