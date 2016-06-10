__global__ void mat_add(float *a, float *b, float *c,
                        int size_x, int size_y) {
    const int i = threadIdx.x,
              j = threadIdx.y,
              real_pos = i * size_y + j;

    if (i < size_x && j < size_y)
        c[real_pos] = a[real_pos] + b[real_pos];
}
