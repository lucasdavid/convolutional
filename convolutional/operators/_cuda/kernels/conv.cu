__global__ void
conv(float *t, float *tk, float *out,
     int t_rows, int t_columns, int n_channels,
     int k_rows, int k_columns, int n_kernels)
{
    const int i_out = blockDim.y * blockIdx.y + threadIdx.y,
              j_out = blockDim.x * blockIdx.x + threadIdx.x;

    int i0 = i_out - k_rows/2,
        j0 = j_out - k_columns/2;

    if (i_out < t_rows && j_out < t_columns)
        for (int k = 0; k < n_kernels; k++)
        {
            float convolution = 0;

            for (int m = 0; m < k_rows; m++)
                for (int n = 0; n < k_columns; n++)
                    for (int c = 0; c < n_channels; c++)
                        if (-1 < i0 + m && i0 + m < t_rows &&
                            -1 < j0 + n && j0 + n < t_columns)
                            convolution += t[((i0 + m)*t_columns + (j0 + n))*n_channels + c]
                                           * tk[(m*k_columns + n)*n_kernels + k];

            out[(i_out*t_columns + j_out)*n_kernels + k] = convolution;
        }
}
