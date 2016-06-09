import numpy as np
import pycuda
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code = """
__global__ void  MatrixAddKernel(int *a, int *b, int *c,  int rows, int columns)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int i = row * columns + col;

    if (row < rows && col < columns)
       c[row][col] = a[row][col] + b[row][col];
}
"""


def add(a, b, out=None):
    assert a.shape == b.shape

    module = compiler.SourceModule(kernel_code)
    # get the kernel function from the compiled module
    op = module.get_function("MatrixAddKernel")

    # call the kernel on the card
    c_gpu = gpuarray.empty(a.shape, np.float32)
    op(
        gpuarray.to_gpu(a),
        gpuarray.to_gpu(b),
        c_gpu
    )

    if out is None:
        out = c_gpu.get()
    else:
        out[:] = c_gpu.get()

    return out


def dot(a, b, out=None):
    assert a.shape[1] == b.shape[0]

    module = compiler.SourceModule(kernel_code)
    # get the kernel function from the compiled module
    op = module.get_function("MatrixDotKernel")

    # call the kernel on the card
    c_gpu = gpuarray.empty(a.shape, np.float32)
    op(
        gpuarray.to_gpu(a),
        gpuarray.to_gpu(b),
        c_gpu
    )

    if out is None:
        out = c_gpu.get()
    else:
        out[:] = c_gpu.get()

    return out


ops = {
    'add': add,
    'dot': dot,
}
