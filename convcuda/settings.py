# Options are: ('sequential', 'vectorized', 'gpu')
DEFAULT_OPERATION_MODE = 'vectorized'

# CUDA Options.
MAX_THREADS_PER_BLOCK = 1024
MAX_BLOCKS_PER_GRID = 65535

block = (32, 32, 1)
