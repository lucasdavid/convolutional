# Convolutional CUDA

Operations such as `hadamad`, `dot` or `conv` implemented through
multiple back-ends and provided by an abstract layer.

Authors:

* Lucas Oliveira David 188972
* Paulo Ricardo Finardi 144809

## Installing and Testing

This project has [pyCUDA](https://documen.tician.de/pycuda/index.html)
as a dependency! Please refer to the installation page for info on
[how to install pyCUDA](https://wiki.tiker.net/PyCuda/Installation/Linux).

With pyCUDA installed, install convcuda:
```shell
python setup.py install --user
```

If you are trying to debug this, install the development dependencies:
```shell
pip install -r requirements-dev.txt --user
```

We have nose tests!
```shell
cd path/to/convolutional-cuda/
nosetests
```


## Examples

```python
import numpy as np
from numpy.testing import assert_array_almost_equal
from convcuda import op, Device

X, W, b = np.array(100, 10), np.array(10, 10), np.array(10,)

with Device('vectorized'):
    # Dot and vector add operations are computed with NumPY.
    y_cpu = op.add(op.dot(X, W), b)

with Device('gpu'):
    # Dot and vector add operations are computed in the GPU and transfered
    # back to us transparently.
    y_gpu = op.add(op.dot(X, W), b)

# Assertion is True.
assert_array_almost_equal(y_cpu, y_gpu)
```

This library's usage is exemplifying with Fully-connected and
Convolutional Networks implementations:

```python
import numpy as np
from convcuda.networks import FullyConnected

X, y = np.array(10, 784), np.array(10,)

mlp = FullyConnected([784, 1024, 10])
mlp.fit(X[:-10], y[:-10])

print('Score: %.2f' % mlp.score(X[-10:], y[-10:]))
```
