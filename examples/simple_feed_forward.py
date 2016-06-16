"""Simple Feed Forward Example.

Train a simple Feed-forward neural network to recognize the mnist data set.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT License 2016 (c)
"""
import numpy

from convcuda import operators
from convcuda.utils import Timer, mnist_loader

operators.set_mode('gpu')

from sklearn.neural_network import MLPClassifier


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        mnist_loader.load_data_wrapper()))
        X, y = data['train']
        print('Done (%s).' % t.get_time_hhmmss())

        print('Training our model...')
        nn = MLPClassifier([784, 392, 196], verbose=True)
        nn.fit(X, y)

        print('Done (%s)' % t.get_time_hhmmss())

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
