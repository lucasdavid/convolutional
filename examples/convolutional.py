"""Simple Feed Forward Example.

Train a simple Feed-forward neural network to recognize the mnist data set.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT License 2016 (c)
"""
import numpy

from convcuda import operators
from convcuda.utils import Timer, dataset_loader

# operators.set_mode('sequential')
operators.set_mode('gpu')

from convcuda.networks import FullyConnected


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        dataset_loader.load_data_wrapper()))
        X, y = data['train']
        print('Done (%s).' % t.get_time_hhmmss())

        print('Training our model...')

        nn = ComposedNetwork(
            ConvolutionalNeuralNetwork((
                [3, 3, 30],
                [3, 3, 100],
            )),
            FullyConnected([784, 392, 196]),
        )

        nn.fit(X, y)
        print('Done (%s)' % t.get_time_hhmmss())
        print('Accuracy: %.2f' % nn.accuracy(data['test']))

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
