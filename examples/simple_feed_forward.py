"""Simple Feed Forward Example.

Train a simple Feed-forward neural network to recognize the mnist data set.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT License 2016 (c)
"""
from convcuda import Network, costs
from convcuda import operators
from convcuda.utils import Timer, mnist_loader

operators.set_mode('gpu')


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        mnist_loader.load_data_wrapper()))
        print('Done (%s).' % t.get_time_hhmmss())

        print('Training our model...')
        nn = Network([784, 50, 30, 10], cost=costs.CrossEntropyCost)
        nn.fit(data['train'], 30, 10, .1, evaluation_data=data['test'],
               monitor_training_accuracy=True, monitor_evaluation_accuracy=True)
        print('Done (%s)' % t.get_time_hhmmss())

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
