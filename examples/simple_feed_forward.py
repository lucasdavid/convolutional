"""Simple Feed Forward Example.

Train a simple Feed-forward neural network to recognize the mnist data set.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT License 2016 (c)
"""
import logging
import sys

from networks import Network, costs
from networks.utils import Timer, mnist_loader


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        mnist_loader.load_data_wrapper()))
        print('Done (%s).' % t.get_time_hhmmss())

        print('Training Neural Network...')
        nn = Network([784, 30, 10], cost=costs.CrossEntropyCost)
        nn.fit(data['train'], 30, 10, .1, evaluation_data=data['test'],
               monitor_evaluation_accuracy=True)

        # Flush all print calls made by SGD algorithm.
        sys.stdout.flush()

    except KeyboardInterrupt: logging.info('interrupted by user')
    except Exception as e: print(e)
    finally: print('done (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    sys.stdout.flush()

    main()
