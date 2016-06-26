"""Transforming data with a Convolutional Neural Network.

Example fo a simple feed-forward procedure in a Convolutional Network over
the mnist data set.

Authors:
    Lucas David   -- <ld492@drexel.edu>
    Paulo Finardi -- <ra144809@ime.unicamp.br>

License: MIT License 2016 (c)
"""

from convcuda import networks, Device
from convcuda.utils import Timer, dataset_loader

PARAMS = {
    'epochs': 1000,
    'n_batch': 10,
    'eta': .1,
    'regularization': 0,
    'verbose': True
}


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        dataset_loader.load_data()))
        print('Done (%s).' % t.get_time_hhmmss())

        for device_name in ('sequential', 'gpu'):
            print('Training our model on {%s} device...' % device_name)
            t.restart()

            with Device(device_name):
                nn = networks.Convolutional((
                    ([3, 3, 10], [2, 2], [4, 4]),
                    ([3, 3, 100], [2, 2], [4, 4]),
                ), **PARAMS)
                nn.transform(*data['train'])

            print('Done (%s).' % t.get_time_hhmmss())
            del nn

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
