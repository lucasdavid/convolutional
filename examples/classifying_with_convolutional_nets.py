"""Classifying with Convolutional Networks.

Train a predicting samples from the MNIST data-set using a network composed
by convolutional and fully connected layers.

Authors:
    Lucas David   -- <ld492@drexel.edu>
    Paulo Finardi -- <ra144809@ime.unicamp.br>

License: MIT License 2016 (c)
"""

from convcuda import Device
from convcuda.networks import Composed, Convolutional, FullyConnected
from convcuda.utils import Timer, dataset_loader

PARAMS = {
    'epochs': 1000,
    'n_batch': 10,
    'eta': .1,
    'regularization': 0,
    'verbose': True,
}


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        dataset_loader.load_data()))
        print('Done (%s).' % t.get_time_hhmmss())

        for device_name in ('gpu', 'vectorized'):
            print('Training our model on {%s} device...' % device_name)
            t.restart()

            with Device(device_name):
                composed = Composed(
                    convolutional_layers=(
                        ([3, 3, 10], [2, 2], [4, 4]),
                        ([3, 3, 100], [2, 2], [4, 4]),
                    ),
                    connected_layers=[784, 392, 10],
                    **PARAMS)
                composed.fit(*data['train'])

            with Device('vectorized'):
                print('Score on test data-set: %.4f' % composed.score(*data['test']))

            print('Done (%s).' % t.get_time_hhmmss())
            del composed

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
