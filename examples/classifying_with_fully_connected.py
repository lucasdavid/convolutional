"""Simple Feed Forward Example.

Train a simple Feed-forward neural network to recognize the mnist data set.

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
                nn = networks.FullyConnected([784, 392, 10], **PARAMS)
                nn.fit(*data['train'])

            with Device('vectorized'):
                print('Score on test data-set: %.4f' % nn.score(*data['test']))

            print('Done (%s).' % t.get_time_hhmmss())
            del nn

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
