"""Training of a Composed Neural Network.

Trains a Neural Network composed by two convolutional layers and
three fully connected layers using the gpu.

Authors:
    Lucas David   -- <ld492@drexel.edu>
    Paulo Finardi -- <ra144809@ime.unicamp.br>

License: MIT License 2016 (c)
"""

from convcuda import Device
from convcuda.networks import Composed
from convcuda.utils import Timer, dataset_loader

PARAMS = {
    'epochs': 1000,
    'n_batch': 10,
    'eta': .1,
    'regularization': 0,
    'verbose': True,
}

OPERATION_MODE = 'gpu'


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        dataset_loader.load_data()))
        print('Done (%s).' % t.get_time_hhmmss())

        print('Training our model on {%s} device...' % OPERATION_MODE)
        t.restart()

        with Device(OPERATION_MODE):
            composed = (Composed(
                convolutional_layers=(
                    ([3, 3, 10], [2, 2], [4, 4]),
                    ([3, 3, 100], [2, 2], [4, 4]),
                ),
                connected_layers=(
                    78400, 392, 10
                ),
                **PARAMS).fit(*data['train']))

        print('Done (%s).' % t.get_time_hhmmss())
        score = composed.score(*data['test'])
        print('Score on test data-set: %.4f' % score)

        del composed

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
