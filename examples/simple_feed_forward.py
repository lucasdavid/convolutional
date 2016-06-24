"""Simple Feed Forward Example.

Train a simple Feed-forward neural network to recognize the mnist data set.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT License 2016 (c)
"""

from convcuda import networks, Device
from convcuda.utils import Timer, dataset_loader

# operators.set_mode('sequential')
# operators.set_mode('gpu')


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

        print('Training our model...')

        with Device('gpu'):
            # Training step with the GPU.
            nn = networks.FullyConnected([784, 392, 10], **PARAMS)
            nn.fit(*data['train'])

        print('Done (%s)' % t.get_time_hhmmss())

        with Device('vectorized'):
            # Predicting step with the default (vectorized) device.
            score = nn.score(*data['test'])
        print('Score: %.4f' % score)

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


if __name__ == '__main__':
    print(__doc__)
    main()
