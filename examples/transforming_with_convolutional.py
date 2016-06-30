"""Transforming data with a Convolutional Neural Network.

Example fo a simple feed-forward procedure in a Convolutional Network over
the mnist data set.

Authors:
    Lucas David   -- <ld492@drexel.edu>
    Paulo Finardi -- <ra144809@ime.unicamp.br>

License: MIT License 2016 (c)
"""

import matplotlib.pyplot as plt
import numpy as np

from convcuda import networks, Device
from convcuda.utils import Timer, dataset_loader

PARAMS = {
    'verbose': True
}

OPERATION_MODES = ('sequential', 'vectorized', 'gpu',)
DATA_SIZE = 300


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        data = dict(zip(('train', 'valid', 'test'),
                        dataset_loader.load_data()))
        X, y = data['test']
        X, y = X[:DATA_SIZE], y[:DATA_SIZE]

        print('Done (%s).' % t.get_time_hhmmss())

        times = []

        for device_name in OPERATION_MODES:
            if device_name == 'sequential':
                times.append(0)
                continue

            print('Training our model on {%s} device...' % device_name)
            t.restart()

            with Device(device_name):
                cnn = networks.Convolutional((
                    ([3, 3, 3], [2, 2], [4, 4]),
                    ([3, 3, 10], [2, 2], [4, 4]),
                ), **PARAMS)
                cnn.transform(X, y)

            times.append(t.elapsed())
            print('Done (%s).' % t.get_time_hhmmss())

        save_figures(OPERATION_MODES, times)

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


def save_figures(modes, times):
    width = .5
    fig = plt.figure()
    ax = plt.subplot(111)

    fig.subplots_adjust(top=1)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(fontsize=10)
    plt.xticks(np.arange(len(modes)) + width / 2, modes, fontsize=14)

    plt.title('Time Elapsed on Convolutional Feed-forward', y=1.05)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off",
                    labelleft="on")

    ax.set_xlabel('operators')
    ax.set_ylabel('time elapsed (sec)')

    plt.bar(range(len(modes)), times, width, color=(.4, 0, .2))
    fig.savefig('reports/conv_transf_times.png', bbox_inches='tight')


if __name__ == '__main__':
    print(__doc__)
    main()
