"""Training of a Fully Connected Network on All Modes
('sequential', 'vectorized' and 'gpu').

Train a simple Feed-forward neural network to recognize the mnist data set.

Authors:
    Lucas David   -- <ld492@drexel.edu>
    Paulo Finardi -- <ra144809@ime.unicamp.br>

License: MIT License 2016 (c)
"""

import numpy as np

import matplotlib.pyplot as plt

from convcuda import networks, Device
from convcuda.utils import Timer, dataset_loader

NN_PARAMS = {
    'epochs': 10,
    'n_batch': 10,
    'eta': .1,
    'regularization': 0,
    'verbose': True,
}

TEST_SIZE = 100
OPERATION_MODES = ('sequential', 'gpu', 'vectorized')


def main():
    times, scores = [], []

    t = Timer()
    try:
        print('Loading MNIST dataset...')
        train, _, test = dataset_loader.load_data()
        test = test[0][:TEST_SIZE], test[1][:TEST_SIZE]
        print('Done (%s).' % t.get_time_hhmmss())

        for mode in OPERATION_MODES:
            print('Training our model with %s operations...' % mode)
            t.restart()

            with Device(mode):
                nn = (networks
                      .FullyConnected([784, 392, 10], **NN_PARAMS)
                      .fit(*train))

            times.append(t.elapsed())
            print('Done (%s).' % t.get_time_hhmmss())

            scores.append(nn.score(*test))
            print('Score on test data-set: %.2f' % scores[-1])

        save_figures(OPERATION_MODES, scores, times)

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


def save_figures(modes, scores, times):
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
    plt.xticks(np.arange(len(modes)) + width / 2, modes, fontsize=10)

    plt.title('Time Elapsed on Training and Testing', y=1.05)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off",
                    labelleft="on")

    ax.set_xlabel('operators')
    ax.set_ylabel('time elapsed (sec)')

    plt.bar(range(len(modes)), times, width, color=(.4, 0, .2))
    fig.savefig('reports/fc_training_times.png', bbox_inches='tight')


if __name__ == '__main__':
    print(__doc__)
    main()
