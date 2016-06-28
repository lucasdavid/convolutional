"""Training of a Fully Connected Network on CPUs.

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
    'epochs': 1000,
    'n_batch': 10,
    'eta': .1,
    'regularization': 0,
    'verbose': True,
}

VALID_SIZE = 100
OPERATION_MODE = 'vectorized'


def main():
    t = Timer()
    try:
        print('Loading MNIST dataset...')
        train, (X_valid, y_valid), test = dataset_loader.load_data()
        X_valid, y_valid = X_valid[:VALID_SIZE], y_valid[:VALID_SIZE]
        print('Done (%s).' % t.get_time_hhmmss())

        print('Training our model with %s operations...' % OPERATION_MODE)
        t.restart()

        with Device(OPERATION_MODE):
            nn = (networks
                  .FullyConnected([784, 392, 10], **NN_PARAMS)
                  .fit(*train, validation_data=(X_valid, y_valid)))

            print('Done (%s).' % t.get_time_hhmmss())
            print('Score on test data-set: %.2f' % nn.score(*test))

        save_scores(nn.score_history_)

    except KeyboardInterrupt:
        print('Interrupted by user (%s)' % t.get_time_hhmmss())


def save_scores(history):
    n_scores = len(history)
    floor, ceil = np.min(history), np.max(history)
    delta = (ceil - floor) / 2

    fig = plt.figure()

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(range(10), [str(10 * x) + "%" for x in range(10)],
               fontsize=10)
    plt.xticks(fontsize=10)
    for y in range(10):
        plt.plot(range(n_scores), [y] * n_scores, "--", lw=0.5,
                 color="black", alpha=0.3)

    plt.title('Score on Evaluation Dataset Over Training Epochs')
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off",
                    labelleft="on")

    ax.set_xlabel('epochs')
    ax.set_ylabel('score')

    plt.plot(range(n_scores), [10 * s for s in history],
             lw=2.5, color=(0.84, .15, .16))
    fig.savefig('reports/fc_v_score.png', bbox_inches='tight')


if __name__ == '__main__':
    print(__doc__)
    main()
