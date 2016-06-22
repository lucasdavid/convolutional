"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import gzip

import numpy as np
import six.moves.cPickle as cPickle


def load_data(path='../data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        return cPickle.load(f, encoding='latin1')
