from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tf.__path__="D:\Anaconda3\Lib\site-packages\tensorflow"
# pylint: disable=wildcard-import
from tensorflow.python import *
# pylint: enable=wildcard-import
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
