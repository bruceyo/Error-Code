from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import numpy as np
import os
import sys
import tensorflow as tf
from six.moves import xrange
import cifar10_input
import matplotlib.pyplot as plt


data_dir = os.path.join('/mnt/nas/ntu-rgbd/other_Datasets/cifar10_data', 'cifar-10-batches-bin')
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
# Create a queue that produces the filenames to read.
filename_queue = tf.train.string_input_producer(filenames)
read_input = cifar10_input.read_cifar10(filename_queue)

print(type(tf.Session().run(read_input.uint8image)))

'''
# Create a queue that produces the filenames to read.
filename_queue = tf.train.string_input_producer(filenames)
read_input = cifar10_input.read_cifar10(filename_queue)

from PIL import Image
import numpy as np

im = Image.open('images.jpeg')
im = (np.array(im))

r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()
label = [1]

out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
out.tofile("out.bin")
'''
