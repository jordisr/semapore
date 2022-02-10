import tensorflow as tf
import numpy as np

def greedy_decode(logits):
    # doesn't merge repeated
    # returns a tf.RaggedTensor
    best_path = tf.cast(tf.math.argmax(logits, axis=2), np.int32)
    labels = tf.ragged.boolean_mask(best_path, best_path < 4)
    return labels

def pileup_to_label(x):
    x_ = x - 3
    return tf.ragged.boolean_mask(x_, x_ >= 0) % 4