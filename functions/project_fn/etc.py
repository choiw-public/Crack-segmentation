import tensorflow as tf
import numpy as np


def count_trainable():
    all_trainables = tf.trainable_variables()
    parameters = 0
    for variable in all_trainables:
        parameters += np.prod([int(para) for para in variable.shape])
    print(parameters)
