from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
import numpy as np


class CustomLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        print('init')
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print('input_shape', input_shape)
        # create a trainable weight variable for this layer
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print('inputs', inputs)
        print('kernel', self.kernel)
        o = K.dot(inputs, self.kernel)
        return o

    def compute_output_dim(self, input_shape):
        print('compute_output_dim')
        return input_shape[0], self.output_dim
