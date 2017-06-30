import keras.backend as K
from keras.layers.merge import _Merge
from keras.layers.merge import Concatenate

class WeightedAverage(_Merge):
    """Layer that computes a mean between samples in two tensors according to weights computed in another vector.
    E.g. if applied to k tensors `a_0` and `a_k` of shape `(batch_size, n)`, and a tensor `b` of shape `(batch_size, k)`
    the output will be a tensor of shape `(batch_size, n)`
    where each entry `i` will be the dot product between
    `sum(a_j[i] * b[j], j)`.
    """
    def __init__(self, **kwargs):
        self.weight_tensor = None
        super(WeightedAverage, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        if self.weight_tensor is None:
            raise ValueError('A `WeightedAverage` layer should have a weight layer')
        shape = inputs[0].get_shape().as_list()
        k = shape[1]
        n = len(inputs)
        all_inputs = K.reshape(Concatenate()(inputs), (-1, n, k))
        res = K.reshape(K.batch_dot(all_inputs, self.weight_tensor, axes=(1, 2)), (-1, k))
        return res

    def call(self, inputs, weights):
        if weights.shape[1] != len(inputs):
            raise ValueError('A `WeightedAverage` weights layer should have as many'
                             'outputs as there are inputs.')
        shape = weights.get_shape().as_list()
        self.weight_tensor = K.reshape(weights, (-1, 1, shape[1]))
        return super(WeightedAverage, self).call(inputs)
