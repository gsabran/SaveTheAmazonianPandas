import keras
import numpy as np
from keras.layers import *
from keras.layers.merge import *


"""Layer that computes a mean between samples in two tensors according to weights computed in another vector.
E.g. if applied to k tensors `a_0` and `a_(k-1)` of shape `(batch_size, n)`, and a tensor `b` of shape `(batch_size, k)`
the output will be a tensor of shape `(batch_size, n)`
where each entry `i` will be the dot product between
`sum(a_j[i] * b[j], j)`.
"""

def weighted_average(inputs,weights,name="weighted_average"):
    if (len(weights.get_shape())!=2):
        raise ValueError('Shape error in weighted_average')
    if (len(inputs)!= weights.get_shape()[1]):
        raise ValueError('there should be as many inputs as weights in the weighted average')

    input_shape=inputs[0].get_shape().as_list()
    size=1
    for dim_size in input_shape[1:]:
        size=size*int(dim_size)
    reshaped_inputs=[]

    for inp in inputs:
        reshaped_inputs.append(Reshape((1,-1))(inp))
    print(reshaped_inputs)
    average=concatenate(reshaped_inputs,axis=1)
    result=Dot(axes=0)([average,weights])
    reshaped_result=Reshape(target_shape=input_shape[1:],name=name)(result)
    return reshaped_result

def residual_convolution(x,kernel_shape,depth,activation="relu",trainable=True):
    return add([Conv2D(depth, kernel_shape, padding='same', activation=activation)(x),x])