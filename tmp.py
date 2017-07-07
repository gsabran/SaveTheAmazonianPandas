from keras.layers.core import Dense, Flatten
from keras.layers import Input
import keras.backend as K
from keras.layers.merge import Concatenate


s = (256, 256, 3)
inputs = [Input(shape=s) for i in range(4)]
inputs = [Flatten()(inp) for inp in inputs]
inputs = [Dense(17, activation='sigmoid')(inp) for inp in inputs]
weights = Input(shape=s)
weights = Flatten()(weights)
weights = Dense(4, activation='sigmoid')(weights)

all_inputs = K.reshape(Concatenate()(inputs), (-1, 4, 17))
weights = K.reshape(weights, (-1, 1, 4))
result = K.reshape(K.batch_dot(all_inputs, weights, axes=(1, 2)), (-1, 17))
print("result shape", result.shape)
