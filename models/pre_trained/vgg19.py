import numpy as np
from keras.applications.vgg16 import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout

from .pretrained_model import PretrainedModel

class VGG19CNN(PretrainedModel):
  """
  An adaptation of the VGG19 model
  """

  def _load_pretrained_model(self):
    inp = Input(shape=self.input_shape)
    x = BatchNormalization(input_shape=self.input_shape)(inp)
    return VGG19(include_top=False, input_shape=self.input_shape, input_tensor=x)

  def _add_top_dense_layers(self, x):
    x = Flatten()(x)
    return Dense(len(self.data.labels), activation='sigmoid')(x)

  def _normalize_images_data(self, image):
    return preprocess_input(np.array([image]))[0]
