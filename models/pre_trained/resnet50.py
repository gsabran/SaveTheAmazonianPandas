import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten

from .pretrained_model import PretrainedModel

class ResNet50CNN(PretrainedModel):
  """
  An adaptation of the ResNet50 model
  """

  def _load_pretrained_model(self):
    return ResNet50(include_top=False, input_shape=self.input_shape)

  def _add_top_dense_layers(self, x):
    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.25)(x)
    # # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    # and a logistic layer
    return Dense(len(self.data.labels), activation='sigmoid')(x)

  def _normalize_images_data(self, image):
    return preprocess_input(np.array([image]))[0]
