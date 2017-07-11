import numpy as np
from keras.applications.inception_v3 import InceptionV3 as BaseCNN, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout

from .pretrained_model import PretrainedModel

class InceptionV3(PretrainedModel):
	"""
	An adaptation of the Inception model
	"""

	def _load_pretrained_model(self):
		return BaseCNN(include_top=False, input_shape=self.input_shape)

	def _add_top_dense_layers(self, x):
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.25)(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.25)(x)
		# and a logistic layer
		return Dense(len(self.data.labels), activation='sigmoid')(x)

	def _normalize_images_data(self, image):
		return preprocess_input(image)
