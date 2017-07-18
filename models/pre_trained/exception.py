import keras
from keras.applications.xception import Xception, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout

from .pretrained_model import PretrainedModel

class XceptionCNN(PretrainedModel):
	"""
	An adaptation of the Xception model
	"""

	def _load_pretrained_model(self):
		return Xception(include_top=False, input_shape=self.input_shape)

	def _add_top_dense_layers(self, x):
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.5)(x)
		# let's add a fully-connected layer
		x = Dense(85, activation='relu')(x)
		x = Dropout(0.5)(x)
		# and a logistic layer
		return Dense(len(self.data.labels), activation='sigmoid')(x)

	def _prepare_deep_training(self):
		super(XceptionCNN, self)._prepare_deep_training()
		for layer in self.model.layers[:54]:
			layer.trainable = False
		for layer in self.model.layers[54:]:
			layer.trainable = True

	def _normalize_images_data(self, image):
		return preprocess_input(image)
