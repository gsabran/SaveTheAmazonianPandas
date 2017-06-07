import keras
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout

from .model import Model
from constants import NUM_TAGS, NUM_WEATHER

class XceptionCNN(Model):
	"""
	An adaptation of the Xception model
	"""
	def create_base_model(self):
		base_model = Xception(include_top=False, input_shape=self.input_shape)

		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.5)(x)
		# let's add a fully-connected layer
		x = Dense(85, activation='relu')(x)
		x = Dropout(0.5)(x)
		# and a logistic layer -- let's say we have 200 classes
		predictions = Dense(NUM_WEATHER + NUM_TAGS, activation='sigmoid')(x)

		# this is the model we will train
		model = keras.models.Model(inputs=base_model.input, outputs=predictions)

		# first: train only the top layers (which were randomly initialized)
		# i.e. freeze all convolutional InceptionV3 layers
		for layer in base_model.layers:
			layer.trainable = False
		self.model = model

	def paralelize(self):
		pass # paralelization done during fitting

	def compile(self):
		self.model.compile(optimizer='rmsprop', loss='binary_crossentropy')

	def fit(self, n_epoch, batch_size, validating=True, generating=False):
		print("Fitting top dense layers")
		super(XceptionCNN, self).fit(5, batch_size, validating=False, generating=generating)

		for layer in self.model.layers[:54]:
			layer.trainable = False
		for layer in self.model.layers[54:]:
			layer.trainable = True

		if self.n_gpus > 0:
			super(XceptionCNN, self).paralelize()

		self.compile()

		print("Fitting lower conv layers")
		return super(XceptionCNN, self).fit(n_epoch, batch_size, validating=validating, generating=generating)
