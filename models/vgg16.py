import keras
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

from .model import Model
from constants import NUM_TAGS, NUM_WEATHER

class VGG16CNN(Model):
	"""
	An adaptation of the Xception model
	"""
	def create_base_model(self):
		base_model = VGG16(include_top=False, input_shape=self.input_shape)

		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.25)(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.25)(x)
		x = Dense(512, activation='relu')(x)
		x = Dropout(0.25)(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.25)(x)
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
		self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', decay=0.5)

	def compile_sgd(self):
		self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay=0.2), loss='binary_crossentropy')

	def fit(self, n_epoch, batch_size, validating=True, generating=False):
		print("Fitting top dense layers")
		super(VGG16CNN, self).fit(5, batch_size, validating=validating, generating=False)

		for layer in self.model.layers:
			layer.trainable = True

		if self.multi_gpu:
			super(VGG16CNN, self).paralelize()

		self.compile_sgd()

		print("Fitting lower conv layers")
		return super(VGG16CNN, self).fit(n_epoch, batch_size, validating=validating, generating=False)
