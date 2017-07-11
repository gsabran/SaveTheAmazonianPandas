import keras
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

from .model import Model

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
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.25)(x)
		# and a logistic layer
		predictions = Dense(len(self.data.labels), activation='sigmoid')(x)

		# this is the model we will train
		model = keras.models.Model(inputs=base_model.input, outputs=predictions)

		# first: train only the top layers (which were randomly initialized)
		# i.e. freeze all convolutional InceptionV3 layers
		for layer in base_model.layers:
			layer.trainable = False
		self.model = model

	def paralelize(self):
		pass # paralelization done during fitting

	def compile(self, learn_rate=0.001):
		opt = Adam(lr=learn_rate)
		self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	def fit(self, n_epoch, batch_size, validating=True, generating=False):
		print("Fitting top dense layers")
		super(VGG16CNN, self).fit(30, batch_size, validating=validating, generating=generating)

		for layer in self.model.layers:
			layer.trainable = True

		if self.multi_gpu:
			super(VGG16CNN, self).paralelize()

		self.compile(learn_rate=0.00005)

		print("Fitting lower conv layers")
		return super(VGG16CNN, self).fit(n_epoch, batch_size, validating=validating, generating=generating)

	def _normalize_images_data(self, image):
		"""
		Apply some preprocessing to images that are (3, w, h) floats on a scale of 0 to 255
		"""
		return preprocess_input([image])[0]
