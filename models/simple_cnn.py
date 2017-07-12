import keras
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Sequential

from .model import Model

class SimpleCNN(Model):
	"""
	A basic CNN
	"""
	def create_base_model(self):
		inp = Input(shape=self.input_shape)
		model = Lambda(lambda x: x / 127.5 -1, output_shape=self.input_shape, input_shape=self.input_shape)(inp)
		model = Conv2D(32, kernel_size=(3, 3),
										 activation="relu",
										 input_shape=self.input_shape)(model)

		model = Conv2D(64, (3, 3), activation="relu")(model)
		model = MaxPooling2D(pool_size=(2, 2))(model)
		model = Dropout(0.25)(model)
		model = Flatten()(model)
		model = Dense(128, activation="relu")(model)
		model = Dropout(0.5)(model)
		model = Dense(len(self.data.labels), activation="sigmoid")(model)
		model = keras.models.Model(inputs=inp, outputs=model)
		self.model = model
