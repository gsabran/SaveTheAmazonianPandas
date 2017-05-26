from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

from .model import Model
from constants import NUM_TAGS, NUM_WEATHER

class SimpleCNN(Model):
	"""
	A basic CNN
	"""
	def create_base_model(self):
		model = Sequential()
		model.add(Lambda(lambda x: x / 127.5 -1, output_shape=self.input_shape, input_shape=self.input_shape))
		model.add(Conv2D(32, kernel_size=(3, 3),
										 activation='relu',
										 input_shape=self.input_shape))

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(NUM_WEATHER + NUM_TAGS, activation='sigmoid'))
		self.model = model
