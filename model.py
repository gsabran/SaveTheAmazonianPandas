import keras
from keras.callbacks import CSVLogger
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras import backend as K

from skimage.io import imread, imshow, imsave, show

import numpy as np

import random

import os

IMG_ROWS = 256
IMG_COLS = 256
CHANNELS = 3
NUM_TAGS = 13
NUM_WEATHER = 4
BATCH_SIZE = 64

class CNN(object):

	def __init__(self, dataset):
		self.data = dataset
		self.image_data_fmt = K.image_data_format()
		if self.image_data_fmt == 'channels_first':
			self.input_shape = (CHANNELS, IMG_ROWS, IMG_COLS)
		else:
			self.input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)

		model = Sequential()
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

		model.compile(loss=keras.losses.binary_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['binary_crossentropy'])

		self.model = model

	def fit(self):
		csv_logger = CSVLogger('training.log')
		(x_train, y_train) = self.data.training(self.image_data_fmt)
		return self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger])

class Dataset(object):

	def __init__(self, list_files, labels_file):
		self.labels_file = labels_file
		self.labels = self._get_labels()
		self.test_ratio = 0.2
		self.files = list_files
		self.train_idx = random.sample(range(len(self.files)), int(len(self.files) * (1 - self.test_ratio)))
		self.train_set = [f for i, f in enumerate(list_files) if i in self.train_idx]
		self.test_set = [f for i, f in enumerate(list_files) if i not in self.train_idx]

	def _get_labels(self):
		labels = ['water', 'cloudy', 'partly_cloudy', 'haze', 'selective_logging', 'agriculture', 'blooming', 'cultivation', 'habitation', 'road', 'bare_ground', 'clear', 'conventional_mine', 'artisinal_mine', 'slash_burn', 'primary', 'blow_down']
		labels.sort()

		labels_dict = {}

		with open(self.labels_file) as f:
			f.readline()
			for l in f:
				filename, rawTags = l.strip().split(',')
				tags = rawTags.split(' ')
				labels_dict[filename] = [1 if tag in tags else 0 for tag in labels]
		return labels_dict

	def training(self, image_data_fmt):
		labels = []
		training = []
		for f in self.train_set:
			img = imread(f)
			if image_data_fmt == 'channels_first':
				img = img.reshape((CHANNELS, IMG_ROWS, IMG_COLS))
			training.append(img)
			file = f.split('/')[-1].split('.')[0]
			labels.append(self.labels[file])

		return (np.array(training), np.array(labels))

	def testing(self):
		pass

if __name__ == "__main__":
	DATA_DIR = "./train-jpg-sample"
	LABEL_FILE = "train.csv"

	list_imgs = os.listdir(DATA_DIR)
	list_imgs = [os.path.join(DATA_DIR, f) for f in list_imgs]
	data = Dataset(list_imgs, LABEL_FILE)
	cnn = CNN(data)
	cnn.fit()
	cnn.model.save_weights("model.h5", overwrite=True)

