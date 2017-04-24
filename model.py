import argparse
import os
directory = os.path.dirname(os.path.abspath(__file__))

import keras
from keras.callbacks import CSVLogger
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.optimizers import SGD , Adam
from keras import backend as K
import tensorflow as tf

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
N_GPU = 8

# see https://github.com/fchollet/keras/issues/2436#issuecomment-291874528
def slice_batch(x, n_gpus, part):
	"""
	Divide the input batch into [n_gpus] slices, and obtain slice no. [part].
	i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
	"""
	sh = K.shape(x)
	L = sh[0] // n_gpus
	if part == n_gpus - 1:
		return x[part*L:]
	return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=N_GPU):
	"""Given a keras [model], return an equivalent model which parallelizes
	the computation over [n_gpus] GPUs.

	Each GPU gets a slice of the input batch, applies the model on that slice
	and later the outputs of the models are concatenated to a single tensor, 
	hence the user sees a model that behaves the same as the original.
	"""
	with tf.device('/cpu:0'):
		x = Input(model.input_shape[1:])

	towers = []
	for g in range(n_gpus):
		with tf.device('/gpu:' + str(g)):
			slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
			towers.append(model(slice_g))

	with tf.device('/cpu:0'):
		merged = Concatenate(axis=0)(towers)

	return Model(inputs=[x], outputs=[merged])

class CNN(object):

	def __init__(self, dataset):
		self.data = dataset
		self.image_data_fmt = K.image_data_format()
		if self.image_data_fmt == 'channels_first':
			self.input_shape = (CHANNELS, IMG_ROWS, IMG_COLS)
		else:
			self.input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)

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
		model = to_multi_gpu(model)
		model.compile(loss=keras.losses.binary_crossentropy,
									optimizer=keras.optimizers.Adadelta(),
									metrics=['binary_crossentropy'])

		self.model = model

	def fit(self, epochs):
		csv_logger = CSVLogger('training.log')
		(x_train, y_train) = self.data.training(self.image_data_fmt)
		return self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger],
			epochs=epochs,
		)

class Dataset(object):

	def __init__(self, list_files, labels_file):
		self.labels_file = labels_file
		self.labels = self._get_labels()
		self.test_ratio = 0.2
		self.files = list_files
		self.train_idx = random.sample(range(len(self.files)), int(len(self.files) * (1 - self.test_ratio)))
		self.train_set = [f for i, f in enumerate(list_files) if i in self.train_idx]
		self.test_set = [f for i, f in enumerate(list_files) if i not in self.train_idx]
		with open(directory + '/train-idx.csv', 'w') as f:
			f.write(','.join([str(i) for i in self.train_idx]))

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

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('-e', '--epochs', default=10, help='the number of epoch for fitting', type=int)

if __name__ == "__main__":
	args = vars(parser.parse_args())
	print('args', args)

	epochs = args['epochs']

	DATA_DIR = "./rawInput/train-jpg"
	LABEL_FILE = "./rawInput/train.csv"

	list_imgs = os.listdir(DATA_DIR)
	list_imgs = [os.path.join(DATA_DIR, f) for f in list_imgs]
	data = Dataset(list_imgs, LABEL_FILE)
	cnn = CNN(data)
	cnn.fit(epochs)
	cnn.model.save_weights("model.h5", overwrite=True)

