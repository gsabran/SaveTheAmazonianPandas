import argparse
import os
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.optimizers import SGD , Adam
from keras import backend as K
from keras.applications.xception import Xception
import tensorflow as tf
from shutil import copyfile
from skimage.io import imread, imshow, imsave, show
import numpy as np
import random

from constants import LABELS
from utils import get_uniq_name


directory = os.path.dirname(os.path.abspath(__file__))

sessionId = get_uniq_name()
IMG_ROWS = 256
IMG_COLS = 256
CHANNELS = 3
NUM_TAGS = 13
NUM_WEATHER = 4
BATCH_SIZE = 24
N_GPU = 8
N_EPOCH = 10
TEST_RATIO = 0.0
TRAINED_MODEL = "train/model.h5"
os.makedirs("train/archive", exist_ok=True)
os.makedirs("train/tensorboard", exist_ok=True)

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

	def __init__(self, dataset, gpus=True):
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
		if gpus:
			model = to_multi_gpu(model)
		model.compile(loss=keras.losses.binary_crossentropy,
									optimizer=keras.optimizers.Adadelta(),
									metrics=['binary_crossentropy'])

		self.model = model

	def fit(self):
		(x_train, y_train) = self.data.training(self.image_data_fmt)

		csv_logger = CSVLogger('train/training.log')
		checkpoint = ModelCheckpoint(filepath='train/checkpoint.hdf5', monitor='binary_crossentropy', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)

		return self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=N_EPOCH,
		)

class XceptionCNN(object):
	def __init__(self, dataset, gpus=True):
		self.image_data_fmt = K.image_data_format()
		if self.image_data_fmt == 'channels_first':
			self.input_shape = (CHANNELS, IMG_ROWS, IMG_COLS)
		else:
			self.input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)
		base_model = Xception(include_top=False, input_shape=self.input_shape)

		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		# and a logistic layer -- let's say we have 200 classes
		predictions = Dense(NUM_WEATHER + NUM_TAGS, activation='sigmoid')(x)

		# this is the model we will train
		model = Model(inputs=base_model.input, outputs=predictions)

		# first: train only the top layers (which were randomly initialized)
		# i.e. freeze all convolutional InceptionV3 layers
		for layer in base_model.layers:
		    layer.trainable = False

		# compile the model (should be done *after* setting layers to non-trainable)
		model.compile(optimizer='rmsprop', loss='binary_crossentropy')

		self.model = model

	def fit(self):
		(x_train, y_train) = self.data.training(self.image_data_fmt)

		csv_logger = CSVLogger('train/training.log')
		checkpoint = ModelCheckpoint(filepath='train/checkpoint.hdf5', monitor='binary_crossentropy', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)
		# This fit is in 2 steps, first we fit the top layers we added, then we fit some top conv layers too
		self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=5,
		)
		for layer in self.model.layers[:54]:
			layer.trainable = False
		for layer in self.model.layers[54:]:
			layer.trainable = True

		return self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=N_EPOCH,
		)

		model.compile(optimizer='rmsprop', loss='binary_crossentropy')



class Dataset(object):

	def __init__(self, list_files, labels_file, train_idx=None):
		self.labels_file = labels_file
		self.labels = self._get_labels()
		self.test_ratio = TEST_RATIO
		self.files = list_files

		if train_idx is None:
			self.train_idx = random.sample(range(len(self.files)), int(len(self.files) * (1 - self.test_ratio)))
			with open('train/train-idx.csv', 'w') as f:
				f.write(','.join([str(i) for i in self.train_idx]))
		else:
			self.train_idx = train_idx

		self.train_set = [f for i, f in enumerate(list_files) if i in self.train_idx]
		self.test_set = [f for i, f in enumerate(list_files) if i not in self.train_idx]
		copyfile('train/train-idx.csv', 'train/archive/{f}-train-idx.csv'.format(f=sessionId))

	def _get_labels(self):
		labels_dict = {}
		with open(self.labels_file) as f:
			f.readline()
			for l in f:
				filename, rawTags = l.strip().split(',')
				tags = rawTags.split(' ')
				labels_dict[filename] = [1 if tag in tags else 0 for tag in LABELS]
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
	with K.get_session():
		parser = argparse.ArgumentParser(description='train model')
		parser.add_argument('-e', '--epochs', default=N_EPOCH, help='the number of epochs for fitting', type=int)
		parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, help='the number items per training batch', type=int)
		parser.add_argument('-t', '--test-ratio', default=TEST_RATIO, help='the proportion of labeled input kept aside of training for testing', type=float)
		parser.add_argument('-g', '--gpu', default=N_GPU, help='the number of gpu to use', type=int)
		parser.add_argument('-m', '--model', default='', help='A pre-built model to load', type=str)
		parser.add_argument('-c', '--cnn', default='', help='Which CNN to use. Can be "xception" or left blank for now.', type=str)

		args = vars(parser.parse_args())
		print('args', args)

		N_EPOCH = args['epochs']
		BATCH_SIZE = args['batch_size'] * N_GPU
		N_GPU = min(N_GPU, args['gpu'])
		TEST_RATIO = args['test_ratio']

		DATA_DIR = "./rawInput/train-jpg"
		LABEL_FILE = "./rawInput/train.csv"
		list_imgs = sorted(os.listdir(DATA_DIR))
		list_imgs = [os.path.join(DATA_DIR, f) for f in list_imgs]

		if args["model"] != '':
			with open('train/train-idx.csv') as f_train_idx:
				train_idx = [int(i) for i in f_train_idx.readline().split(",")]
				data = Dataset(list_imgs, LABEL_FILE, train_idx=train_idx)
		else:
			data = Dataset(list_imgs, LABEL_FILE)

		if args["cnn"] == "xception":
			cnn = XceptionCNN(data)
		else:
			cnn = CNN(data)

		if args["model"] != '':
			cnn.model.load_weights(args["model"])
		else:
			cnn = CNN(data)

		cnn.fit()
		cnn.model.save_weights(TRAINED_MODEL, overwrite=True)
		copyfile(TRAINED_MODEL, "train/archive/{f}-model.h5".format(f=sessionId))
		print('Done running')
