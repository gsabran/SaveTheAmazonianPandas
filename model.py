from tqdm import tqdm
import argparse
import os
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from validation_checkpoint import ValidationCheckpoint
from keras.models import model_from_json, load_model
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

from constants import LABELS, ORIGINAL_DATA_DIR, DATA_DIR, ORIGINAL_LABEL_FILE
from utils import get_uniq_name, get_generated_images, get_predictions, files_proba, files_and_cdf_from_proba, pick
from validate_model import F2Score

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

class ModelCNN(object):
	pass

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

class CNN(ModelCNN):

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
		(x_validate, y_validate) = self.data.validation(self.image_data_fmt)

		csv_logger = CSVLogger('train/training.log')
		checkpoint = ModelCheckpoint(filepath='train/checkpoint.hdf5', monitor='binary_crossentropy', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)

		def score(model, data_set, expectations):
			rawPredictions = model.predict(data_set, verbose=1)
			predictions = get_predictions(np.array(rawPredictions))
			predictions = np.array([x for x in predictions])
			return np.mean([F2Score(
				prediction,
				[LABELS[i] for i, x in enumerate(expectation) if x == 1]
			) for prediction, expectation in zip(predictions, expectations)])

		validationCheckpoint = ValidationCheckpoint(scoring=score, validation_input=x_validate, validation_output=y_validate)

		return self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard, validationCheckpoint],
			epochs=N_EPOCH
		)

class XceptionCNN(ModelCNN):
	def __init__(self, dataset, gpus=True):
		self.data = dataset
		self.gpus = gpus
		self.image_data_fmt = K.image_data_format()
		if self.image_data_fmt == 'channels_first':
			self.input_shape = (CHANNELS, IMG_ROWS, IMG_COLS)
		else:
			self.input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)
		base_model = Xception(include_top=False, input_shape=self.input_shape)

		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.25)(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
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
		print("Fitting on data of size", x_train.shape, y_train.shape)

		csv_logger = CSVLogger('train/training.log')
		checkpoint = ModelCheckpoint(filepath='train/checkpoint.hdf5', monitor='binary_crossentropy', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)
		# This fit is in 2 steps, first we fit the top layers we added, then we fit some top conv layers too
		print("Fitting top dense layers")
		self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=5
		)

		for layer in self.model.layers[:54]:
			layer.trainable = False
		for layer in self.model.layers[54:]:
			layer.trainable = True

		if self.gpus:
			self.model = to_multi_gpu(self.model)

		self.model.compile(optimizer='rmsprop', loss='binary_crossentropy')

		print("Fitting lower conv layers")
		return self.model.fit(x_train, y_train,
			batch_size=BATCH_SIZE,
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=N_EPOCH
		)
	def fit_generator(self):
		csv_logger = CSVLogger('train/training.log')
		checkpoint = ModelCheckpoint(filepath='train/checkpoint.hdf5', monitor='binary_crossentropy', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)
		# This fit is in 2 steps, first we fit the top layers we added, then we fit some top conv layers too
		print("Fitting top dense layers")
		self.model.fit_generator(
			self.data.batch_generator(BATCH_SIZE, self.image_data_fmt),
			len(self.data.train_set),
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=5
		)

		for layer in self.model.layers[:54]:
			layer.trainable = False
		for layer in self.model.layers[54:]:
			layer.trainable = True

		if self.gpus:
			self.model = to_multi_gpu(self.model)

		self.model.compile(optimizer='rmsprop', loss='binary_crossentropy')

		print("Fitting lower conv layers")
		return self.model.fit_generator(
			self.data.batch_generator(BATCH_SIZE, self.image_data_fmt),
			len(self.data.train_set),
			verbose=1,
			callbacks=[csv_logger, checkpoint, tensorboard],
			epochs=N_EPOCH
		)

class Dataset(object):

	def __init__(self, list_files, labels_file, training_files=None, validation_files=None):
		self.labels_file = labels_file
		self.labels = self._get_labels()
		self.test_ratio = TEST_RATIO

		if training_files is None or validation_files is None:
			train_idx = random.sample(range(len(list_files)), int(len(list_files) * (1 - self.test_ratio)))
			self.training_files = [f for i, f in enumerate(list_files) if i in train_idx]
			self.validation_files = [f for i, f in enumerate(list_files) if i not in train_idx]

		self.train_set = self.training_files
		self.test_set = self.validation_files

		self.proba = files_proba({f: labels for f, labels in self.labels.items() if f in self.train_set}, LABELS)
		self.files_and_cdf = files_and_cdf_from_proba(self.proba)

		self.write_sets()

	def batch_generator(self, n, image_data_fmt):
		files, cdf = self.files_and_cdf
		data = self.__xyData(image_data_fmt, True)
		dataset = self.train_set if isTraining else self.test_set
		data_dict = {}
		for i, f in enumerate(dataset):
			data_dict[f] = (data[0][i], data[1][i])
		while True:
			batch_files = pick(n, files, cdf)
			inputs = np.array([data_dict[f][0] for f in batch_files])
			targets = np.array([data_dict[f][1] for f in batch_files])
			yield (inputs, targets)

	def write_sets(self):
		with open('train/training-files.csv', 'w') as f:
			f.write(','.join(self.training_files))
		with open('train/validation-files.csv', 'w') as f:
			f.write(','.join(self.validation_files))
		copyfile('train/training-files.csv', 'train/archive/{f}-training-files.csv'.format(f=sessionId))
		copyfile('train/validation-files.csv', 'train/archive/{f}-validation-files.csv'.format(f=sessionId))

	def _get_labels(self):
		labels_dict = {}
		with open(self.labels_file) as f:
			f.readline()
			for l in f:
				filename, rawTags = l.strip().split(',')
				tags = rawTags.split(' ')
				bool_tags = [1 if tag in tags else 0 for tag in LABELS]
				file = filename.split('/')[-1].split('.')[0]
				labels_dict[file] = bool_tags
		return labels_dict

	def __xyData(self, image_data_fmt, isTraining):
		dataset = self.train_set if isTraining else self.test_set
		Y = []
		X = []
		print("Reading inputs")
		with tqdm(total=len(dataset)) as pbar:
			for f in dataset:
				img = imread(f)
				if image_data_fmt == 'channels_first':
					img = img.reshape((CHANNELS, IMG_ROWS, IMG_COLS))
				X.append(img)
				file = f.split('/')[-1].split('.')[0]
				Y.append(self.labels[file])
				pbar.update(1)
		return (np.array(X), np.array(Y))

	def training(self, image_data_fmt):
		return self.__xyData(image_data_fmt, True)

	def validation(self, image_data_fmt):
		return self.__xyData(image_data_fmt, False)

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
		parser.add_argument('--data-proportion', default=1, help='A proportion of the data to use for training', type=float)

		args = vars(parser.parse_args())
		print('args', args)

		N_EPOCH = args['epochs']
		BATCH_SIZE = args['batch_size'] * N_GPU
		N_GPU = min(N_GPU, args['gpu'])
		TEST_RATIO = args['test_ratio']

		list_imgs = [f.split(".")[0] for f in sorted(os.listdir(ORIGINAL_DATA_DIR))]
		list_imgs = random.sample(list_imgs, int(len(list_imgs) * args['data_proportion']))

		data = Dataset(list_imgs, ORIGINAL_LABEL_FILE)
		if args["cnn"] == "xception":
			print("Using Xception architecture")
			cnn = XceptionCNN(data)
		else:
			print("Using simple model architecture")
			cnn = CNN(data)

		if args["model"] != '':
			print("Loading model {m}".format(m=args['model']))
			with open('train/training-files.csv') as f_training_files:
				training_files = f_training_files.readline().split(",")
				data = Dataset(list_imgs, ORIGINAL_LABEL_FILE, training_files=training_files)
			cnn.model = load_model(args['model'])

		cnn.fit()
		cnn.model.save(TRAINED_MODEL, overwrite=True)
		copyfile(TRAINED_MODEL, "train/archive/{f}-model.h5".format(f=sessionId))
		print('Done running')
