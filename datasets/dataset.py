import random
from shutil import copyfile
import os
from tqdm import tqdm
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from scipy.misc import imresize
import numpy as np

from constants import LABELS, TRAIN_DATA_DIR
from utils import files_proba, files_and_cdf_from_proba, pick, get_labels_dict, get_resized_image

class Dataset(object):
	labels = LABELS
	"""
	A dataset that can be fed to a model
	"""
	def __init__(self, list_files, validation_ratio, sessionId=None, training_files=None, validation_files=None):
		"""
		list_files: the list of paths to all images that can be used
		validation_ratio: the proportion of data to keep aside for model validation
		sessionId: a uniq string used to write output files
		training_files: a list of paths to files that should be used for training
		validation_files: a list of paths to files that should be used for validation
		"""
		self.outputs = get_labels_dict() # outputs might contain files that are unused
		self.validation_ratio = validation_ratio
		self.sessionId = sessionId

		if training_files is None or validation_files is None:
			train_idx = random.sample(range(len(list_files)), int(len(list_files) * (1 - self.validation_ratio)))
			training_files = [f for i, f in enumerate(list_files) if i in train_idx]
			validation_files = [f for i, f in enumerate(list_files) if i not in train_idx]

		self.training_files = training_files
		self.validation_files = validation_files

		self.proba = files_proba({f: labels for f, labels in self.outputs.items() if f in self.training_files}, self.labels)
		self.files_and_cdf = files_and_cdf_from_proba(self.proba)

		self._write_sets()
		self.image_data_generator = ImageDataGenerator(
			# featurewise_center=False,
			# samplewise_center=False,
			# featurewise_std_normalization=False,
			# samplewise_std_normalization=False,
			# zca_whitening=False,
			rotation_range=180,
			# width_shift_range=0.,
			# height_shift_range=0.,
			# shear_range=0.,
			# zoom_range=0.,
			# channel_shift_range=0.,
			fill_mode='reflect',
			# cval=0.,
			horizontal_flip=True,
			vertical_flip=True,
			# rescale=None,
			# preprocessing_function=None
		)

	def batch_generator(self, n, image_data_fmt, input_shape, balancing=True, augment=True):
		"""
		Generate batches fo size n, using images from the original data set,
			selecting them according to some tfidf proportions and rotating them
		"""
		files, cdf = self.files_and_cdf
		data = self.trainingSet(image_data_fmt, input_shape)
		data_dict = {}
		for i, f in enumerate(self.training_files):
			if input_shape:
				data_dict[f] = (imresize(data[0][i], input_shape), data[1][i])
			else:
				data_dict[f] = (data[0][i], data[1][i])
		while True:
			if balancing:
				batch_files = pick(n, files, cdf)
			else:
				batch_files = random.sample(files, n)
			outputs = np.array([data_dict[f][1] for f in batch_files])
			inputs = np.array([data_dict[f][0] for f in batch_files])

			batch_x = np.zeros(tuple([n] + list(inputs.shape)[1:]), dtype=K.floatx())
			for i, inp in enumerate(inputs):
				if augment:
					x = self.image_data_generator.random_transform(inp.astype(K.floatx()))
					x = self.image_data_generator.standardize(x)
					batch_x[i] = x
				else:
					batch_x[i] = inp.astype(K.floatx())
			yield (batch_x, outputs)

	def _write_sets(self):
		if self.sessionId is None:
			return

		with open('train/training-files.csv', 'w') as f:
			f.write(','.join(self.training_files))
		with open('train/validation-files.csv', 'w') as f:
			f.write(','.join(self.validation_files))
		copyfile('train/training-files.csv', 'train/archive/{f}-training-files.csv'.format(f=self.sessionId))
		copyfile('train/validation-files.csv', 'train/archive/{f}-validation-files.csv'.format(f=self.sessionId))

	def __xyData(self, image_data_fmt, isTraining, input_shape):
		dataset = self.training_files if isTraining else self.validation_files
		Y = []
		X = []
		print("Reading inputs")
		with tqdm(total=len(dataset)) as pbar:
			for f in dataset:
				X.append(get_resized_image(f, TRAIN_DATA_DIR, image_data_fmt, input_shape))
				file = f.split('/')[-1].split('.')[0]
				Y.append(self.outputs[file])
				pbar.update(1)
		return (np.array(X), np.array(Y))

	def trainingSet(self, image_data_fmt, input_shape):
		"""
		The training set for a Keras model
		"""
		return self.__xyData(image_data_fmt, True, input_shape)

	def validationSet(self, image_data_fmt, input_shape):
		"""
		The validation set for a Keras model
		"""
		return self.__xyData(image_data_fmt, False, input_shape)
