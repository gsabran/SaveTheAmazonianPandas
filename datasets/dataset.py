import random
from shutil import copyfile
import os
from tqdm import tqdm
from keras import backend as K
from datasets.image_generation import ImageDataGenerator
from scipy.misc import imresize
import numpy as np

from constants import LABELS, TRAIN_DATA_DIR
from utils import files_proba, files_and_cdf_from_proba, pick, get_labels_dict, get_resized_image, addNoise, rotate_images

class Dataset(object):
	"""
	The labels used by the dataset
	"""
	labels = LABELS
	"""
	The length of the input data
	"""
	input_length = 1

	"""
	A dataset that can be fed to a model
	"""
	def __init__(self, list_files, validation_ratio=0, sessionId=None, training_files=None, validation_files=None):
		"""
		list_files: the list of paths to all images that can be used
		validation_ratio: the proportion of data to keep aside for model validation
		sessionId: a uniq string used to write output files
		training_files: a list of paths to files that should be used for training
		validation_files: a list of paths to files that should be used for validation
		"""
		self.validation_ratio = validation_ratio
		self.sessionId = sessionId

		label_idx = np.array([i for i, l in enumerate(LABELS) if l in self.labels])
		self.label_idx = label_idx
		self.labels = np.array([LABELS[k] for k in label_idx])

		self.outputs = {k: v[self.label_idx] for k, v in get_labels_dict().items()} # outputs might contain files that are unused

		list_files = self.select_files(list_files)
		if training_files is None or validation_files is None:
			train_idx = set(random.sample(range(len(list_files)), int(len(list_files) * (1 - self.validation_ratio))))
			training_files = [f for i, f in enumerate(list_files) if i in train_idx]
			validation_files = [f for i, f in enumerate(list_files) if i not in train_idx]
		
		self.training_files = training_files
		self.validation_files = validation_files

		training_files_set = set(self.training_files)
		self.proba = files_proba({f: labels for f, labels in self.outputs.items() if f in training_files_set}, self.labels)
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
			# preprocessing_function= lambda img: addNoise("gauss", img)
		)
		self._cached_training_set = None
		self._cached_validation_set = None
		self._image_normalization = None

	def select_files(self, list_files):
		"""
		Select files that can be part of the dataset
		"""
		return list_files

	def batch_generator(self, n, image_data_fmt, input_shape, balancing=True, augment=True):
		"""
		Generate batches fo size n, using images from the original data set,
			selecting them according to some tfidf proportions and rotating them
		"""
		input_length = self.input_length
		files, cdf = self.files_and_cdf
		data = self.trainingSet(image_data_fmt, input_shape)
		if input_length == 1:
			data[0] = [data[0]]

		data_dict = {}
		for i, f in enumerate(self.training_files):
			if input_shape:
				data_dict[f] = (
					[imresize(data[0][0][i], input_shape)] + [data[0][k + 1][i] for k in range(input_length - 1)],
					data[1][i]
				)
			else:
				data_dict[f] = (
					[data[0][k][i] for k in range(input_length)],
					data[1][i]
				)
		while True:
			if balancing:
				batch_files = pick(n, files, cdf)
			else:
				batch_files = random.sample(files, n)
			outputs = np.array([data_dict[f][1] for f in batch_files])
			inputs = [np.array([data_dict[f][0][k].astype(K.floatx()) for f in batch_files]) for k in range(input_length)]

			if not augment:
				if input_length == 1:
					yield (inputs[0], outputs)
				else:
					yield (inputs, outputs)
			else:
				batch_x = [np.zeros(tuple([n] + list(inputs[k].shape)[1:]), dtype=K.floatx()) for k in range(input_length)]
				for i in range(n):
					for k in range(input_length):
						batch_x[k][i] = inputs[k][i]

				for i, inp in enumerate(inputs[0]):
					x = self.image_data_generator.random_transform(inp)
					x = self.image_data_generator.standardize(x)
					batch_x[0][i] = x

				if input_length == 1:
					yield (batch_x[0], outputs)
				else:
					yield (batch_x, outputs)

	def _write_sets(self):
		if self.sessionId is None:
			return

		for file, data in zip(["training-files", "validation-files", "dataset-labels"], [self.training_files, self.validation_files, [str(i) for i in self.label_idx]]):
			filename = "train/{file}.csv".format(file=file)
			with open(filename, "w") as f:
				f.write(','.join(data))	
			copyfile(filename, "train/archive/{id}-{file}.csv".format(id=self.sessionId, file=file))

	def _xyData(self, image_data_fmt, isTraining, input_shape):
		dataset = self.training_files if isTraining else self.validation_files
		Y = []
		X = []
		print("Reading inputs...")
		with tqdm(total=len(dataset)) as pbar:
			for f in dataset:
				file = f.split('/')[-1].split('.')[0]
				X.append(self.get_input(f, TRAIN_DATA_DIR, image_data_fmt, input_shape))
				Y.append(self.outputs[file])
				pbar.update(1)
		return [np.array(X), np.array(Y)]

	def trainingSet(self, image_data_fmt, input_shape):
		"""
		The training set for a Keras model
		"""
		if self._cached_training_set is None:
			self._cached_training_set = self._xyData(image_data_fmt, True, input_shape)
		return self._cached_training_set

	def validationSet(self, image_data_fmt, input_shape):
		"""
		The validation set for a Keras model
		"""
		if self._cached_validation_set is None:
			self._cached_validation_set = self._xyData(image_data_fmt, False, input_shape)
		return self._cached_validation_set

	def get_input(self, image_name, data_dir, image_data_fmt, input_shape):
		"""
		Return the input corresponding to one image file
		"""
		return get_resized_image(
			image_name,
			data_dir,
			image_data_fmt,
			input_shape,
			self._image_normalization
		)


	def generate_tta(self, inputs):
		"""
		Generate augmented data sets for more stable prediction
		"""
		tta = []
		image_data = inputs if self.input_length == 1 else inputs[0]
		for i in range(4):
			tta.append(rotate_images(inputs, i, flip=False))
			tta.append(rotate_images(inputs, i, flip=True))
		if self.input_length == 1:
			return np.array(tta)
		for i, ds in enumerate(tta):
			tta[i] = [x for x in inputs]
			tta[i][0] = ds
		return np.array(tta)

	def set_normalization(self, image_normalization):
		"""
		Set the image normalization that should be used before feeding
		data to models
		"""
		if self._image_normalization is not None:
			raise RuntimeError("Data normalization already set")
		self._image_normalization = image_normalization
