import random
from shutil import copyfile
import os
from tqdm import tqdm
from keras import backend as K
from datasets.image_generation import ImageDataGenerator
from scipy.misc import imresize
import numpy as np

from constants import LABELS, TRAIN_DATA_DIR
from utils import tf_idf, idf, pick, get_labels_dict, get_resized_image, addNoise, rotate_images, get_inputs_shape

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

		self.outputs = { k: v[self.label_idx] for k, v in get_labels_dict().items() } # outputs might contain files that are unused

		list_files = self.select_files(list_files)
		if training_files is None or validation_files is None:
			train_idx = set(random.sample(range(len(list_files)), int(len(list_files) * (1 - self.validation_ratio))))
			training_files = [f for i, f in enumerate(list_files) if i in train_idx]
			validation_files = [f for i, f in enumerate(list_files) if i not in train_idx]
		
		self.training_files = training_files
		self.validation_files = validation_files

		training_files_set = set(self.training_files)
		self.idf = idf({f: labels for f, labels in self.outputs.items() if f in training_files_set}, self.labels)
		self.image_data_fmt, self.input_shape, _ = get_inputs_shape()

		self._write_sets()
		self.image_data_generator = ImageDataGenerator(
			# featurewise_center=False,
			# samplewise_center=False,
			# featurewise_std_normalization=False,
			# samplewise_std_normalization=False,
			# zca_whitening=False,
			# rotation_range=180,
			# width_shift_range=0.,
			# height_shift_range=0.,
			# shear_range=0.,
			# zoom_range=0.15,
			# channel_shift_range=0.,
			fill_mode='reflect',
			# cval=0.,
			# rescale=None,
			preprocessing_function= lambda img: rotate_images([img], np.random.choice(range(4)), flip=np.random.random() < 0.5)[0]
		)
		self._cached_training_set = None
		self._cached_validation_set = None
		self._image_normalization = None

	def select_files(self, list_files):
		"""
		Select files that can be part of the dataset
		"""
		return list_files

	def batch_generator(self, file_names, batch_size, balance=False, augment=True):
		print(balance, augment)
		"""
		Generate batches fo size batch_size, using images from the original data set,
			selecting them according to some tfidf proportions and rotating them
		"""
		if augment:
			print("Using augmented data")

		input_length = self.input_length
		
		i = 0
		while True:
			if balance:
				_tf_idf = tf_idf(file_names, self.idf)
				cdf = np.cumsum(list(map(lambda i: i[1], _tf_idf)))
				batch_files = pick(batch_size, files, cdf)
			else:
				batch_files = file_names[i:i+batch_size]
			i += 1

			inputs, outputs = self.getXY(batch_files)
			if not augment:
				yield (inputs, outputs)
			else:
				if input_length == 1:
					inputs = [inputs]
				batch_x = [np.zeros(tuple([batch_size] + list(inputs[k].shape)[1:]), dtype=K.floatx()) for k in range(input_length)]
				for i in range(batch_size):
					for k in range(1, input_length):
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

	def getXY(self, file_names):
		X = None
		Y = None
		print("Reading inputs...")
		with tqdm(total=len(file_names)) as pbar:
			for i, f in enumerate(file_names):
				file = f.split('/')[-1].split('.')[0]
				x = self.get_input(f, TRAIN_DATA_DIR)
				y = self.outputs[file]
				if X is None:
					X = np.zeros((len(file_names),) + x.shape)
					Y = np.zeros((len(file_names),) + y.shape)
				X[i] = x
				Y[i] = y
				pbar.update(1)
		return [X, Y]

	def __xyData(self, isTraining):
		dataset = self.training_files if isTraining else self.validation_files
		return self.getXY(dataset)

	def trainingSet(self):
		"""
		The training set for a Keras model
		"""
		if self._cached_training_set is None:
			self._cached_training_set = self.__xyData(isTraining=True)
		return self._cached_training_set

	def validationSet(self):
		"""
		The validation set for a Keras model
		"""
		if self._cached_validation_set is None:
			self._cached_validation_set = self.__xyData(isTraining=False)
		return self._cached_validation_set

	def get_input(self, image_name, data_dir):
		"""
		Return the input corresponding to one image file
		"""
		return get_resized_image(
			image_name,
			data_dir,
			self.image_data_fmt,
			self.input_shape,
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
