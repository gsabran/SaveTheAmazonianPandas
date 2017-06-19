import keras
from keras import backend as K
from keras.callbacks import CSVLogger, TensorBoard
import numpy as np

from .parallel_model import to_multi_gpu, get_gpu_max_number
from constants import LABELS
from callbacks.validation_checkpoint import ValidationCheckpoint
from callbacks.logger import Logger
from callbacks.reduce_lr_on_plateau import ReduceLROnPlateau
from utils import get_predictions, get_inputs_shape, F2Score

class Model(object):
	"""
	An absctract structure for a model that can be trained and make predictions
	"""
	def __init__(self, data, model=None, n_gpus=-1):
		"""
		data: the dataset to use
		n_gpus: The number of GPUs to use. -1 for the max, 0 for CPU only
		"""
		self.data = data
		self.n_gpus = n_gpus
		if n_gpus == -1:
			self.n_gpus = get_gpu_max_number()

		self.image_data_fmt, self.input_shape, self.img_size = get_inputs_shape()

		if model is None:
			self.model = None
			self.create_base_model()
			if self.n_gpus != 0:
				self.paralelize()
			self.compile()
		else:
			self.model = model

	def create_base_model(self):
		"""
		Create the model structure. Compilation and paralelizing are handle later
		"""
		pass

	def paralelize(self):
		"""
		Adapt the model to parallel GPU architecture
		"""
		if self.n_gpus > 1:
			self.model = to_multi_gpu(self.model, self.n_gpus)

	def compile(self):
		self.model.compile(
			loss=keras.losses.binary_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['binary_crossentropy']
		)

	def fit(self, n_epoch, batch_size, validating=True, generating=False):
		"""
		Fit the model
		n_epoch: the max number of epoch to run
		batch_size: the size of each training batch
		validating: wether validation should occur after each batch,
			if so training will terinate as soon as validation stop increasing
		generating: wether the data should be generated on the fly
		"""

		print("Fitting on data of size", self.input_shape)
		checkpoint_path = "train/checkpoint.hdf5"
		csv_logger = CSVLogger('train/training.log')
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)
		learning_rate_reduction = ReduceLROnPlateau(
			model=self,
			monitor='f2_val_score' if validating else 'acc',
			patience=3,
			verbose=1,
			factor=0.5,
			min_lr_factor=0.0001,
			mode='max',
			checkpoint_path=checkpoint_path if validating else None
		)
		callbacks = [csv_logger, tensorboard, learning_rate_reduction, Logger()]

		if validating:
			(x_train, y_train) = self.data.trainingSet(self.image_data_fmt, self.input_shape)
			(x_validate, y_validate) = self.data.validationSet(self.image_data_fmt, self.input_shape)

			def score(model, data_set, expectations):
				rawPredictions = model.predict(data_set, verbose=1)
				predictions = get_predictions(rawPredictions, self.data.labels)
				predictions = np.array([x for x in predictions])
				return np.mean([F2Score(
					prediction,
					[LABELS[i] for i, x in enumerate(expectation) if x == 1 and i in self.data.label_idx]
				) for prediction, expectation in zip(predictions, expectations)])

			validationCheckpoint = ValidationCheckpoint(
				scoring=score,
				training_input=x_train,
				training_output=y_train,
				validation_input=x_validate,
				validation_output=y_validate,
				checkpoint_path=checkpoint_path,
				patience=10
			)
			callbacks.insert(0, validationCheckpoint)

		if generating:
			print("Fitting with generated data")
			return self.model.fit_generator(
				self.data.batch_generator(batch_size, self.image_data_fmt, self.input_shape, balancing=False),
				int(len(self.data.training_files) / batch_size),
				verbose=1,
				callbacks=callbacks,
				epochs=n_epoch
			)
		else:
			(x_train, y_train) = self.data.trainingSet(self.image_data_fmt, self.input_shape)
			return self.model.fit(x_train, y_train,
				batch_size=batch_size,
				verbose=1,
				callbacks=callbacks,
				epochs=n_epoch
			)
