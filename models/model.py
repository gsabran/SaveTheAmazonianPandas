import keras
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import numpy as np

from .parallel_model import to_multi_gpu, get_gpu_max_number
from constants import CHANNELS, IMG_ROWS, IMG_COLS, LABELS
from validate_model import F2Score
from validation_checkpoint import ValidationCheckpoint
from utils import get_predictions

class Model(object):
	"""
	An absctract structure for a model that can be trained and make predictions
	"""
	def __init__(self, data, multi_gpu=True, tiff_model=False):
		"""
		data: the dataset to use
		multi_gpu: wether the model should use several GPUs or not
		"""
		if tiff_model:
			CHANNELS += 1

		self.data = data
		self.multi_gpu = multi_gpu
		self.model = None

		self.image_data_fmt = K.image_data_format()
		if self.image_data_fmt == 'channels_first':
			self.input_shape = (CHANNELS, IMG_ROWS, IMG_COLS)
		else:
			self.input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)

		self.create_base_model()
		self.n_gpus = get_gpu_max_number()
		if self.multi_gpu:
			self.paralelize()
		self.compile()
		print("Model", self.n_gpus)

	def create_base_model(self):
		"""
		Create the model structure. Compilation and paralelizing are handle later
		"""
		pass

	def paralelize(self):
		"""
		Adapt the model to parallel GPU architecture
		"""
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
		(x_train, y_train) = self.data.trainingSet(self.image_data_fmt)
		print("Fitting on data of size", x_train.shape, y_train.shape)

		csv_logger = CSVLogger('train/training.log')
		checkpoint = ModelCheckpoint(filepath='train/checkpoint.hdf5', monitor='binary_crossentropy', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='train/tensorboard', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)
		callbacks = [csv_logger, checkpoint, tensorboard]

		if validating:
			(x_validate, y_validate) = self.data.validationSet(self.image_data_fmt)

			def score(model, data_set, expectations):
				rawPredictions = model.predict(data_set, verbose=1)
				predictions = get_predictions(np.array(rawPredictions))
				predictions = np.array([x for x in predictions])
				return np.mean([F2Score(
					prediction,
					[LABELS[i] for i, x in enumerate(expectation) if x == 1]
				) for prediction, expectation in zip(predictions, expectations)])

			validationCheckpoint = ValidationCheckpoint(scoring=score, validation_input=x_validate, validation_output=y_validate, patience=5)
			callbacks.append(validationCheckpoint)

		if generating:
			return self.model.fit_generator(
				self.data.batch_generator(batch_size, self.image_data_fmt),
				int(len(self.data.training_files) / batch_size),
				verbose=1,
				callbacks=callbacks,
				epochs=n_epoch
			)
		else:
			return self.model.fit(x_train, y_train,
				batch_size=batch_size,
				verbose=1,
				callbacks=callbacks,
				epochs=n_epoch
			)
