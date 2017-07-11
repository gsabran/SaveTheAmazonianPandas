import keras
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout

from .model import Model

class PretrainedModel(Model):
	"""
	An abstract model that uses a pretrained model as a base
	and adds dense layers on top
	"""

	def create_base_model(self):
		base_model = self._load_pretrained_model()
		self.base_model = base_model
		predictions = self._add_top_dense_layers(base_model.output)
		self.model = keras.models.Model(inputs=base_model.input, outputs=predictions)

	def _load_pretrained_model(self):
		"""
		Return the pretrained model, expecting dense layers to be added on top
		"""
		pass

	def _add_top_dense_layers(self, x):
		"""
		Adds dense layers to bridge the output of the pretrained model
		with the predictions layer
		"""
		pass

	def _prepare_shalow_training(self):
		for layer in self.base_model.layers:
			layer.trainable = False

	def _prepare_deep_training(self):
		for layer in self.base_model.layers:
			layer.trainable = True

	def paralelize(self):
		pass # paralelization done during fitting

	def fit(self, n_epoch, batch_size, validating=True, generating=False):
		print("Fitting top dense layers")
		self._prepare_shalow_training()
		super(PretrainedModel, self).fit(n_epoch, batch_size, validating=False, generating=generating)

		self._prepare_deep_training()

		if self.n_gpus > 1:
			super(PretrainedModel, self).paralelize()

		self.compile(learn_rate=0.00005)

		print("Fitting lower conv layers")
		return super(PretrainedModel, self).fit(n_epoch, batch_size, validating=validating, generating=generating)
