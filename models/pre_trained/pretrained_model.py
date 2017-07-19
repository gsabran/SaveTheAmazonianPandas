import keras
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam, SGD

from ..model import Model

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
		for layer in self.model.layers:
			layer.grad_multiplier = 10.
		for layer in self.base_model.layers:
			layer.trainable = False
			layer.grad_multiplier = 1.

	def _prepare_deep_training(self):
		for layer in self.model.layers:
			layer.grad_multiplier = 10.
		for layer in self.base_model.layers:
			layer.trainable = True
			layer.grad_multiplier = 1.

	def paralelize(self):
		pass # paralelization done during fitting

	def compile(self, learn_rate=0.0001):
		opt = Adam(lr=learn_rate)
		# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_crossentropy', 'accuracy'])

	def fit(self, n_epoch, batch_size, validating=True, generating=False):
		# 0-10: 0.001

		# 10-20: 0.0001

		# 20-25: 0.00001
		print("Fitting top dense layers")
		self._prepare_shalow_training()
		self.compile(learn_rate=0.001)
		super(PretrainedModel, self).fit(10, batch_size, validating=False, generating=generating)

		self._prepare_deep_training()

		print("Fitting lower conv layers I")
		if self.n_gpus > 1:
			super(PretrainedModel, self).paralelize()
		self.compile(learn_rate=0.0001)
		super(PretrainedModel, self).fit(30, batch_size, validating=validating, generating=generating)

		# print("Fitting lower conv layers II")
		# self.compile(learn_rate=0.0001)
		# super(PretrainedModel, self).fit(10, batch_size, validating=validating, generating=generating)

		# print("Fitting lower conv layers III")
		# self.compile(learn_rate=0.00001)
		# return super(PretrainedModel, self).fit(5, batch_size, validating=validating, generating=generating)
