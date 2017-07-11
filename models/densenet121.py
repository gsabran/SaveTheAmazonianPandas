import keras
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from layers.scale import Scale

from .pretrained_model import PretrainedModel

class DenseNet121(PretrainedModel):
	"""
	An adaptation of the DenseNet121 model
	"""

	def _load_pretrained_model(self):
		return self.DenseNet(include_top=False, input_shape=self.input_shape)

	def _add_top_dense_layers(self, x):
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.25)(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation="relu")(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation="relu")(x)
		x = Dropout(0.5)(x)
		x = Dense(128, activation="relu")(x)
		x = Dropout(0.25)(x)
		# and a logistic layer
		return Dense(len(self.data.labels), activation="sigmoid")(x)

	def DenseNet(self, input_shape, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0,
							 dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None,
							 include_top=True):
		"""Instantiate the DenseNet 121 architecture,
				# Arguments
						nb_dense_block: number of dense blocks to add to end
						growth_rate: number of filters to add per dense block
						nb_filter: initial number of filters
						reduction: reduction factor of transition blocks.
						dropout_rate: dropout rate
						weight_decay: weight decay factor
						classes: optional number of classes to classify images
						weights_path: path to pre-trained weights
				# Returns
						A Keras model instance.
		"""
		eps = 1.1e-5

		# compute compression factor
		compression = 1.0 - reduction

		# Handle Dimension Ordering for different backends
		if K.image_dim_ordering() == "tf":
			self.concat_axis = 3
		else:
			self.concat_axis = 1
		img_input = Input(shape=input_shape, name="data")

		# From architecture for ImageNet (Table 1 in the paper)
		nb_filter = 64
		nb_layers = [6,12,24,16] # For DenseNet-121

		# Initial convolution
		x = ZeroPadding2D((3, 3), name="conv1_zeropadding")(img_input)
		x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name="conv1", use_bias=False)(x)
		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name="conv1_bn")(x)
		x = Scale(axis=self.concat_axis, name="conv1_scale")(x)
		x = Activation("relu", name="relu1")(x)
		x = ZeroPadding2D((1, 1), name="pool1_zeropadding")(x)
		x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

		# Add dense blocks
		for block_idx in range(nb_dense_block - 1):
			stage = block_idx+2
			x, nb_filter = self.__dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

			# Add __transition_block
			x = self.__transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
			nb_filter = int(nb_filter * compression)

		final_stage = stage + 1
		x, nb_filter = self.__dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name="conv"+str(final_stage)+"_blk_bn")(x)
		x = Scale(axis=self.concat_axis, name="conv"+str(final_stage)+"_blk_scale")(x)
		x = Activation("relu", name="relu"+str(final_stage)+"_blk")(x)
		if include_top:
			x = GlobalAveragePooling2D(name="pool"+str(final_stage))(x)

			x = Dense(classes, name="fc6")(x)
			x = Activation("softmax", name="prob")(x)

		model = keras.models.Model(img_input, x, name="densenet")

		if weights_path is not None:
			model.load_weights(weights_path, by_name=True)

		return model

	def __conv_block(self, x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
		"""Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
				# Arguments
						x: input tensor 
						stage: index for dense block
						branch: layer index within each dense block
						nb_filter: number of filters
						dropout_rate: dropout rate
						weight_decay: weight decay factor
		"""
		eps = 1.1e-5
		conv_name_base = "conv" + str(stage) + "_" + str(branch)
		relu_name_base = "relu" + str(stage) + "_" + str(branch)

		# 1x1 Convolution (Bottleneck layer)
		inter_channel = nb_filter * 4  
		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+"_x1_bn")(x)
		x = Scale(axis=self.concat_axis, name=conv_name_base+"_x1_scale")(x)
		x = Activation("relu", name=relu_name_base+"_x1")(x)
		x = Conv2D(inter_channel, (1, 1), name=conv_name_base+"_x1", use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		# 3x3 Convolution
		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+"_x2_bn")(x)
		x = Scale(axis=self.concat_axis, name=conv_name_base+"_x2_scale")(x)
		x = Activation("relu", name=relu_name_base+"_x2")(x)
		x = ZeroPadding2D((1, 1), name=conv_name_base+"_x2_zeropadding")(x)
		x = Conv2D(nb_filter, (3, 3), name=conv_name_base+"_x2", use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		return x


	def __transition_block(self, x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
		""" Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
				# Arguments
						x: input tensor
						stage: index for dense block
						nb_filter: number of filters
						compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
						dropout_rate: dropout rate
						weight_decay: weight decay factor
		"""

		eps = 1.1e-5
		conv_name_base = "conv" + str(stage) + "_blk"
		relu_name_base = "relu" + str(stage) + "_blk"
		pool_name_base = "pool" + str(stage) 

		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+"_bn")(x)
		x = Scale(axis=self.concat_axis, name=conv_name_base+"_scale")(x)
		x = Activation("relu", name=relu_name_base)(x)
		x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

		return x

	def __dense_block(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
		""" Build a __dense_block where the output of each __conv_block is fed to subsequent ones
				# Arguments
						x: input tensor
						stage: index for dense block
						nb_layers: the number of layers of __conv_block to append to the model.
						nb_filter: number of filters
						growth_rate: growth rate
						dropout_rate: dropout rate
						weight_decay: weight decay factor
						grow_nb_filters: flag to decide to allow number of filters to grow
		"""

		eps = 1.1e-5
		concat_feat = x

		for i in range(nb_layers):
			branch = i+1
			x = self.__conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
			concat_feat = Concatenate(axis=self.concat_axis, name="concat_"+str(stage)+"_"+str(branch))([concat_feat, x])

			if grow_nb_filters:
				nb_filter += growth_rate

		return concat_feat, nb_filter
