import numpy as np
import os

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import keras as k
from keras.models import Sequential, Model as KerasModel
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.merge import Concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras import backend

from .model import Model
from constants import NUM_WEATHER
from layers.weighted_average import *
class GuiNet(Model):
    def __init__(self, weather_model, *args, **kwargs):
        self.weather_model = weather_model
        super(GuiNet, self).__init__(data, *args, **kwargs)

    def _create_submodel(self, inp, i):
        # add conv layers
        x = BatchNormalization(input_shape=self.input_shape, name="batch_1_{i}".format(i=i))(inp)

        x = Conv2D(32, (3, 3), padding='same', activation='relu', name="conv_1.1_{i}".format(i=i))(x)
        x = Conv2D(32, (3, 3), activation='relu', name="conv_1.2_{i}".format(i=i))(x)
        x = MaxPooling2D(pool_size=2, name="pool_1_{i}".format(i=i))(x)
        x = Dropout(0.25, name="drop_1_{i}".format(i=i))(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu', name="conv_2.1_{i}".format(i=i))(x)
        x = Conv2D(64, (3, 3), activation='relu', name="conv_2.2_{i}".format(i=i))(x)
        x = MaxPooling2D(pool_size=2, name="pool_2_{i}".format(i=i))(x)
        x = Dropout(0.25, name="drop_2_{i}".format(i=i))(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu', name="conv_3.1_{i}".format(i=i))(x)
        x = Conv2D(128, (3, 3), activation='relu', name="conv_3.2_{i}".format(i=i))(x)
        x = MaxPooling2D(pool_size=2, name="pool_3_{i}".format(i=i))(x)
        x = Dropout(0.25, name="drop_3_{i}".format(i=i))(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu', name="conv_4.1_{i}".format(i=i))(x)
        x = Conv2D(256, (3, 3), activation='relu', name="conv_4.2_{i}".format(i=i))(x)
        x = MaxPooling2D(pool_size=2, name="pool_4_{i}".format(i=i))(x)
        x = Dropout(0.25, name="drop_4_{i}".format(i=i))(x)

        # add dense layers
        x = Flatten(name="flatten_{i}".format(i=i))(x)
        x = Dense(512, activation='relu', name="dense_1_{i}".format(i=i))(x)
        x = BatchNormalization(name="batch_2_{i}".format(i=i))(x)
        x = Dropout(0.5, name="drop_5_{i}".format(i=i))(x)
        return Dense(len(self.data.labels), activation='sigmoid', name="dense_2_{i}".format(i=i))(x)

    def create_base_model(self):
        input1 = Input(shape=self.input_shape)

        submodels = [self._create_submodel(input1, i) for i in range(NUM_WEATHER)]

        print("submodels", submodels[0].shape, self.weather_model.output)
        x = Concatenate()(submodels)
        x = weighted_average(inputs=submodels, weights=self.weather_model(input1))
        model = KerasModel(inputs=input1, outputs=x)

        self.model = model

    def compile(self, learn_rate=0.001):
        opt = Adam(lr=learn_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, n_epoch, batch_size, validating=True, augmenting=False, learn_rate=0.001):
        # for layer in self.weather_model.layers:
        #     layer.trainable = False
        self.compile(learn_rate)
        super(GuiNet, self).fit(n_epoch, batch_size, validating=validating, augmenting=augmenting)
