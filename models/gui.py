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
from .constants import NUM_WEATHER

class GuiNet(Model):
    def create_base_model(self):
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=(NUM_WEATHER,))

        x = BatchNormalization(input_shape=self.input_shape)(input1)

        model = Sequential()

        # add conv layers
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)


        # add dense layers
        x = Flatten()(x)
        x = Concatenate()([x, input2])
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(len(self.data.labels), activation='sigmoid')(x)
        model = KerasModel(inputs=[input1, input2], outputs=x)

        self.model = model

    def compile(self, learn_rate=0.001):
        opt = Adam(lr=learn_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, n_epoch, batch_size, validating=True, generating=False, learn_rate=0.001):
        self.compile(learn_rate)
        super(GuiNet, self).fit(n_epoch, batch_size, validating=validating, generating=generating)
