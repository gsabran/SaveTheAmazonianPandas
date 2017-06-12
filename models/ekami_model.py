import numpy as np
import os

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import keras as k
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras import backend

from constants import NUM_TAGS, NUM_WEATHER
from .model import Model

class AmazonKerasClassifier(Model):
    def create_base_model(self):
        model = Sequential()

        model.add(BatchNormalization(input_shape=self.input_shape))

        # add conv layers
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        # add dense layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(NUM_WEATHER + NUM_TAGS, activation='sigmoid'))
        self.model = model

    def compile(self, learn_rate=0.001):
        opt = Adam(lr=learn_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, n_epoch, batch_size, validating=True, generating=True, learn_rate=0.001):
        self.compile(learn_rate)
        super(AmazonKerasClassifier, self).fit(20, batch_size, validating=validating, generating=generating)
        self.compile(learn_rate * 0.1)
        super(AmazonKerasClassifier, self).fit(5, batch_size, validating=validating, generating=generating)
        self.compile(learn_rate * 0.01)
        super(AmazonKerasClassifier, self).fit(5, batch_size, validating=validating, generating=generating)
