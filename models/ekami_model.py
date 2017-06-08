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
        self.model = Sequential()
        self.add_conv_layer(self.img_size)
        self.add_flatten_layer()
        self.add_ann_layer(NUM_WEATHER + NUM_TAGS)

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.model.add(BatchNormalization(input_shape=(self.img_size[0], self.img_size[1], img_channels)))

        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))


    def add_flatten_layer(self):
        self.model.add(Flatten())


    def add_ann_layer(self, output_size):
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_size, activation='sigmoid'))

    def compile(self, learn_rate=0.001):
        opt = Adam(lr=learn_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def _get_fbeta_score(self, model, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def fit(self, n_epoch, batch_size, validating=True, generating=True, learn_rate=0.001):
        self.compile(learn_rate)
        super(AmazonKerasClassifier, self).fit(20, batch_size, validating=validating, generating=generating)
        self.compile(learn_rate * 0.1)
        super(AmazonKerasClassifier, self).fit(5, batch_size, validating=validating, generating=generating)
        self.compile(learn_rate * 0.01)
        super(AmazonKerasClassifier, self).fit(5, batch_size, validating=validating, generating=generating)
