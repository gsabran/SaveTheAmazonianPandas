import numpy as np
import os

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import keras as k
from keras.models import Sequential, Model as KerasModel
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.merge import *
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from layers.custom_layers import *
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras import backend

from .model import Model
from constants import NUM_WEATHER


class MomoWeatherNet(Model):
    def __init__(self, data, model=None, n_gpus=-1,trainable=True):
        super(MomoWeatherNet, self).__init__(data, model=model, n_gpus=n_gpus)
        self.trainable=trainable

    def create_base_model(self):
        inp = Input(shape=self.input_shape)

        x = BatchNormalization(input_shape=self.input_shape)(inp)

        # add conv layers
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x= residual_convolution(x,(3,3),32,trainable=self.trainable)
        x= residual_convolution(x,(3,3),32,trainable=self.trainable)
        x= residual_convolution(x,(3,3),32,trainable=self.trainable)
        x = MaxPooling2D(pool_size=2)(x)
        x=concatenate([x,x])
        
        x= residual_convolution(x,(3,3),64,trainable=self.trainable)
        x= residual_convolution(x,(3,3),64,trainable=self.trainable)
        x= residual_convolution(x,(3,3),64,trainable=self.trainable)
        x= residual_convolution(x,(3,3),64,trainable=self.trainable)
        x = MaxPooling2D(pool_size=2)(x)
        x=concatenate([x,x])

        x= residual_convolution(x,(3,3),128,trainable=self.trainable)
        x= residual_convolution(x,(3,3),128,trainable=self.trainable)
        x= residual_convolution(x,(3,3),128,trainable=self.trainable)
        x= residual_convolution(x,(3,3),128,trainable=self.trainable)
        x = MaxPooling2D(pool_size=2)(x)
        x=concatenate([x,x])


        x = x= residual_convolution(x,(3,3),256,trainable=self.trainable)
        x = x= residual_convolution(x,(3,3),256,trainable=self.trainable)
        x = x= residual_convolution(x,(3,3),256,trainable=self.trainable)
        x = MaxPooling2D(pool_size=2)(x)

        # add dense layers  
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x=Dense(256,activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(len(self.data.labels), activation='softmax')(x)
        model = KerasModel(inputs=inp, outputs=x)
        self.model = model

    def compile(self, learn_rate=0.001):
        opt = Adam(lr=learn_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, n_epoch, batch_size, validating=True, generating=False, learn_rate=0.001):
        # for layer in self.weather_model.layers:
        #     layer.trainable = False
        self.compile(learn_rate)
        super(MomoWeatherNet, self).fit(n_epoch, batch_size, validating=validating, generating=generating)
