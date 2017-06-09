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

from .model import Model

class AmazonKerasClassifier(Model):
    def create_base_model(self):
        self.model = Sequential()
        self.add_conv_layer(self.img_size)
        self.add_flatten_layer()
        self.add_ann_layer(len(self.data.labels))

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
        self.model.add(Dense(output_size, activation='softmax'))

    def compile(self, learn_rate=0.001):
        opt = Adam(lr=learn_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def _get_fbeta_score(self, model, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def fit(self, n_epoch, batch_size, validating=True, generating=False, learn_rate=0.001):
        self.compile(learn_rate)
        super(AmazonKerasClassifier, self).fit(20, batch_size, validating=validating, generating=generating)
        self.compile(learn_rate * 0.1)
        super(AmazonKerasClassifier, self).fit(5, batch_size, validating=validating, generating=generating)
        self.compile(learn_rate * 0.1)
        super(AmazonKerasClassifier, self).fit(5, batch_size, validating=validating, generating=generating)
'''def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
                history = LossHistory()

                X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                                      test_size=validation_split_size)



                self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


                # early stopping will auto-stop training process if model stops learning after 3 epochs
                earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

                self.classifier.fit(X_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epoch,
                                    verbose=1,
                                    validation_data=(X_valid, y_valid),
                                    callbacks=[history, *train_callbacks, earlyStopping])
                fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
                return [history.train_losses, history.val_losses, fbeta_score]

            def save_weights(self, weight_file_path):
                self.classifier.save_weights(weight_file_path)

            def load_weights(self, weight_file_path):
                self.classifier.load_weights(weight_file_path)

            def predict(self, x_test):
                predictions = self.classifier.predict(x_test)
                return predictions

            def map_predictions(self, predictions, labels_map, thresholds):
                """
                Return the predictions mapped to their labels
                :param predictions: the predictions from the predict() method
                :param labels_map: the map
                :param thresholds: The threshold of each class to be considered as existing or not existing
                :return: the predictions list mapped to their labels
                """
                predictions_labels = []
                for prediction in predictions:
                    labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
                    predictions_labels.append(labels)

                return predictions_labels

            def close(self):
                backend.clear_session()'''
