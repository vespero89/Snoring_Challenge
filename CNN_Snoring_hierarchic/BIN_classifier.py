#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: andrea
"""

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.layers import Input, Dense, Reshape, Flatten
from keras.optimizers import Adam, Adadelta, SGD, RMSprop
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, History
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
import math


class Bin_Classifier():
    def __init__(self, input_shape, fit=True):
        print("__init__ (Bin Classifier)")
        self._fit_net = fit  # se è False carica il modello e i pesi dal disco.

        self._config = 0
        self._weight = 0
        self._classifier = 0
        self._input_shape = input_shape

    def define_parametric_arch(self, params):
        print("define_parametric arch (Bin Classifier)")

        d = self._input_shape[0]
        h = self._input_shape[1]
        w = self._input_shape[2]
        print("(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        input_img = Input(shape=self._input_shape)
        x = input_img

        x = Flatten()(x)
        x = Dense(1, activation=params.bin_activation)(x)

        self._classifier = Model(input_img, x)
        self._classifier.summary()

        return self._classifier

    def model_compile(self, p_optimizer, p_loss, model=None):
        '''
        compila il modello con i parametri passati: se non viene passato compila il modello istanziato dalla classe
        '''
        print("model_compile (Bin Classifier)")

        if model == None:
            self._classifier.compile(optimizer=p_optimizer, loss=p_loss)
        else:
            model.compile(optimizer=p_optimizer, loss=p_loss)

    def model_fit(self, x_train, y_train, path, params, x_test=None, y_test=None):
        print("model_fit (Bin Classifier)")
        model_name = os.path.join(path, 'best_bin.h5')
        logfile = os.path.join(path, 'training_bin.log')
        if not self._fit_net:
            # if i want to load from disk the model
            classifier = load_model(model_name)
            self._classifier = classifier
        else:
            checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
            es = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')
            csv_logger = CSVLogger(logfile)
            hist = History()
            if x_test != None and y_test != None:
                self._classifier.fit(x_train, y_train,
                                     nb_epoch=params.epoch_bin,
                                     batch_size=params.batch_size_bin,
                                     callbacks=[checkpointer, es, csv_logger, hist],
                                     validation_data=(x_test, y_test),
                                     shuffle=params.shuffle_bin,
                                     verbose=2
                                     )
            else:
                self._classifier.fit(x_train, y_train,
                                     nb_epoch=params.epoch_bin,
                                     batch_size=params.batch_size_bin,
                                     validation_split=params.valid_split_bin,
                                     callbacks=[checkpointer, es, csv_logger, hist],
                                     shuffle=params.shuffle_bin,
                                     verbose=2
                                     )

            self._config = self._classifier.get_config()
            self._weight = self._classifier.get_weights()

        self._fit_net = False
        return hist, self._classifier

    def class_predictions(self, x_test, model_path):

        print("Writing Predictions")
        model_name = os.path.join(model_path, 'best_bin.h5')
        classifier = load_model(model_name)
        class_prob = classifier.predict(x_test)
        return class_prob

    def compute_score(self, predictions, labels, thr):
        print("compute_score")

        y_pred = []
        for d in predictions:
            if d <= thr:
                y_pred.append(0)
            else:
                y_pred.append(1)

        y_true = []
        for n in labels:
            y_true.append(int(n))

        A = accuracy_score(y_true, y_pred)
        UAR = recall_score(y_true, y_pred, average='macro')
        CM = confusion_matrix(y_true, y_pred)

        cm = CM.astype(int)
        print("FINAL REPORT")
        print("\t\t V\t OTE")
        print(" V  \t" + str(cm[0, 0]) + "\t  " + str(cm[0, 1]))
        print("OTE \t" + str(cm[1, 0]) + "\t  " + str(cm[1, 1]))

        print("\n" + classification_report(y_true, y_pred, target_names=['V', 'OTE']))

        return A, UAR, CM, y_pred
