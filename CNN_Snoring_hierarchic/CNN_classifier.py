#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: buckler
"""

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
          
from keras.layers import Input, Dense, Flatten, BatchNormalization, Convolution2D, MaxPooling2D
from keras.optimizers import Adam, Adadelta, SGD
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, History
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
import math



class Cnn_Classifier():
    
    def __init__(self, kernel_shape, number_of_kernel, input_shape, fit=True):
        print("__init__")
        self._fit_net = fit #se è False carica il modello e i pesi dal disco.

        self._ks = kernel_shape
        self._nk = number_of_kernel        
        self._config = 0
        self._weight = 0
        self._classifier = 0
        self._input_shape = input_shape
    
    def define_arch(self):
        print("define_arch")
       #################################################################
       # ToDo: architettura dinamica in base alla matrice kernel_shape #
       #################################################################

        input_img = Input(shape=self._input_shape)

        x = Convolution2D(self._nk[0], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(self._nk[1], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(self._nk[2], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        # at this point the representation is (8, 4, 4) i.e. 128-dimensional
        
        x = Flatten()(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(256, activation='tanh')(x)

        predictions = Dense(3, activation='softmax')(x)

        self._classifier = Model(input_img, predictions)

        return self._classifier

    def define_parametric_arch(self, params):
        print("define_parametric arch")
        # ---------------------------------------------------------- Convolutional Section
        d = self._input_shape[0]
        h = self._input_shape[1]
        w = self._input_shape[2]
        print("(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        input_img = Input(shape=self._input_shape)
        x = input_img

        for i in range(len(params.kernel_number)):
            x = Convolution2D(params.kernel_number[i],
                              params.kernel_shape[i][0],
                              params.kernel_shape[i][1],
                              init=params.cnn_init,
                              activation=params.cnn_conv_activation,
                              border_mode=params.border_mode,
                              subsample=tuple(params.strides),
                              W_regularizer=params.w_reg,
                              b_regularizer=params.b_reg,
                              activity_regularizer=params.a_reg,
                              W_constraint=params.w_constr,
                              b_constraint=params.b_constr,
                              bias=params.bias)(x)


            if params.border_mode == 'same':
                ph = params.kernel_shape[i][0] - 1
                pw = params.kernel_shape[i][1] - 1
            else:
                ph = pw = 0
            h = int((h - params.kernel_shape[i][0] + ph) / params.strides[0]) + 1
            w = int((w - params.kernel_shape[i][1] + pw) / params.strides[1]) + 1
            d = params.kernel_number[i]
            print("conv " + str(i) + "->(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

            if not params.pool_only_to_end:
                x = MaxPooling2D(params.m_pool[i], border_mode='same')(x)
                # if border=='valid' h=int(h/params.params.m_pool[i][0])
                h = math.ceil(h / params.m_pool[i][0])
                w = math.ceil(w / params.m_pool[i][1])
                print("pool " + str(i) + "->(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        if params.pool_only_to_end:
            x = MaxPooling2D(params.m_pool[0], border_mode='same')(x)
            # if border=='valid' h=int(h/params.params.m_pool[i][0])
            h = math.ceil(h / params.m_pool[i][0])
            w = math.ceil(w / params.m_pool[i][1])
            print("pool->  (" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        x = Flatten()(x)

        for i in range(len(params.dense_layers_inputs)):
            x = Dense(params.dense_layers_inputs[i],
                      init=params.cnn_init,
                      activation=params.cnn_dense_activation,
                      W_regularizer=params.w_reg,
                      b_regularizer=params.b_reg,
                      activity_regularizer=params.a_reg,
                      W_constraint=params.w_constr,
                      b_constraint=params.b_constr,
                      bias=params.bias)(x)

        predictions = Dense(3, activation='softmax')(x)

        self._classifier = Model(input_img, predictions)
        self._classifier.summary()

        return self._classifier

    def model_compile(self, p_optimizer, learn_rate, p_loss, model=None):
        '''
        compila il modello con i parametri passati: se non viene passato compila il modello istanziato dalla classe
        '''
        print("model_compile")

        if p_optimizer == "adam":
            lr_default =0.001
            opt = Adam(lr=lr_default, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            if learn_rate != lr_default:
                opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif p_optimizer == "adadelta":
            lr_default=1.0
            opt = Adadelta(lr=lr_default, rho=0.95, epsilon=1e-08, decay=0.0)
            if learn_rate != lr_default:
                opt = Adadelta(lr=learn_rate, rho=0.95, epsilon=1e-08, decay=0.0)
        elif p_optimizer == "sgd":
            lr_default=0.01
            opt = SGD(lr=lr_default, momentum=0.0, decay=0.0, nesterov=False)
            if learn_rate != lr_default:
                opt = SGD(lr=lr_rate, momentum=0.0, decay=0.0, nesterov=False)

        if model==None:
            #passare "opt" se si vogliono differenti learning rate
            self._classifier.compile(optimizer=p_optimizer, loss=p_loss)
        else:
            # passare "opt" se si vogliono differenti learning rate
            model.compile(optimizer=p_optimizer, loss=p_loss)
        
    def model_fit(self, x_train, y_train, path, params, x_test=None, y_test=None):
        print("model_fit")
        model_name = os.path.join(path, 'best_cnn.h5')
        logfile = os.path.join(path, 'training_cnn.log')
        if not self._fit_net:
            #if i want to load from disk the model
            classifier=load_model(model_name)
            self._classifier=classifier
        else:
            checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
            es=EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')
            csv_logger = CSVLogger(logfile)
            hist = History()
            if x_test != None and y_test != None:
                self._classifier.fit(x_train, y_train,
                                nb_epoch=params.epoch,
                                batch_size=params.batch_size,
                                callbacks=[checkpointer, es, csv_logger, hist],
                                validation_data=(x_test, y_test),
                                shuffle=params.shuffle,
                                verbose=2
                                )
            else:
                self._classifier.fit(x_train, y_train,
                        nb_epoch=params.epoch,
                        batch_size=params.batch_size,
                        validation_split=params.valid_split,
                        callbacks=[checkpointer, es, csv_logger, hist],
                        shuffle=params.shuffle,
                        verbose=2
                        )
            #save the model an weights on disk
            #self.save_model(self._classifier,'my_model2.h5')
            #self._classifier.save('my_model2.h5')
            #self._classifier.save_weights('my_model_weights2.h5')
            #save the model and wetight on varibles
            self._config = self._classifier.get_config()
            self._weight = self._classifier.get_weights()

        self._fit_net=False
        return hist, self._classifier
    
    # def save_model(model,name):
    #         '''
    #         salva il modello e i pesi.
    #         TODO gestire nomi dei file in maniera intelligente in base ai parametri e case, in
    #         modo tale che siano riconoscibili alla fine
    #         '''
    #         model.save(name)
    #         model.save_weights(name)
    #
    def class_predictions(self,x_test, model_path):
        '''
        decodifica i vettori in ingresso.
        '''
        print("Writing Predictions")
        model_name = os.path.join(model_path, 'best_cnn.h5')
        classifier = load_model(model_name)
        class_prob = classifier.predict(x_test)
        return class_prob

    def compute_score(self, predictions, labels):
        print("compute_score")

        y_pred = []
        for d in predictions:
            y_pred.append(np.argmax(d))

        y_true = []
        for n in labels:
            y_true.append(int(n))

        A=accuracy_score(y_true, y_pred)
        UAR=recall_score(y_true, y_pred, average='macro')
        CM=confusion_matrix(y_true, y_pred)

        cm = CM.astype(int)
        print("FINAL REPORT")
        print("\t O\t T\t E")
        print("O  \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]) + "\t" + str(cm[0, 2]))
        print("T  \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]) + "\t" + str(cm[1, 2]))
        print("E  \t" + str(cm[2, 0]) + "\t" + str(cm[2, 1]) + "\t" + str(cm[2, 2]))

        print("\n" + classification_report(y_true, y_pred, target_names=['O', 'T', 'E']))

        return A, UAR, CM, y_pred

    def compute_score_global(self, predictions, labels):
        print("compute_score_global")

        y_pred = []
        for d in predictions:
            y_pred.append(np.argmax(d))

        y_true = []
        for n in labels:
            y_true.append(int(n))

        A = accuracy_score(y_true, y_pred)
        UAR = recall_score(y_true, y_pred, average='macro')
        CM = confusion_matrix(y_true, y_pred)

        cm = CM.astype(int)
        print("FINAL REPORT")
        print("\t V\t O\t T\t E")
        print("V \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]) + "\t" + str(cm[0, 2]) + "\t" + str(cm[0, 3]))
        print("O \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]) + "\t" + str(cm[1, 2]) + "\t" + str(cm[1, 3]))
        print("T \t" + str(cm[2, 0]) + "\t" + str(cm[2, 1]) + "\t" + str(cm[2, 2]) + "\t" + str(cm[2, 3]))
        print("E \t" + str(cm[3, 0]) + "\t" + str(cm[3, 1]) + "\t" + str(cm[3, 2]) + "\t" + str(cm[3, 3]))

        print("\n" + classification_report(y_true, y_pred, target_names=['V', 'O', 'T', 'E']))

        return A, UAR, CM, y_pred

