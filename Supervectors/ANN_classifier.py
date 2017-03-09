
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: buckler
"""

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
          
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, History
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report

class Classifier():
    
    def __init__(self, input_shape, fit=True):
        print("__init__")
        self._fit_net = fit
        self._config = 0
        self._weight = 0
        self._classifier = 0
        self._input_shape = input_shape

    def define_parametric_arch(self, params):
        print("define_parametric arch")

        self._classifier = Sequential()
        self._classifier.add(Dense(params.dense_layers_inputs[0],
                                   input_dim=self._input_shape[1],
                                   init=params.init,
                                   activation=params.dense_activation,
                                   W_regularizer=params.w_reg,
                                   b_regularizer=params.b_reg,
                                   activity_regularizer=params.a_reg,
                                   W_constraint=params.w_constr,
                                   b_constraint=params.b_constr,
                                   bias=params.bias))

        for i in range(1, len(params.dense_layers_inputs)-1):
            self._classifier.add(Dense(params.dense_layers_inputs[i],
                                       init=params.init,
                                       activation=params.dense_activation,
                                       W_regularizer=params.w_reg,
                                       b_regularizer=params.b_reg,
                                       activity_regularizer=params.a_reg,
                                       W_constraint=params.w_constr,
                                       b_constraint=params.b_constr,
                                       bias=params.bias))

        self._classifier.add(Dense(4, activation='softmax'))
        #self._classifier.summary()

        return self._classifier




    def model_compile(self, p_optimizer, p_loss, model=None):
        '''
        compila il modello con i parametri passati: se non viene passato compila il modello istanziato dalla classe
        '''
        print("model_compile")

        if model==None:
            self._classifier.compile(optimizer=p_optimizer, loss=p_loss)
        else:
            model.compile(optimizer=p_optimizer, loss=p_loss)
        
    def model_fit(self, x_train, y_train, path, params, x_test=None, y_test=None):
        print("model_fit")
        model_name = os.path.join(path, 'best_cnn.h5')
        logfile = os.path.join(path, 'training.log')
        if not self._fit_net:
            #if i want to load from disk the model
            classifier=load_model(model_name)
            self._classifier=classifier
        else:
            checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=True)
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
                                verbose=0
                                )
            else:
                self._classifier.fit(x_train, y_train,
                        nb_epoch=params.epoch,
                        batch_size=params.batch_size,
                        validation_split=params.valid_split,
                        callbacks=[checkpointer, es, csv_logger, hist],
                        shuffle=params.shuffle,
                        verbose=0
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

        print("Accuracy: " + str(A))
        print("UAR: " + str(UAR))

        # cm = CM.astype(int)
        # print("FINAL REPORT")
        # print("\t V\t O\t T\t E")
        # print("V \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]) + "\t" + str(cm[0, 2]) + "\t" + str(cm[0, 3]))
        # print("O \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]) + "\t" + str(cm[1, 2]) + "\t" + str(cm[1, 3]))
        # print("T \t" + str(cm[2, 0]) + "\t" + str(cm[2, 1]) + "\t" + str(cm[2, 2]) + "\t" + str(cm[2, 3]))
        # print("E \t" + str(cm[3, 0]) + "\t" + str(cm[3, 1]) + "\t" + str(cm[3, 2]) + "\t" + str(cm[3, 3]))

        # print(classification_report(y_true, y_pred, target_names=['V', 'O', 'T', 'E']))

        return A, UAR, CM, y_pred
