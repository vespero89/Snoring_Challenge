# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import numpy as np;
from sklearn.svm import SVC;
import sklearn.preprocessing as preprocessing;
from sklearn.externals import joblib;
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
import os;
import sys;
import scipy.io

import utils.dataset_manupulation as dm
import utils.utils as utl

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = 'PNCC'
filetype = 'htk'

#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

ubmsPath = os.path.join(targePath, featureset, "ubms")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score_hier")
snoreClassPath =os.path.join(targePath, featureset, "score_hier","final_score.csv")

sys.stdout = open(os.path.join(scoresPath,'test.txt'), 'w')   #log to a file 
print "TEST: "+featureset; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'test_err.txt'), 'w')   #log to a file

nMixtures_bin = joblib.load(os.path.join(scoresPath,'nmix_bin'));
Cs_bin =joblib.load(os.path.join(scoresPath,'cBestValues_bin')); # Best
gammas_bin =joblib.load(os.path.join(scoresPath,'gBestValues_bin')); # Best

nMixtures_class = joblib.load(os.path.join(scoresPath,'nmix_class'));
Cs_class =joblib.load(os.path.join(scoresPath,'cBestValues_class')); # Best
gammas_class =joblib.load(os.path.join(scoresPath,'gBestValues_class')); # Best

nFolds = 1;

scores = np.zeros((nFolds,1));


def compute_score(predictions, labels):
    #print("compute_score")
    y_pred = []
    for d in predictions:
        y_pred.append(int(d))

    y_true = []
    for n in labels:
        y_true.append(int(n))

    A = accuracy_score(y_true, y_pred)
    UAR = recall_score(y_true, y_pred, average='macro')
    CM = confusion_matrix(y_true, y_pred)

    return A, UAR, CM, y_pred


for fold in range(0, nFolds):
        print "Binary Predictions"
        print("Fold: " + str(fold));
        C_bin = Cs_bin[fold-1];
        gamma_bin = gammas_bin[fold-1];

        curSupervecPath_bin = os.path.join(supervecPath, "trainset_" + str(fold), str(nMixtures_bin));
        #TODO LOAD FEATURES
        trainFeatures, trainClassLabels =
        testFeatures =
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1));
        scaler.fit(trainFeatures_bin);
        svm = SVC(C=C_bin, kernel='rbf', gamma=gamma_bin);
        svm.fit(scaler.transform(trainFeatures), trainClassLabels);
        predLabels = svm.predict(scaler.transform(testFeatures));

        print "Multiclass Predictions"
        #TODO FUNZIONE CHE RIACCORPA predLabels ed etichetta nome, quindi ricarico solo quelle delle classi B-C-D
        C_class = Cs_class[fold - 1];
        gamma_class = gammas_class[fold - 1];

        curSupervecPath_bin = os.path.join(supervecPath, "trainset_" + str(fold), str(nMixtures_class));
        # TODO LOAD FEATURES
        trainFeatures, trainClassLabels =
        testFeatures_bin =
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1));
        scaler.fit(trainFeatures);
        svm = SVC(C=C_class, kernel='rbf', gamma=gamma_class);
        svm.fit(scaler.transform(trainFeatures_bin), trainClassLabels);

        predLabels_class = svm.predict(scaler.transform(predLabels))

        print "GLOBAL Predictions - SCORES"
        output_global = []
        i = 0
        for c in predLabels:
            if c == 0:
                output_global.append(0)
            else:
                output_global.append(predLabels_class[i])
                i += 1
        output_global = np.asarray(output_global)
        out_net_filename = os.path.join(scoresPath, 'preds_global.csv')
        np.savetxt(out_net_filename, output_global, delimiter=';')

        A, UAR, ConfMatrix, class_pred = compute_score(output_global, y_devel) #TODO LOAD TOTAL DEVSET LABELS

        sys.stdout = open(snoreClassPath, 'a')
        print "Featureset = " + featureset
        print("**** Results GLOBAL****")

        print("Accuracy (Global): " + str(A))
        print("UAR (Global): " + str(UAR))



