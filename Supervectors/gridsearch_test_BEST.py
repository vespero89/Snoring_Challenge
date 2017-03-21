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
import utils.create_arff as arff

import utils.dataset_manupulation as dm
import utils.utils as utl

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = 'SCAT'
filetype = 'htk'

#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

ubmsPath = os.path.join(targePath, featureset, "ubms")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score_best")
snoreClassPath =os.path.join(targePath, featureset, "score_best","final_score_TEST_ER_t.csv")

sys.stdout = open(os.path.join(scoresPath,'test.txt'), 'w')   #log to a file
print "TEST: "+featureset; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'test_err.txt'), 'w')   #log to a file

#LOAD DATASET
snoring_dataset = dm.load_ComParE2017(featPath, filetype) # load dataset
trainset, develset, testset = dm.split_ComParE2017_simple(snoring_dataset)  # creo i trainset per calcolare media e varianza per poter normalizzare
labels = dm.label_loading(os.path.join(root_dir,'lab','ComParE2017_Snore.tsv'))
trainset_l, develset_l, _ = dm.split_ComParE2017_simple(labels)
del snoring_dataset

y = []
for seq in trainset:
    y.append(seq[0])

yd = []
for seq in develset:
    yd.append(seq[0])

yt = []
for seq in testset:
    yt.append(seq[0])

y_train, y_train_lab, _ = dm.label_organize(trainset_l, y)
y_devel, y_devel_lab, y_devel_lit = dm.label_organize(develset_l, yd)

#EXTEND TRAINSET
#y_train_lab = np.append(y_train_lab,y_devel_lab[:140])
#y_devel_lab = y_devel_lab[140:]


nMixtures = joblib.load(os.path.join(scoresPath,'nmix2'));
Cs = joblib.load(os.path.join(scoresPath,'cBestValues2')); # Best
gammas =joblib.load(os.path.join(scoresPath,'gBestValues2')); # Best
Best_model = joblib.load(os.path.join(scoresPath,'best_model')); # Best
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

    cm = CM.astype(int)
    print("FINAL REPORT")
    print("\t V\t O\t T\t E")
    print("V \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]) + "\t" + str(cm[0, 2]) + "\t" + str(cm[0, 3]))
    print("O \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]) + "\t" + str(cm[1, 2]) + "\t" + str(cm[1, 3]))
    print("T \t" + str(cm[2, 0]) + "\t" + str(cm[2, 1]) + "\t" + str(cm[2, 2]) + "\t" + str(cm[2, 3]))
    print("E \t" + str(cm[3, 0]) + "\t" + str(cm[3, 1]) + "\t" + str(cm[3, 2]) + "\t" + str(cm[3, 3]))

    print(classification_report(y_true, y_pred, target_names=['V', 'O', 'T', 'E']))
    recall_report = recall_score(y_true, y_pred, labels=['0', '1', '2', '3'], average=None)

    return A, UAR, CM, y_pred, recall_report


for fold in range(0, nFolds):
        print("Fold: " + str(fold));
        C = Cs[fold];
        gamma = gammas[fold];
        BM = Best_model

        print "Final Test SVM Classifier"
        curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold), str(nMixtures));
        #TODO LOAD FEATURES
        trainFeatures = utl.readfeatures(curSupervecPath, y)
        devFeatures = utl.readfeatures(curSupervecPath, yd)
        testFeatures = utl.readfeatures(curSupervecPath, yt)
        trainClassLabels = y_train_lab

        #EXTEND TRAINSET
        #trainFeatures = np.vstack((trainFeatures,devFeatures[:140,:]))
        #devFeatures = devFeatures[140:]

        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1));
        scaler.fit(trainFeatures);
        svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto', probability=True)
        svm.fit(scaler.transform(trainFeatures), trainClassLabels);
        predLabels_train = svm.predict(scaler.transform(trainFeatures));
        predLabels_dev = svm.predict(scaler.transform(devFeatures));

        sys.stdout = open(snoreClassPath, 'w')
        print featureset
        print("**** Results SVM****")
        print("train set")
        A, UAR, ConfMatrix, class_pred, recall_report = compute_score(predLabels_train, y_train_lab)  # TODO LOAD TOTAL DEVSET LABELS
        print("Accuracy: " + str(A))
        print("UAR: " + str(UAR))

        print("devel set")
        print("N GAUSS:" + str(nMixtures) + " C:" + str(C) + " gamma:" + str(gamma))
        A, UAR, ConfMatrix, class_pred, recall_report = compute_score(predLabels_dev,y_devel_lab)  # TODO LOAD TOTAL DEVSET LABELS
        print("Accuracy: " + str(A))
        print("UAR: " + str(UAR))

        predLabels_test = svm.predict(scaler.transform(testFeatures));
        for p in range(0, len(predLabels_test)):
            predClass = [0, 0, 0, 0]
            index = int(predLabels_test[p])
            predClass[index] = 1
            if p == 0:
                predProb = predClass
            else:
                predProb = np.vstack((predProb, predClass))

        class_pred = []
        for d in predLabels_test:
            class_pred.append(int(d))

        V_tot = np.sum(predProb[:,0])
        O_tot = np.sum(predProb[:, 1])
        T_tot = np.sum(predProb[:, 2])
        E_tot = np.sum(predProb[:, 3])

        print("N Predictions VOTE;" + str(V_tot) + ";" + str(O_tot) + ";" + str(T_tot) + ";" + str(E_tot))

        # # CREATE ARFF FILES
        arff.create_arff_test(scoresPath, yt, predProb, class_pred)
        # arff.create_pred(scoresPath, y_devel_lab, y_devel_lit, predProb, class_pred)
        # arff.create_result(scoresPath, A, UAR, ConfMatrix, recall_report)

