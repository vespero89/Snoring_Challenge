import sys
sys.path.append('..')
import numpy as np
from sklearn.svm import SVC
import sklearn.preprocessing as preprocessing
from sklearn.externals import joblib
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
import os
import sys
import utils.dataset_manupulation as dm
import utils.utils as utl
import utils.create_arff as arff

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = 'SCAT'
filetype = 'htk'

#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

#ubmsPath = os.path.join(targePath, featureset, "ubms_ext")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score")
snoreClassPath =os.path.join(targePath, featureset, "score","final_score_ET.csv")#used for save best c-best gamma-best nmix so that extract_supervector_test.py and test.py can read it

sys.stdout = open(os.path.join(scoresPath,'gridsearch_ET.txt'), 'w')   #log to a file
print "experiment: "+targePath; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'gridsearch_err_ET.txt'), 'w')   #log to a file

#variables inizialization
nFolds = 1;
C_range = 2.0 ** np.arange(-5, 15+2, 2)    # libsvm range
gamma_range = 2.0 ** np.arange(-15, 3+2, 2) # libsvm range
mixtures = 2**np.arange(0, 7, 1)

# C_range = 2.0 ** np.arange(5, 6, 1)    # libsvm range
# gamma_range = 2.0 ** np.arange(-12, -11, 1) # libsvm range
# mixtures = 2**np.arange(6, 7, 1)

scores      = np.zeros((mixtures.shape[0], nFolds))
cBestValues = np.zeros((mixtures.shape[0], nFolds))
gBestValues = np.zeros((mixtures.shape[0], nFolds))
mIdx = 0;

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

y_train, y_train_lab, _ = dm.label_organize(trainset_l, y)
y_devel, y_devel_lab, y_devel_lit = dm.label_organize(develset_l, yd)

# TRAIN INVERSO
# y = []
# for seq in develset:
#     y.append(seq[0])
#
# yd = []
# for seq in trainset:
#     yd.append(seq[0])
# y_train, y_train_lab, _ = dm.label_organize(develset_l, y)
# y_devel, y_devel_lab, y_devel_lit = dm.label_organize(trainset_l, yd)

# ##EXTEND TRAINSET
# y_train_lab = np.append(y_train_lab,y_devel_lab[:140])
# y_devel_lab = y_devel_lab[140:]

def compute_score(predictions, labels):
    print("compute_score")

    y_pred = []
    for d in predictions:
        y_pred.append(int(d))

    y_true = []
    for n in labels:
        y_true.append(int(n))

    A = accuracy_score(y_true, y_pred)
    UAR = recall_score(y_true, y_pred, average='macro')
    CM = confusion_matrix(y_true, y_pred)

    print("Accuracy: " + str(A))
    print("UAR: " + str(UAR))

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

for m in mixtures:
    print("Mixture: " + str(m))
    sys.stdout.flush()
    mixScores = np.zeros((nFolds*(nFolds-1), 1))
    fIdx = 0;
    for fold in range(0,nFolds):
        cGammaScores = np.zeros((C_range.shape[0], gamma_range.shape[0])) #inizializza matrice dei punteggi
        print("Fold: " + str(fold))
        sys.stdout.flush()
        curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold))
        for sf in range(0,nFolds):
            print("Subfold: " + str(sf))
            sys.stdout.flush()
            curSupervecSubPath = os.path.join(curSupervecPath, str(m))
            trainFeatures = utl.readfeatures(curSupervecSubPath, y)
            trainClassLabels = y_train_lab

            # devFeatures = utl.readfeatures(curSupervecSubPath, yd)
            # devClassLabels = y_devel_lab

            #ERROR TEST
            devFeatures = utl.readfeatures(curSupervecSubPath, y)
            devClassLabels = y_train_lab


            # #EXTEND TRAINSET
            # trainFeatures = np.vstack((trainFeatures,devFeatures[:140,:]))
            # devFeatures = devFeatures[140:]

            cIdx = 0;
            for C in C_range:
                gIdx = 0;
                for gamma in gamma_range:
                    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                    scaler.fit(trainFeatures);
                    svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto')
                    svm.fit(scaler.transform(trainFeatures), trainClassLabels) #nomealizzazione e adattamento
                    predLabels = svm.predict(scaler.transform(devFeatures))
                    print "C= " + str(C) + "; GAMMA= " + str(gamma)
                    A, UAR, ConfMatrix, class_pred, recall_report = compute_score(predLabels, y_train_lab)
                    cGammaScores[cIdx,gIdx] += UAR
                    gIdx += 1;
                cIdx += 1;

        idxs = np.unravel_index(cGammaScores.argmax(), cGammaScores.shape) #trova l'indirizzo all'interno della matrice cGammaScores a cui corrisponde il valore max
        cBestValues[mIdx,fold] = C_range[idxs[0]]       #per ogni cartella (trainset+devset_(1)) si salva il valore di C che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        gBestValues[mIdx,fold] = gamma_range[idxs[1]]   #per ogni cartella (trainset+devset_(1)) si salva il valore di GAMMA che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        scores[mIdx,fold] = cGammaScores.max()

    mIdx += 1

scoresAvg = scores.mean(axis=1)
mIdx = 0

sys.stdout = open(snoreClassPath, 'w')
print "Featureset = " + featureset
print("**** Results ****")
print "N-GAUSS;UAR"
for score in scoresAvg:
    print(str(mixtures[mIdx]) + ";" + str(score))
    mIdx += 1;
idx_max_score = scoresAvg.argmax()

print "best vale of c for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(cBestValues[idx_max_score])
print "best vale of g for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(gBestValues[idx_max_score])

# #save best c-best gamma-best nmix
joblib.dump(mixtures[idx_max_score],os.path.join(scoresPath, "nmixED"))
joblib.dump(cBestValues[idx_max_score],os.path.join(scoresPath, "cBestValuesED"))
joblib.dump(gBestValues[idx_max_score],os.path.join(scoresPath, "gBestValuesED"))
#
#PRINT BEST VALUES
mix = mixtures[idx_max_score]
curSupervecSubPath = os.path.join(supervecPath, "trainset_" + str(fold), str(mix))
trainFeatures = utl.readfeatures(curSupervecSubPath, y)
trainClassLabels = y_train_lab

devFeatures = utl.readfeatures(curSupervecSubPath, yd)
devClassLabels = y_devel_lab

C=cBestValues[idx_max_score]
gamma=gBestValues[idx_max_score]

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler.fit(trainFeatures);
svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto')
svm.fit(scaler.transform(trainFeatures), trainClassLabels)  # nomealizzazione e adattamento
Best_MODEL = svm.get_params()
joblib.dump(Best_MODEL, os.path.join(scoresPath, "best_model"))
predLabels = svm.predict(scaler.transform(devFeatures))
for p in range(0,len(predLabels)):
    predClass = [0, 0, 0, 0]
    index = int(predLabels[p])
    predClass[index] = 1
    if p == 0:
        predProb = predClass
    else:
        predProb=np.vstack((predProb,predClass))

A, UAR, ConfMatrix, class_pred, recall_report = compute_score(predLabels, y_devel_lab)
#
#CREATE ARFF FILES
arff.create_arff(scoresPath, yd, predProb, class_pred)
arff.create_pred(scoresPath, y_devel_lab, y_devel_lit, predProb, class_pred)
arff.create_result(scoresPath, A, UAR, ConfMatrix, recall_report)
#
