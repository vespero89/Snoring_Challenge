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
ivecPath = os.path.join(targePath, featureset, "ivectors")
scoresPath = os.path.join(targePath, featureset, "score_ivec")
snoreClassPath =os.path.join(targePath, featureset, "score_ivec","final_score.csv")#used for save best c-best gamma-best nmix so that extract_supervector_test.py and test.py can read it

sys.stdout = open(os.path.join(scoresPath,'gridsearch_ivec.txt'), 'w')   #log to a file
print "experiment: "+targePath #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'gridsearch_err_ivec.txt'), 'w')   #log to a file

#variables inizialization
vec_lengths = [250, 400]
C_range = 2.0 ** np.arange(-5, 15+2, 2)    # libsvm range
gamma_range = 2.0 ** np.arange(-15, 3+2, 2) # libsvm range
mixtures = 2**np.arange(0, 7, 1)
mixtures = 2**np.arange(5, 6, 1)
nFolds = 1

scores      = np.zeros((mixtures.shape[0], len(vec_lengths)))
cBestValues = np.zeros((mixtures.shape[0], len(vec_lengths)))
gBestValues = np.zeros((mixtures.shape[0], len(vec_lengths)))


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

y_train, y_train_lab = dm.label_organize(trainset_l, y)
y_devel, y_devel_lab = dm.label_organize(develset_l, yd)

##EXTEND TRAINSET
#y_train_lab = np.append(y_train_lab,y_devel_lab[:140])
#y_devel_lab = y_devel_lab[140:]

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

    return A, UAR, CM, y_pred

mIdx = 0
for m in mixtures:
    print("Mixture: " + str(m))
    sys.stdout.flush()
    curIvecPath = os.path.join(ivecPath, str(m))
    ividx = 0
    for ivl in vec_lengths:
        cGammaScores = np.zeros((C_range.shape[0], gamma_range.shape[0])) #inizializza matrice dei punteggi
        print("I-Vector Length: " + str(ivl))
        sys.stdout.flush()
        curIvecSubPath = os.path.join(curIvecPath, str(ivl))
        trainFeatures = utl.readIvecFeatures(curIvecSubPath, y) #TODO READ SUPERVEC
        trainClassLabels = y_train_lab

        devFeatures = utl.readIvecFeatures(curIvecSubPath, yd)
        devClassLabels = y_devel_lab

        ##EXTEND TRAINSET
        #trainFeatures = np.vstack((trainFeatures,devFeatures[:140,:]))
        #devFeatures = devFeatures[140:]

        cIdx = 0
        for C in C_range:
            gIdx = 0
            for gamma in gamma_range:
                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                scaler.fit(trainFeatures);
                svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto')
                svm.fit(scaler.transform(trainFeatures), trainClassLabels) #nomealizzazione e adattamento
                predLabels = svm.predict(scaler.transform(devFeatures))
                A, UAR, ConfMatrix, class_pred = compute_score(predLabels, y_devel_lab)
                cGammaScores[cIdx,gIdx] += UAR
                gIdx += 1;
            cIdx += 1;
        idxs = np.unravel_index(cGammaScores.argmax(), cGammaScores.shape) #trova l'indirizzo all'interno della matrice cGammaScores a cui corrisponde il valore max
        cBestValues[mIdx, ividx] = C_range[idxs[0]]       #per ogni cartella (trainset+devset_(1)) si salva il valore di C che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        gBestValues[mIdx, ividx] = gamma_range[idxs[1]]   #per ogni cartella (trainset+devset_(1)) si salva il valore di GAMMA che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        scores[mIdx, ividx] = cGammaScores.max()

        ividx += 1
    mIdx += 1

sys.stdout = open(snoreClassPath, 'w')
print "Featureset = " + featureset
print("**** Results ****")
print "NOTES; N-GAUSS;i-Vector len; UAR"

mIdx = 0
for m in mixtures:
    scoresMix = scores[mIdx, :]
    ividx = 0
    for ivl in vec_lengths:
        print(";" + str(m) + ";" + str(ivl) + ";" + str(scores[mIdx, ividx]))
        idx_max_score = scoresMix.argmax()
        print "best vale of c for " + str(vec_lengths[idx_max_score]) +" I-Vector Length and "+ str(mixtures[mIdx]) +" gaussian : "+ str(cBestValues[mIdx,idx_max_score])
        print "best vale of g for " + str(vec_lengths[idx_max_score]) +" I-Vector Length and "+ str(mixtures[mIdx]) +" gaussian : "+ str(gBestValues[mIdx, idx_max_score])
        ividx += 1
    mIdx += 1

