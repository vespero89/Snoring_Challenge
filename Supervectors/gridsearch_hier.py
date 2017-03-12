import sys
sys.path.append('..')
import numpy as np
from sklearn.svm import SVC
import sklearn.preprocessing as preprocessing
from sklearn.externals import joblib
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
import os;
import sys
import utils.dataset_manupulation as dm
import utils.utils as utl

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = 'FBANK_E_D'
filetype = 'htk'

#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

ubmsPath = os.path.join(targePath, featureset, "ubms")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score_hier")
if not os.path.exists(scoresPath):
    os.makedirs(scoresPath)
snoreClassPath =os.path.join(targePath, featureset, "score_hier","final_score.csv")#used for save best c-best gamma-best nmix so that extract_supervector_test.py and test.py can read it

sys.stdout = open(os.path.join(scoresPath,'gridsearch.txt'), 'w')   #log to a file
print "experiment: "+targePath; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'gridsearch_err.txt'), 'w')   #log to a file

#variables inizialization
nFolds = 1;
C_range = 2.0 ** np.arange(-5, 15+2, 2)    # libsvm range
gamma_range = 2.0 ** np.arange(-15, 3+2, 2) # libsvm range
mixtures = 2**np.arange(0, 7, 1)

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

y_train, y_train_lab = dm.label_organize(trainset_l, y)
y_devel, y_devel_lab = dm.label_organize(develset_l, yd)

##EXTEND TRAINSET
#y_train_lab = np.append(y_train_lab,y_devel_lab[:140])
#y_devel_lab = y_devel_lab[140:]

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

#FIRST STAGE: BINARY CLASSIFICATION
print "First Stage: Binary Classification"
#Organize label for BIN CLASSIFICATOR
y_train_bin = dm.data_bin_organize(trainset_l,y)
y_devel_bin = dm.data_bin_organize(develset_l,yd)

for m in mixtures:
    print("Mixture: " + str(m))
    sys.stdout.flush()
    mixScores = np.zeros((nFolds*(nFolds-1), 1))
    fIdx = 0;
    for fold in range(0,nFolds):
        cGammaScores = np.zeros((C_range.shape[0], gamma_range.shape[0])) #inizializza matrice dei punteggi
        cGammaScoresUAR = np.zeros((C_range.shape[0], gamma_range.shape[0]))  # inizializza matrice dei punteggi
        curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold))
        for sf in range(0,nFolds):
            curSupervecSubPath = os.path.join(curSupervecPath, str(m))
            trainFeatures = utl.readfeatures(curSupervecSubPath, y)
            trainClassLabels = y_train_bin

            devFeatures = utl.readfeatures(curSupervecSubPath, yd)
            devClassLabels = y_devel_bin

            ##EXTEND TRAINSET
            #trainFeatures = np.vstack((trainFeatures,devFeatures[:140,:]))
            #devFeatures = devFeatures[140:]

            cIdx = 0;
            for C in C_range:
                gIdx = 0;
                for gamma in gamma_range:
                    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                    scaler.fit(trainFeatures);
                    svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto')
                    svm.fit(scaler.transform(trainFeatures), trainClassLabels) #nomealizzazione e adattamento
                    predLabels = svm.predict(scaler.transform(devFeatures))
                    A, UAR, ConfMatrix, class_pred = compute_score(predLabels, y_devel_bin)
                    cGammaScores[cIdx,gIdx] += A #EVALUATION on ACCURACY
                    gIdx += 1;
                cIdx += 1;

        idxs = np.unravel_index(cGammaScores.argmax(), cGammaScores.shape) #trova l'indirizzo all'interno della matrice cGammaScores a cui corrisponde il valore max
        cBestValues[mIdx,fold-1] = C_range[idxs[0]]       #per ogni cartella (trainset+devset_(1)) si salva il valore di C che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        gBestValues[mIdx,fold-1] = gamma_range[idxs[1]]   #per ogni cartella (trainset+devset_(1)) si salva il valore di GAMMA che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        scores[mIdx,fold-1] = cGammaScores.max()

    mIdx += 1

scoresAvg = scores.mean(axis=1)
mIdx = 0

sys.stdout = open(snoreClassPath, 'w')
print "Featureset = " + featureset
print("**** Results Binary Classification ****")
print "N-GAUSS;Accuracy"
for score in scoresAvg:
    print(str(mixtures[mIdx]) + ";" + str(score))
    mIdx += 1;
idx_max_score = scoresAvg.argmax()

print "best vale of c for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(cBestValues[idx_max_score])
print "best vale of g for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(gBestValues[idx_max_score])

#save best c-best gamma-best nmix
joblib.dump(mixtures[idx_max_score],os.path.join(scoresPath, "nmix_bin"))
joblib.dump(cBestValues[idx_max_score],os.path.join(scoresPath, "cBestValues_bin"))
joblib.dump(gBestValues[idx_max_score],os.path.join(scoresPath, "gBestValues_bin"))

print "Producing predictions from best model"
curSupervecPath_bin = os.path.join(supervecPath, "trainset_0", str(mixtures[idx_max_score]));
trainFeatures = utl.readfeatures(curSupervecPath_bin, y)
testFeatures = utl.readfeatures(curSupervecPath_bin, yd)
trainClassLabels = y_train_lab
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1));
scaler.fit(trainFeatures);
svm = SVC(C=cBestValues[idx_max_score], kernel='rbf', gamma=gBestValues[idx_max_score], class_weight='auto');
svm.fit(scaler.transform(trainFeatures), trainClassLabels);
predLab_bin = svm.predict(scaler.transform(testFeatures));
A, UAR, CM, class_pred = compute_score(predLab_bin, y_devel_bin)
cm = CM.astype(int)

sys.stdout = open(os.path.join(scoresPath,'gridsearch.txt'), 'a')   #log to a file
sys.stderr = open(os.path.join(scoresPath,'gridsearch_err.txt'), 'a')   #log to a file

print "SCORES AFTER FIRST STAGE:"
print("Accuracy: " + str(A))
print("UAR: " + str(UAR))
cm = CM.astype(int)
print("FINAL REPORT")
print("\t\t V\t OTE")
print(" V  \t" + str(cm[0, 0]) + "\t  " + str(cm[0, 1]))
print("OTE \t" + str(cm[1, 0]) + "\t  " + str(cm[1, 1]))


#SECOND STAGE - MULTICLASS
print "Second Stage: Multiclass Classification"

scores      = np.zeros((mixtures.shape[0], nFolds))
scores_G    = np.zeros((mixtures.shape[0], nFolds))
cBestValues = np.zeros((mixtures.shape[0], nFolds))
gBestValues = np.zeros((mixtures.shape[0], nFolds))
cBestValues_G = np.zeros((mixtures.shape[0], nFolds))
gBestValues_G = np.zeros((mixtures.shape[0], nFolds))
mIdx = 0;
for m in mixtures:
    print("Mixture: " + str(m))
    sys.stdout.flush()
    mixScores = np.zeros((nFolds*(nFolds-1), 1))
    fIdx = 0;
    for fold in range(0,nFolds):
        cGammaScores = np.zeros((C_range.shape[0], gamma_range.shape[0])) #inizializza matrice dei punteggi
        cGammaScores_GLOB = np.zeros((C_range.shape[0], gamma_range.shape[0]))  # inizializza matrice dei punteggi
        curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold))
        for sf in range(0,nFolds):
            curSupervecSubPath = os.path.join(curSupervecPath, str(m))
            trainFeatures = utl.readfeatures(curSupervecSubPath, y)
            devFeatures = utl.readfeatures(curSupervecSubPath, yd)
            # Organize data for Multiclass-CLASSIFICATOR
            trainFeatures_class, _, y_train_class = dm.data_class_organize(trainset_l, y, trainFeatures)
            # Questi devel sono utili SOLO per valutare il buon training (o meno) del Multiclass
            devFeatures_class, _, y_devel_class = dm.data_class_organize(develset_l, yd, devFeatures)

            trainClassLabels = y_train_class
            devClassLabels = y_devel_class

            ##EXTEND TRAINSET
            #trainFeatures = np.vstack((trainFeatures,devFeatures[:140,:]))
            #devFeatures = devFeatures[140:]

            cIdx = 0;
            for C in C_range:
                gIdx = 0;
                for gamma in gamma_range:
                    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                    scaler.fit(trainFeatures);
                    svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto')
                    svm.fit(scaler.transform(trainFeatures_class), trainClassLabels) #nomealizzazione e adattamento
                    predLabels = svm.predict(scaler.transform(devFeatures_class))
                    A, UAR, ConfMatrix, class_pred = compute_score(predLabels, y_devel_class)
                    cGammaScores[cIdx, gIdx] += A

                    #GLOBAL Predictions - SCORES
                    testFeatures = utl.readfeatures(curSupervecSubPath, yd)
                    c_to_remove = []
                    i = 0
                    for c in range(0, len(predLab_bin)):
                        if predLab_bin[c] == 0:
                            c_to_remove.append(c)
                    testFeatures = np.delete(testFeatures, c_to_remove, axis=0)
                    predLabels_class = svm.predict(scaler.transform(testFeatures))
                    output_global = []
                    i = 0
                    for c in predLab_bin:
                        if c == 0:
                            output_global.append(0)
                        else:
                            output_global.append((predLabels_class[i] + 1))
                            i += 1
                    output_global = np.asarray(output_global)

                    A_G, UAR_G, ConfMatrix_G, class_pred_G = compute_score(output_global,y_devel_lab)
                    cGammaScores_GLOB[cIdx,gIdx] += UAR_G
                    gIdx += 1;
                cIdx += 1;

        idxs = np.unravel_index(cGammaScores.argmax(), cGammaScores.shape)
        cBestValues[mIdx,fold-1] = C_range[idxs[0]]
        gBestValues[mIdx,fold-1] = gamma_range[idxs[1]]
        scores[mIdx,fold-1] = cGammaScores.max()

        idxs_G = np.unravel_index(cGammaScores_GLOB.argmax(), cGammaScores_GLOB.shape)
        cBestValues_G[mIdx, fold - 1] = C_range[idxs_G[0]]
        gBestValues_G[mIdx, fold - 1] = gamma_range[idxs_G[1]]
        scores_G[mIdx, fold - 1] = cGammaScores_GLOB.max()

    mIdx += 1

scoresAvg = scores.mean(axis=1)
scoresAvg_G = scores_G.mean(axis=1)


sys.stdout = open(snoreClassPath, 'a')
mIdx = 0
print("**** Results MultiClass****")
print "N-GAUSS;Accuracy"
for score in scoresAvg:
    print(str(mixtures[mIdx]) + ";" + str(score))
    mIdx += 1;
idx_max_score = scoresAvg.argmax()

print "best vale of c for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(cBestValues[idx_max_score])
print "best vale of g for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(gBestValues[idx_max_score])

#save best c-best gamma-best nmix
joblib.dump(mixtures[idx_max_score],os.path.join(scoresPath, "nmix_class"))
joblib.dump(cBestValues[idx_max_score],os.path.join(scoresPath, "cBestValues_class"))
joblib.dump(gBestValues[idx_max_score],os.path.join(scoresPath, "gBestValues_class"))


mIdx = 0
print("**** Results GLOBAL****")
print "N-GAUSS;UAR"
for score in scoresAvg_G:
    print(str(mixtures[mIdx]) + ";" + str(score))
    mIdx += 1;
idx_max_score = scoresAvg_G.argmax()

print "best vale of c for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(cBestValues_G[idx_max_score])
print "best vale of g for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(gBestValues_G[idx_max_score])

#save best c-best gamma-best nmix
joblib.dump(mixtures[idx_max_score],os.path.join(scoresPath, "nmix_glob"))
joblib.dump(cBestValues_G[idx_max_score],os.path.join(scoresPath, "cBestValues_glob"))
joblib.dump(gBestValues_G[idx_max_score],os.path.join(scoresPath, "gBestValues_glob"))
