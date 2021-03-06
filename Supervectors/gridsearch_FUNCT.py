import numpy as np;
from sklearn.svm import SVC;
import sklearn.preprocessing as preprocessing;
from sklearn.externals import joblib;
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
import os
import sys
import dataset_manupulation as dm
import utils
import arff

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = "ComParE_Functionals"
#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'arff', "ComParE2017_Snore.ComParE.")

ubmsPath = os.path.join(targePath, featureset, "ubms")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score")
snoreClassPath=os.path.join(targePath, featureset, "score");#used for save best c-best gamma-best nmix so that extract_supervector_test.py and test.py can read it

sys.stdout = open(os.path.join(scoresPath,'gridsearch.txt'), 'w')   #log to a file 
print "experiment: "+targePath; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'gridsearch_err.txt'), 'w')   #log to a file 


#variables inizialization
nFolds = 1;
C_range = 2.0 ** np.arange(-5, 15+2, 2);    # libsvm range
gamma_range = 2.0 ** np.arange(-15, 3+2, 2); # libsvm range
mixtures = 2**np.arange(0, 7, 1);

scores      = np.zeros((mixtures.shape[0], nFolds));
cBestValues = np.zeros((mixtures.shape[0], nFolds));
gBestValues = np.zeros((mixtures.shape[0], nFolds));
mIdx = 0;

#LOAD DATASET
trainPath =  featPath + "train.arff"
data = arff.load(open(trainPath))
trainset = data['data']

develPath = featPath + "devel.arff"
data = arff.load(open(develPath))
develset = data['data']

del data

y = []
y_train_lab = []
for seq in trainset:
    name = str(seq.pop(0)).replace('.wav','.npy')
    y.append(name)
    label = seq.pop(-1)
    y_train_lab.append(utils.labToClass(label))

yd = []
y_devel_lab = []
for seq in develset:
    name = str(seq.pop(0)).replace('.wav','.npy')
    yd.append(name)
    label = seq.pop(-1)
    y_devel_lab.append(utils.labToClass(label))




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

    # cm = CM.astype(int)
    # print("FINAL REPORT")
    # print("\t V\t O\t T\t E")
    # print("V \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]) + "\t" + str(cm[0, 2]) + "\t" + str(cm[0, 3]))
    # print("O \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]) + "\t" + str(cm[1, 2]) + "\t" + str(cm[1, 3]))
    # print("T \t" + str(cm[2, 0]) + "\t" + str(cm[2, 1]) + "\t" + str(cm[2, 2]) + "\t" + str(cm[2, 3]))
    # print("E \t" + str(cm[3, 0]) + "\t" + str(cm[3, 1]) + "\t" + str(cm[3, 2]) + "\t" + str(cm[3, 3]))
    #
    # print(classification_report(y_true, y_pred, target_names=['V', 'O', 'T', 'E']))

    return A, UAR, CM, y_pred


for m in mixtures:
    print("Mixture: " + str(m));
    sys.stdout.flush()
    mixScores = np.zeros((nFolds*(nFolds-1), 1));
    fIdx = 0;
    for fold in range(0,nFolds):
        cGammaScores = np.zeros((C_range.shape[0], gamma_range.shape[0])); #inizializza matrice dei punteggi
        print("Fold: " + str(fold));
        sys.stdout.flush()
        #curListPath = os.path.join(ListPath, "trainset_" + str(fold));       #fall_detection/lists/lolo_right/trainset+devset_(1)
        curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold));   #fall_detection/supervectors/lolo_right/trainset+devset_(1)
        for sf in range(0,nFolds):   #per ogni coppia di devset_x trainset_x. In questo caso sono 3 coppie quindi 3 subfold
            print("Subfold: " + str(sf));
            sys.stdout.flush()
            curSupervecSubPath = os.path.join(curSupervecPath, str(m));        #fall_detection/supervectors/lolo_right/trainset+devset_1/(1)/1
            #trainListFile = os.path.join(curListPath, "trainset_" + str(sf) + ".lst");  #fall_detection/lists/lolo_right/trainset+devset_1/trainset_(1).lst
            #trainFilepaths, trainClassLabels, trainDistLabels = utils.readlistfile(trainListFile,labelling);
            trainFeatures =  utils.readfeatures(curSupervecSubPath, y); #contiene tutte le features della lista
            trainClassLabels = y_train_lab

            #devListFile = os.path.join(curListPath, "devset_" + str(sf) + ".lst");      #fall_detection/lists/lolo_right/trainset+devset_1/devset_(1).lst
            #devFilepaths, devClassLabels, devDistLabels = utils.readlistfile(devListFile,labelling);
            devFeatures =  utils.readfeatures(curSupervecSubPath, yd);    #contiene tutte le features della lista
            devClassLabels = y_devel_lab

            #TODO: STORE Metrics
            cIdx = 0;
            for C in C_range:
                gIdx = 0;
                for gamma in gamma_range:
                    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1));
                    scaler.fit(trainFeatures);
                    svm = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='auto');
                    svm.fit(scaler.transform(trainFeatures), trainClassLabels);#nomealizzazione e adattamento
                    predLabels = svm.predict(scaler.transform(devFeatures));
                    A, UAR, ConfMatrix, class_pred = compute_score(predLabels, y_devel_lab)
                    cGammaScores[cIdx,gIdx] += UAR;
                    gIdx += 1;
                cIdx += 1;

        idxs = np.unravel_index(cGammaScores.argmax(), cGammaScores.shape);#trova l'indirizzo all'interno della matrice cGammaScores a cui corrisponde il valore max
        cBestValues[mIdx,fold-1] = C_range[idxs[0]];        #per ogni cartella (trainset+devset_(1)) si salva il valore di C che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        gBestValues[mIdx,fold-1] = gamma_range[idxs[1]];    #per ogni cartella (trainset+devset_(1)) si salva il valore di GAMMA che mi da il punteggio maggiore (il tutto lo fa anche per ogni valore di mixture)
        scores[mIdx,fold-1] = cGammaScores.max();   

    mIdx += 1;

scoresAvg = scores.mean(axis=1);

mIdx = 0;
print("\n**** Results ****\n");
for score in scoresAvg:
    print(str(mixtures[mIdx]) + " " + str(score)); 
    mIdx += 1;
    
idx_max_score = scoresAvg.argmax();
print "best vale of c for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(cBestValues[idx_max_score]);
print "best vale of g for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(gBestValues[idx_max_score]);

#save best c-best gamma-best nmix
joblib.dump(mixtures[idx_max_score],os.path.join(snoreClassPath, "nmix"));
joblib.dump(cBestValues[idx_max_score],os.path.join(snoreClassPath, "cBestValues"));
joblib.dump(gBestValues[idx_max_score],os.path.join(snoreClassPath, "gBestValues"));
        
