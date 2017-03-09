# -*- coding: utf-8 -*-

import numpy as np;
from sklearn.svm import SVC;
import sklearn.preprocessing as preprocessing;
from sklearn.externals import joblib;
from sklearn.metrics import f1_score;
from sklearn.metrics import confusion_matrix;
from sklearn.metrics import classification_report;
import os;
import sys;
import utils;
import scipy.io

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


ListPath = sys.argv[1];
targePath =  sys.argv[2];
labelling = sys.argv[3];
if(labelling=='nclass'):
    label=[ 'fall  ', 'bag   ', 'ball  ', 'basket', 'book  ', 'chair ', 'fork  '];
if(labelling=='biclass'):
    label=[ 'fall  ', 'nofall'];


fallDetectionPath=os.path.join(targePath,'fall_detection/');
supervecPath = os.path.join(fallDetectionPath,"supervectors");
scoresPath= os.path.join(targePath,"score");
sys.stdout = open(os.path.join(scoresPath,'test.txt'), 'w')   #log to a file 
print "experiment: "+targePath; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'test_err.txt'), 'w')   #log to a file 
nMixtures = joblib.load(os.path.join(fallDetectionPath,'nmix'));
Cs =joblib.load(os.path.join(fallDetectionPath,'cBestValues')); # Best 
gammas =joblib.load(os.path.join(fallDetectionPath,'gBestValues')); # Best 

nFolds = 4; #combinazioni possibili del primo lolo



scores = np.zeros((nFolds,1));
allPredLabels = [];
allRefLabels  = [];

#inizialization parameter for plotting roc
dist_true=[];
dist_false=[];
for fold in range(1,nFolds+1):
        print("Fold: " + str(fold));
        C = Cs[fold-1];
        gamma = gammas[fold-1];
        curListPath = ListPath;
        curSupervecPath = os.path.join(supervecPath, "trainset+devset_" + str(fold), "global", str(nMixtures));
        
        trainListFile = os.path.join(curListPath, "trainset+devset_" + str(fold) + ".lst");
        trainFilepaths, trainClassLabels, trainDistLabels = utils.readlistfile(trainListFile,labelling);
        trainFeatures =  utils.readfeatures(curSupervecPath, trainFilepaths);
        
        testListFile = os.path.join(curListPath, "testset_" + str(fold) + ".lst");
        testFilepaths, testClassLabels, testDistLabels = utils.readlistfile(testListFile,labelling);
        testFeatures =  utils.readfeatures(curSupervecPath, testFilepaths);
        
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1));
        scaler.fit(trainFeatures);

        svm = SVC(C=C, kernel='rbf', gamma=gamma);
        svm.fit(scaler.transform(trainFeatures), trainClassLabels);
        predLabels = svm.predict(scaler.transform(testFeatures)); #in ogni testlist ci sono 72 elementi quindi fiacvcio un totale di 72 x 4 = 288 test 
        scores[fold-1] = f1_score(testClassLabels, predLabels);
        for index in range(len(testFeatures)):
            if label[testClassLabels[index]] != label[predLabels[index]]:
                print(str(testFilepaths[index])+"-->"+str(label[predLabels[index]]));
           
                
        allPredLabels.extend(predLabels);
        allRefLabels.extend(testClassLabels); 
        
        ##################################################################
        #saving distances for det/roc plot
        if(labelling=='biclass'):
            distances = svm.decision_function(scaler.transform(testFeatures))
            i=0;
            for d in distances: 
                if testClassLabels[i]==0:
                    dist_true=np.append(dist_true,d)
                else:
                    dist_false=np.append(dist_false,d)
                i+=1;

        ################################################################## 

if(labelling=='biclass'):
    dist_true_path = os.path.join(scoresPath,'dist_true');
    dist_false_path = os.path.join(scoresPath,'dist_false'); 
    scipy.io.savemat(dist_true_path, {'dist_true':dist_true})
    scipy.io.savemat(dist_false_path, {'dist_false':dist_false})     
    
print("\n*** Results ***\n");
print("Average score: " + str(scores.mean()));


print(classification_report(allRefLabels, allPredLabels, target_names=label));
cm = confusion_matrix(allRefLabels, allPredLabels)   

if(labelling=='nclass'):
    print "             fall ||bag  ||ball||basket  ||book ||chair  ||fork  |"

if(labelling=='biclass'):
    print "             fall ||nofall |"

row=range(0,cm.shape[0]);
columns=range(0,cm.shape[1]);
for r in row:
   
    print label[r],
    for column in columns: 
        print "\t\t"+str(cm[r,column]),
    print "";#new line