from sklearn import mixture
from MyMixture import GmmMap
import numpy as np
import os
import sys
import copy
from sklearn.externals import joblib
import multiprocessing
import time
import arff

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = "ComParE_Functionals"

#TODO ADD ROOTDIR PARAM
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'arff','ComParE2017_Snore.ComParE.')

ubmsPath = os.path.join(targePath, featureset, "ubms")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score")

# create directory if needed
if (not os.path.exists(scoresPath)):
    os.makedirs(scoresPath)
if (not os.path.exists(ubmsPath)):
    os.makedirs(ubmsPath)
if (not os.path.exists(supervecPath)):
    os.makedirs(supervecPath)

#sys.stdout = open(os.path.join(scoresPath, 'extract_supervector.txt'), 'w')  # log to a file
print "experiment: " + targePath  # to have the reference to experiments in text files
#sys.stderr = open(os.path.join(scoresPath, 'extract_supervector_err.txt'), 'w')  # log to a file


mixtures = np.arange(0, 7, 1);
mixtures = 2 ** mixtures;
nFolds = 1



def process_subfold(sf,fold):
    print("Fold "+str(fold));
    t0 = time.time();

    trainPath =  featPath + "train.arff"
    data = arff.load(open(trainPath))
    trainset = data['data']

    develPath = featPath + "devel.arff"
    data = arff.load(open(develPath))
    develset = data['data']

    del data

    nFrames = len(trainset)
    nFeat = len(trainset[0]) - 2

    trainFeat = np.zeros((nFrames,nFeat))
    startFrame = 0
    for seq in trainset:
        feat = np.asarray(seq[1:len(seq)-1])
        # metto tutte le features in una matrice che poi passero al gmm.fit per adattaare l'UBM
        trainFeat[startFrame,:] = feat



    for m in mixtures:
        # Train the UBM
        print("Fold "+str(fold)+"-->Mixture: "+str(m)+" ");
        sys.stdout.flush();
        gmm = mixture.GMM(n_components=m, n_iter=1000, random_state=1);
        gmm.fit(trainFeat);
        ubmPath = os.path.join(curUbmsPath, str(m));
        if (not os.path.exists(ubmPath)):
            try:#handle the simultaneous creation of folders from multiple processes
                os.makedirs(ubmPath);
            except OSError, e:
                if e.errno != 17:
                    raise   
                else:
                    print "OSError.errno 17 ignored"
                pass
        if (not gmm.converged_):
            print("Fold "+str(fold)+"-->Convergence not reached with " + str(m) +" mixtures");
        joblib.dump(gmm, os.path.join(ubmPath, "ubm_" + str(sf)));
        # Extract trainset supervectors
        curSupervecSubPath = os.path.join(curSupervecPath, str(m));
        if (not os.path.exists(curSupervecSubPath)):
            try:#handle the simultaneous creation of folders from multiple processes
                os.makedirs(curSupervecSubPath);
            except OSError, e:
                if e.errno != 17:
                    raise   
                else:
                    print "OSError.errno 17 ignored"
                pass

        for seq in trainset:
            gmmMap = GmmMap(n_components=m, n_iter=5, params="m");
            #gli passo i parametri del'ubm calcolato in precedenza
            gmmMap.weights_ = copy.deepcopy(gmm.weights_);
            gmmMap.means_   = copy.deepcopy(gmm.means_);
            gmmMap.covars_  = copy.deepcopy(gmm.covars_);
            #leggo una feature
            #adatta l'ubm ad una feature
            svFilePath = os.path.join(curSupervecSubPath, os.path.basename(seq[0]));
            feat = np.asarray(seq[1:len(seq)-1])
            feat = np.reshape(feat, (1, len(feat)))
            gmmMap.map_adapt(feat);
            joblib.dump(gmmMap.means_, svFilePath);


        for seq in develset:
            gmmMap = GmmMap(n_components=m, n_iter=5, params="m");
            gmmMap.weights_ = copy.deepcopy(gmm.weights_);
            gmmMap.means_   = copy.deepcopy(gmm.means_);
            gmmMap.covars_  = copy.deepcopy(gmm.covars_);
            svFilePath = os.path.join(curSupervecSubPath, os.path.basename(seq[0]));
            feat = np.asarray(seq[1:len(seq) - 1])
            feat = np.reshape(feat, (1, len(feat)))
            gmmMap.map_adapt(feat);
            joblib.dump(gmmMap.means_, svFilePath);
    
    t1 = time.time();
    
    print("Fold "+str(fold) + "--Time: "+str(t1-t0));


#****************************************start script************************************


number_of_process=0;
totCore=2;
jobs=[];
oneIsFree=True;
t_start=time.time();
for fold in range(nFolds):
    print("Processing fold " + str(fold));
    curListPath = os.path.join(listPath, "trainset_" + str(fold));
    curUbmsPath = os.path.join(ubmsPath, "trainset_" + str(fold));
    curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold));
    if __name__ == '__main__':
        # quando un processo finisce ne parte subito un' altro
        for sf in range(0, nFolds):
            while oneIsFree == False:
                for p in jobs:
                    if p.is_alive() == False:
                        oneIsFree = True;
                        number_of_process -= 1;
                        print("numero processi attuali " + str(number_of_process));
                        p.join();
                        jobs.remove(p);
                time.sleep(1);

            else:
                p = multiprocessing.Process(target=process_subfold, args=(sf,fold,))
                jobs.append(p)
                p.start()
                number_of_process += 1;
                print("numero processi attuali " + str(number_of_process));
                if (number_of_process >= totCore):
                    oneIsFree = False;
                

for p in jobs:
    p.join();     
t_end=time.time();
print("Total time is: "+str(t_end-t_start));