import sys
sys.path.append('..')

from sklearn import mixture
from MyMixture import GmmMap
import numpy as np
import os
import sys
import copy
from sklearn.externals import joblib
import multiprocessing
import time

import utils.dataset_manupulation as dm

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

featureset = 'SCAT'
filetype = 'htk'


#TODO ADD ROOTDIR PARAM
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

ubmsPath = os.path.join(targePath, featureset, "ubms_INV")
supervecPath = os.path.join(targePath, featureset, "supervectors_INV")
scoresPath = os.path.join(targePath, featureset, "score_INV")

# create directory if needed
if (not os.path.exists(scoresPath)):
    os.makedirs(scoresPath)
if (not os.path.exists(ubmsPath)):
    os.makedirs(ubmsPath)
if (not os.path.exists(supervecPath)):
    os.makedirs(supervecPath)

sys.stdout = open(os.path.join(scoresPath, 'extract_supervector.txt'), 'w')  # log to a file
print "experiment: " + targePath  # to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath, 'extract_supervector_err.txt'), 'w')  # log to a file


mixtures = np.arange(0, 7, 1);
mixtures = 2 ** mixtures;
nFolds = 1

def process_subfold(sf,fold):
    print("Fold "+str(fold));
    t0 = time.time();

    snoring_dataset = dm.load_ComParE2017(featPath, filetype)  # load dataset
    trainset, develset, testset = dm.split_ComParE2017_simple(snoring_dataset)  # creo i trainset per calcolare media e varianza per poter normalizzare
    del snoring_dataset

    # Read dataset size and preallocate
    a=trainset[0][1].shape
    if (filetype == 'npy'):
        nfeat = a[0]
    else:
        nfeat = a[1]

    # Read the features
    trainFeat=np.empty([1,nfeat])
    #for seq in trainset:
    for seq in develset:
        if (filetype == 'npy'):
            feat = seq[1].transpose()
        else:
            feat = seq[1]
        # metto tutte le features in una matrice che poi passero al gmm.fit per adattaare l'UBM
        trainFeat = np.vstack((trainFeat, feat))
    trainFeat = np.delete(trainFeat, 0, 0)
    print("DONE!")

    #trainFeat = trainFeat.astype(dtype='float32')


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
        joblib.dump(gmm, os.path.join(ubmPath, "ubm_" + str(sf)));         #salvo l'ubm. mi crea le varie compie tipo ubm_1_02 ecc... per poterle magari riutilizzare per il debug

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
            if (filetype == 'npy'):
                feat = seq[1].transpose()
            else:
                feat = seq[1]
            gmmMap.map_adapt(feat);
            #adatta l'ubm ad una feature
            #means_and_covars = np.hstack((gmmMap.means_,gmmMap.covars_))
            svFilePath = os.path.join(curSupervecSubPath, os.path.basename(seq[0]));
            joblib.dump(gmmMap.means_, svFilePath);
            #joblib.dump(means_and_covars, svFilePath);

        # Extract devset supervectors
        for seq in develset:
            gmmMap = GmmMap(n_components=m, n_iter=5, params="m");
            gmmMap.weights_ = copy.deepcopy(gmm.weights_);
            gmmMap.means_   = copy.deepcopy(gmm.means_);
            gmmMap.covars_  = copy.deepcopy(gmm.covars_);
            if (filetype == 'npy'):
                feat = seq[1].transpose()
            else:
                feat = seq[1]
            gmmMap.map_adapt(feat)
            # adatta l'ubm ad una feature
            #means_and_covars = np.hstack((gmmMap.means_, gmmMap.covars_))
            svFilePath = os.path.join(curSupervecSubPath,  os.path.basename(seq[0]));
            joblib.dump(gmmMap.means_, svFilePath);
            #joblib.dump(means_and_covars, svFilePath);

        for seq in testset:
            gmmMap = GmmMap(n_components=m, n_iter=5, params="m");
            gmmMap.weights_ = copy.deepcopy(gmm.weights_);
            gmmMap.means_   = copy.deepcopy(gmm.means_);
            gmmMap.covars_  = copy.deepcopy(gmm.covars_);
            if (filetype == 'npy'):
                feat = seq[1].transpose()
            else:
                feat = seq[1]
            gmmMap.map_adapt(feat)
            # adatta l'ubm ad una feature
            #means_and_covars = np.hstack((gmmMap.means_, gmmMap.covars_))
            svFilePath = os.path.join(curSupervecSubPath,  os.path.basename(seq[0]));
            joblib.dump(gmmMap.means_, svFilePath);
            #joblib.dump(means_and_covars, svFilePath);

    t1 = time.time();
    
    print("Fold "+str(fold) + "--Time: "+str(t1-t0));


#****************************************start script************************************


number_of_process=1;
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
                        print("numero processi attuali" + str(number_of_process));
                        p.join();
                        jobs.remove(p);
                time.sleep(1);

            else:
                p = multiprocessing.Process(target=process_subfold, args=(sf,fold,))
                jobs.append(p)
                p.start()
                number_of_process += 1;
                print("numero processi attuali" + str(number_of_process));
                if (number_of_process >= totCore):
                    oneIsFree = False;
                

for p in jobs:
    p.join();     
t_end=time.time();
print("Total time is: "+str(t_end-t_start));