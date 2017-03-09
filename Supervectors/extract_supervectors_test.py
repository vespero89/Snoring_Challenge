from htkutils import *;
from sklearn import mixture;
from MyMixture import GmmMap;
import numpy;
import os;
import sys;
import copy;
from sklearn.externals import joblib;
import time;
import multiprocessing;

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

ext      = ".htk";



featPath = sys.argv[1];
listsPath = sys.argv[2];
targePath =  sys.argv[3];
totCore=int(sys.argv[4]);
fallDetectionPath=os.path.join(targePath,'fall_detection');
ubmsPath = os.path.join(fallDetectionPath, "ubms");
supervecPath = os.path.join(fallDetectionPath ,"supervectors");
nMixtures = joblib.load(os.path.join(fallDetectionPath,'nmix'));
scoresPath= os.path.join(targePath,"score");
sys.stdout = open(os.path.join(scoresPath,'extract_supervector_test.txt'), 'w')   #log to a file 
print "experiment: "+targePath; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'extract_supervector_test_err.txt'), 'w')   #log to a file 




nFolds = 4;

def process_fold(fold):
    
    print("Processing fold " + str(fold));
    t0 = time.time();

    curListPath = os.path.join(listsPath);
    curUbmsPath = os.path.join(ubmsPath, "trainset+devset_" + str(fold));
    curSupervecPath = os.path.join(supervecPath, "trainset+devset_" + str(fold));
    # Read the train set list
    trainFileName = os.path.join(curListPath, "trainset+devset_" + str(fold) + ".lst");
    with open(trainFileName, "r") as tf:
        trainFilenames = tf.readlines();
    
    # Read dataset size and preallocate
    totFrames = 0;
    for line in trainFilenames:
        nframes, nfeat = readhtkdim(os.path.join(featPath, line.rstrip() + ext));
        totFrames += nframes;
    trainFeat = numpy.zeros((totFrames, nfeat));
    
    # Read the features
    startFrame = 0;
    for line in trainFilenames:
        feat = readhtk(os.path.join(featPath, line.rstrip() + ext));
        trainFeat[startFrame:startFrame+feat.shape[0],:] = feat;
        startFrame = startFrame+feat.shape[0];
    
    # Train the UBM    
    gmm = mixture.GMM(n_components=nMixtures, thresh=1e-3, n_iter=1000, random_state=1);
    ubmPath = os.path.join(curUbmsPath, str(nMixtures));
    gmm.fit(trainFeat);
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
        print("Convergence not reached with " + str(nMixtures) +" mixtures");
    joblib.dump(gmm, os.path.join(ubmPath, "ubm_global"));
    
    # Extract trainset supervectors
    curSupervecSubPath = os.path.join(curSupervecPath, "global", str(nMixtures));
    if (not os.path.exists(curSupervecSubPath)):
        try: #handle the simultaneous creation of folders from multiple processes
            os.makedirs(curSupervecSubPath);
        except OSError, e:
            if e.errno != 17:
                raise   
            else:
                print "OSError.errno 17 ignored"
            pass
    for line in trainFilenames:
        gmmMap = GmmMap(n_components=nMixtures, thresh=1e-3, n_iter=5, params="m");
        gmmMap.weights_ = copy.deepcopy(gmm.weights_);
        gmmMap.means_   = copy.deepcopy(gmm.means_);
        gmmMap.covars_  = copy.deepcopy(gmm.covars_);
        feat = readhtk(os.path.join(featPath, line.rstrip() + ext));
        gmmMap.map_adapt(feat);
        svFilePath = os.path.join(curSupervecSubPath, os.path.basename(line.rstrip()));
        joblib.dump(gmmMap.means_, svFilePath);
            
    # Extract testset supervectors
    testsetFileName = os.path.join(curListPath, "testset_" + str(fold) + ".lst");
    with open(testsetFileName, "r") as df:
        testFilenames = df.readlines();
        
    for line in testFilenames:
        gmmMap = GmmMap(n_components=nMixtures, thresh=1e-3, n_iter=5, params="m");
        gmmMap.weights_ = copy.deepcopy(gmm.weights_);
        gmmMap.means_   = copy.deepcopy(gmm.means_);
        gmmMap.covars_  = copy.deepcopy(gmm.covars_);
        feat = readhtk(os.path.join(featPath, line.rstrip() + ext));
        gmmMap.map_adapt(feat);
        svFilePath = os.path.join(curSupervecSubPath, os.path.basename(line.rstrip()));
        joblib.dump(gmmMap.means_, svFilePath);
    
    t1 = time.time();
    print("Fold "+str(fold)+"--Time: "+str(t1-t0));
        
#*******************************************START SCRIPT******************************************        


number_of_process=0;

jobs=[];
oneIsFree=True;
t_start=time.time();
if __name__ == '__main__':
   for fold in range(1,nFolds+1):
        while oneIsFree==False:
            for p in jobs:
                if p.is_alive()==False:
                    oneIsFree=True;
                    number_of_process-=1;
                    print("numero processi attuali"+str(number_of_process));
                    p.join();
                    jobs.remove(p);
            time.sleep(1);

        else:
            p = multiprocessing.Process(target=process_fold, args=(fold,))
            jobs.append(p)
            p.start()
            number_of_process+=1;
            print("numero processi attuali"+str(number_of_process));
            if(number_of_process>=totCore):
                oneIsFree=False;


for p in jobs:
    p.join();
t_end=time.time();
print("Total time is: "+str(t_end-t_start));
            
