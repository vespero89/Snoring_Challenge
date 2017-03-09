import sys
sys.path.append('..')
import numpy as np
import ANN_classifier as ann
import os
import sys
import utils.dataset_manupulation as dm
import utils.utils as utl
import argparse
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


parser = argparse.ArgumentParser(description="ComParE2017 Snoring Classification")

# Global params
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-it", "--featureset", dest="featureset", default='logmel')

# Layout params
parser.add_argument('-ds','--dense-shape', dest="dense_layers_inputs",  nargs='+', default=[128,128])
parser.add_argument('-ad','--dense-activation', dest="dense_activation", default="tanh", choices = ["tanh","relu"])
parser.add_argument('-i','--init', dest="init", default="glorot_uniform", choices = ["glorot_uniform"])
parser.add_argument('-wr','--w-reg', dest="w_reg", default=None) # in autoencoder va usato con eval('funzione(parametri)')
parser.add_argument('-br','--b-reg', dest="b_reg", default=None)
parser.add_argument('-ar','--act-reg', dest="a_reg", default=None)
parser.add_argument('-wc','--w-constr', dest="w_constr", default=None)
parser.add_argument('-bc','--b-constr', dest="b_constr", default=None)
parser.add_argument('-nb', '--no-bias', dest = "bias", default = True, action = 'store_false')
parser.add_argument('-p', '--end-pool', dest = "pool_only_to_end", default = False, action = 'store_true')


# fit params
parser.add_argument("-e", "--epoch", dest = "epoch", default=100, type=int)
parser.add_argument("-ns", "--no-shuffle", dest = "shuffle", default = True, action = 'store_false')
parser.add_argument("-bs", "--batch-size", dest = "batch_size", default=16, type=int)
parser.add_argument("-vs", "--validation-split", dest = "valid_split", default=0.1, type=float)
parser.add_argument("-f", "--fit-net", dest = "fit_net", default = True, action = 'store_true')
parser.add_argument("-o", "--optimizer", dest = "optimizer", default="adam", choices = ["adadelta","adam", "sgd"])
parser.add_argument("-l", "--loss", dest = "loss", default="categorical_crossentropy", choices = ["categorical_crossentropy"])


###############################################################################


args = parser.parse_args()


if (args.config_filename is not None):
    with open(args.config_filename, 'r') as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        if '#' not in line:
            arguments.extend(line.split())
    # First parse the arguments specified in the config file
    args = parser.parse_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)
    # special.default values

featureset = args.featureset
filetype = 'htk'

tmp = ''.join(args.dense_layers_inputs)
sls = tmp.split(',')
nLls = [int(s) for s in sls]
args.dense_layers_inputs = nLls

#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score")
snoreClassFile = os.path.join(targePath, featureset, "score","RandMLPSearch.txt");#used for save best c-best gamma-best nmix so that extract_supervector_test.py and test.py can read it

sys.stdout = open(os.path.join(scoresPath,'gridsearch_ANN.txt'), 'w')   #log to a file
print "experiment: "+targePath; #to have the reference to experiments in text files
sys.stderr = open(os.path.join(scoresPath,'gridsearch_err_ANN.txt'), 'w')   #log to a file

tag = arguments
tag = tag[1::2]
tag = reduce(lambda a, b: a + "\t" + b, tag).replace('--resume', '').replace('/', '-').replace('--', ';').replace('True', 'T').replace('False', 'F')
line = "\n" + os.path.basename(args.config_filename) + "\t" + tag + "\t"
with open(snoreClassFile , 'a+') as f:
    f.write(line)

#variables inizialization
nFolds = 1;
mixtures = 2**np.arange(0, 7, 1);

scores      = np.zeros((mixtures.shape[0], nFolds));
mIdx = 0;

#LOAD DATASET
snoring_dataset = dm.load_ComParE2017(featPath, filetype)  # load dataset
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



for m in mixtures:
    print("Mixture: " + str(m));
    sys.stdout.flush()
    fIdx = 0;
    for fold in range(0,nFolds):
        print("Fold: " + str(fold));
        sys.stdout.flush()
        curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold));
        for sf in range(0,nFolds):
            print("Subfold: " + str(sf));
            sys.stdout.flush()
            curSupervecSubPath = os.path.join(curSupervecPath, str(m));
            trainFeatures = utl.readfeatures(curSupervecSubPath, y); #contiene tutte le features della lista
            mean = np.mean(trainFeatures)
            std = np.std(trainFeatures)
            trainFeatures = ((trainFeatures - mean) / std)

            trainClassLabels = y_train

            devFeatures = utl.readfeatures(curSupervecSubPath, yd);    #contiene tutte le features della lista
            devFeatures = ((devFeatures - mean) / std)
            devClassLabels = y_devel

            MODEL_PATH = os.path.join(curSupervecSubPath,'ann')
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)

            # TODO SCALE INPUT DATA
            input_shape = trainFeatures.shape
            net = ann.Classifier(input_shape, args.fit_net)
            # net.define_arch()
            net.define_parametric_arch(args)
            # parametri di default anche per compile e fit
            net.model_compile(args.optimizer, args.loss)
            loss = net.model_fit(trainFeatures, y_train, MODEL_PATH, args)
            val_loss = np.asarray(loss[0].history['val_loss'])
            train_loss = np.asarray(loss[0].history['loss'])
            best_epoch = np.argmin(val_loss)
            best_val_loss = val_loss[best_epoch]
            best_loss = train_loss[best_epoch]

            # TODO LOAD ONLY BEST MODEL
            output = net.class_predictions(devFeatures, MODEL_PATH)

            A, UAR, ConfMatrix, class_pred = net.compute_score(output, y_devel_lab)
            scores[mIdx] = UAR

    mIdx += 1;

mIdx = 0;
print("\n**** Results ****\n");
for score in scores:
    print(str(mixtures[mIdx]) + " " + str(score));
    mIdx += 1;

idx_max_score = scores.argmax();
print "best vale of AUC for " + str(mixtures[idx_max_score]) +" gaussian : "+ str(scores[idx_max_score]);


line = str(mixtures[idx_max_score]) + "\t" + str(scores[idx_max_score])
with open(snoreClassFile , 'a+') as f:
    f.write(line)
