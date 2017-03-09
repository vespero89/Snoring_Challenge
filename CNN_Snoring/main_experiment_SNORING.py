#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np
import sys
sys.path.append('..')

np.random.seed(888)#for experiment repetibility
import CNN_classifier as cnn
import utils.dataset_manupulation as dm
from os import path, makedirs
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="ComParE2017 Snoring Classification")

# Global params
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-it", "--featureset", dest="featureset", default='logmel_NEW')
#parser.add_argument("-mn", "--model-path", dest="model_path", default='experiments/my_model.h5')

# CNN params
#parser.add_argument('-is','--cnn-input-shape', dest="cnn_input_shape", nargs='+', default=[1, 129, 197], type=int)
parser.add_argument('-kn','--kernels-number', dest="kernel_number", nargs='+', default=[16, 8, 8])
parser.add_argument('-ks','--kernel-shape', dest="kernel_shape", nargs='+', default=([3, 3], [3, 3], [2, 2])) # default after parser.parse_args()
parser.add_argument('-mp','--max-pool-shape', dest="m_pool", nargs='+', default=([3, 3], [3, 3], [2, 2])) # default after parser.parse_args()
parser.add_argument('-ds','--dense-shape', dest="dense_layers_inputs",  nargs='+', default=[512, 512])
parser.add_argument('-i','--cnn-init', dest="cnn_init", default="glorot_uniform", choices = ["glorot_uniform"])
parser.add_argument('-ac','--cnn-conv-activation', dest="cnn_conv_activation", default="tanh", choices = ["tanh","relu"])
parser.add_argument('-ad','--cnn-dense-activation', dest="cnn_dense_activation", default="tanh", choices = ["tanh","relu"])
parser.add_argument('-bm','--border-mode', dest="border_mode", default="same", choices = ["valid","same"])
parser.add_argument('-s','--strides', dest="strides", nargs='+', default=[1,1], type=int)
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
    if not args.kernel_shape:
        args.kernel_shape = [[3, 3], [3, 3], [3, 3]]
    if not args.m_pool:
        args.m_pool = [[2, 2], [2, 2], [2, 2]]

#CNN stuff
tmp = ''.join(args.kernel_number)
cpk = tmp.split(',')
nCPK = [int(s) for s in cpk]

tmp = ''.join(args.dense_layers_inputs)
sls = tmp.split(',')
nLls = [int(s) for s in sls]

tmp = ''.join(args.kernel_shape)
cks = tmp.split(';')
nCKS=[]
for s in cks:
    ss = s.split(',')
    nCKS.append([int(shp) for shp in ss])

tmp = ''.join(args.m_pool)
pks = tmp.split(';')
nPKS = []
for s in pks:
    ss = s.split(',')
    nPKS.append([int(shp) for shp in ss])

args.kernel_number = nCPK
args.kernel_shape = nCKS
args.m_pool = nPKS
args.dense_layers_inputs = nLls

#TODO ADD ROOTDIR PARAM
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
RESULTS_DIR = 'CNN_class'
RESULTS_DIR = path.join(root_dir, RESULTS_DIR)
if not path.exists(RESULTS_DIR):
    makedirs(RESULTS_DIR)
experiments_db = path.join(root_dir,RESULTS_DIR,'experiments_db.csv')


date_key = datetime.strftime(datetime.now(), '%d-%m-%Y_time_%H:%M:%S')
fold_name = date_key

print ("Experiment started at:" + date_key)
####Update entry on experiments report file####

tag = arguments
tag = tag[1::2]
tag = reduce(lambda a, b: a + "\t" + b, tag).replace('--resume', '').replace('/', '-').replace('--', ';').replace('True', 'T').replace('False', 'F')
line = "\n" + fold_name + "\t" + path.basename(args.config_filename) + "\t" + tag + "\t"
with open(experiments_db, 'a+') as f:
    f.write(line)
EXPERIMENT_TAG = fold_name
### Create directories ###
FOLDER_PREFIX = path.join(root_dir, RESULTS_DIR,EXPERIMENT_TAG)
if not path.exists(FOLDER_PREFIX):
    makedirs(FOLDER_PREFIX)
MODEL_PATH = path.join(FOLDER_PREFIX, 'model')
if not path.exists(MODEL_PATH):
    makedirs(MODEL_PATH)
SCORES_PATH = path.join(FOLDER_PREFIX, 'scores')
if not path.exists(SCORES_PATH):
    makedirs(SCORES_PATH)


#GESTIONE DATASET
snoring_dataset = dm.load_ComParE2017(path.join(root_dir,'dataset', args.featureset)) #load dataset
labels = dm.label_loading(path.join(root_dir,'lab','ComParE2017_Snore.tsv'))

trainset, develset, testset = dm.split_ComParE2017_simple(snoring_dataset) #creo i trainset per calcolare media e varianza per poter normalizzare
trainset_l, develset_l, _ = dm.split_ComParE2017_simple(labels)

del snoring_dataset

trainset , mean, std = dm.normalize_data(trainset) #compute mean and std of the trainset and normalize the trainset

#normalize the dataset with the mean and std of the trainset
develset, _, _ = dm.normalize_data(develset, mean, std)
#testset, _, _ = dm.normalize_data(testset, mean, std)

#Find  matrix with biggest second axis
dim_pad = []
dim_pad.append(dm.dim_to_pad(trainset))
dim_pad.append(dm.dim_to_pad(develset))
#dim_pad.append(dm.dim_to_pad(testset))

dim_max = np.amax(dim_pad)

#Padding with white gaussian noise
trainset = dm.awgn_padding_set(trainset, dim_max)
develset = dm.awgn_padding_set(develset, dim_max)
#testset = dm.awgn_padding_set(testset, dim_max)


#Organize data to fed the network
x_train, y = dm.reshape_set(trainset)
x_devel, yd = dm.reshape_set(develset)
#x_test, yt = dm.reshape_set(testset)

y_train, y_train_lab = dm.label_organize(trainset_l,y)
y_devel, y_devel_lab = dm.label_organize(develset_l,yd)

input_shape = x_train.shape[1:]

print("------------------------EXPERIMENT STARTING---------------")

net=cnn.Cnn_Classifier([3,3], [16, 8, 8], input_shape, args.fit_net)
#net.define_arch()
net.define_parametric_arch(args)
#parametri di default anche per compile e fit
net.model_compile(args.optimizer, args.loss)
loss = net.model_fit(x_train, y_train, MODEL_PATH, args, x_devel, y_devel)
val_loss = np.asarray(loss[0].history['val_loss'])
train_loss = np.asarray(loss[0].history['loss'])
best_epoch = np.argmin(val_loss)
best_val_loss = val_loss[best_epoch]
best_loss = train_loss[best_epoch]

#TODO LOAD ONLY BEST MODEL
output = net.class_predictions(x_devel, MODEL_PATH)
out_net_filename = path.join(SCORES_PATH, 'network_preds.csv')
np.savetxt(out_net_filename, output, delimiter=';')

A, UAR, ConfMatrix, class_pred = net.compute_score(output, y_devel_lab)

print("Accuracy: " + str(A))
print("UAR: " + str(UAR))

line = "{}_{:.4f}_{:.4f}_{:.3f}_{:.3f}"
line = line.format(best_epoch,best_loss,best_val_loss,A,UAR)
line = line.replace('_', "\t")
with open(experiments_db, 'a+') as f:
    f.write(line)


#TODO: CREAZIONE ARFF come richesto da Challenge
#TODO: GESTIONE LOOP ESPERIMENTI E STORING RISULTATI
#TODO: CREAZIONE CONF FILES con DIVERSI PARAMETRI CNN