#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np
import hashlib
import fnmatch

np.random.seed(888)#for experiment repetibility
import CNN_classifier as cnn
import BIN_classifier as bin
import dataset_manupulation as dm
from os import path, makedirs, listdir
import argparse
from operator import itemgetter
from datetime import datetime

parser = argparse.ArgumentParser(description="ComParE2017 Snoring Classification")

# Global params
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-it", "--featureset", dest="featureset", default='logmel')
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

# fit params (CNN)
parser.add_argument("-e", "--epoch", dest = "epoch", default=100, type=int)
parser.add_argument("-ns", "--no-shuffle", dest = "shuffle", default = True, action = 'store_false')
parser.add_argument("-bs", "--batch-size", dest = "batch_size", default=16, type=int)
parser.add_argument("-vs", "--validation-split", dest = "valid_split", default=0.1, type=float)
parser.add_argument("-f", "--fit-net", dest = "fit_net", default = True, action = 'store_true')
parser.add_argument("-o", "--optimizer", dest = "optimizer", default="adam", choices = ["adadelta","adam", "sgd"])
parser.add_argument("-lr", "--learning-rate", dest = "learning_rate", default=0.1, type=float)
parser.add_argument("-l", "--loss", dest = "loss", default="categorical_crossentropy", choices = ["categorical_crossentropy"])

# BIN params
parser.add_argument('-a_bin','--bin-activation', dest="bin_activation", default="sigmoid", choices = ["sigmoid","hard_sigmoid"])
parser.add_argument("-thr", "--threshold", dest = "threshold", default=0.38, type=float)

# fit params (BIN)
parser.add_argument("-e_bin", "--epoch-bin", dest = "epoch_bin", default=100, type=int)
parser.add_argument("-ns_bin", "--no-shuffle-bin", dest = "shuffle_bin", default = True, action = 'store_false')
parser.add_argument("-bs_bin", "--batch-size-bin", dest = "batch_size_bin", default=32, type=int)
parser.add_argument("-vs_bin", "--validation-split-bin", dest = "valid_split_bin", default=0.1, type=float)
parser.add_argument("-f_bin", "--fit-net-bin", dest = "fit_net_bin", default = True, action = 'store_true')
parser.add_argument("-o_bin", "--optimizer-bin", dest = "optimizer_bin", default="rmsprop", choices = ["rmsprop","sgd","adadelta","adam"])
parser.add_argument("-l_bin", "--loss-bin", dest = "loss_bin", default="binary_crossentropy", choices = ["binary_crossentropy"])


###############################################################################


args = parser.parse_args()

if (args.config_filename is not None):
    with open(args.config_filename, 'r') as f:
    #with open(path_config, 'r') as f:
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
else:
    args.config_filename = 'test'

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
root_dir = path.realpath('../../')
RESULTS_DIR = 'CNN_Snoring_hierarchic/experiments'
RESULTS_DIR = path.join(root_dir, RESULTS_DIR)
if not path.exists(RESULTS_DIR):
    makedirs(RESULTS_DIR)

experiments_db = path.join(root_dir,RESULTS_DIR,'experiments_db.csv')
experiments_db_bin = path.join(root_dir,RESULTS_DIR,'experiments_db_bin.csv')
experiments_db_cnn = path.join(root_dir,RESULTS_DIR,'experiments_db_cnn.csv')
experiments_db_cnn_test = path.join(root_dir,RESULTS_DIR,'experiments_db_cnn_test.csv')

date_key = datetime.strftime(datetime.now(), '%d-%m-%Y_time_%H:%M:%S')
fold_name = date_key

print ("Experiment started at:" + date_key)
#config_name = 'k2,2,2,2_p3,3,3,3'
####Update entry on experiments report file####

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

trainset, develset, _ = dm.split_ComParE2017_simple(snoring_dataset) #creo i trainset per calcolare media e varianza per poter normalizzare
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

_, y_devel_lab = dm.label_organize(develset_l,yd)

#Organize data for BIN CLASSIFICATOR
x_train_bin = x_train
y_train_bin = dm.data_bin_organize(trainset_l,y)
y_devel_bin = dm.data_bin_organize(develset_l,yd)

#Organize data for CNN CLASSIFICATOR
x_train_cnn, y_train_cnn, _ = dm.data_cnn_organize(trainset_l, y, trainset)
x_train_cnn, _ = dm.reshape_set(x_train_cnn)

#Questi devel sono utili SOLO per valutare il buon training (o meno) della CNN
x_devel_cnn, _, y_devel_cnn_lab = dm.data_cnn_organize(develset_l, yd, develset)
x_devel_cnn, _ = dm.reshape_set(x_devel_cnn)


print("------------------------EXPERIMENT STARTING---------------")


# BINARY CLASSIFIER
line = ("\n" + fold_name + "\t" + path.basename(args.config_filename) + "\t" + args.bin_activation + "\t" +
        str(args.threshold) + "\t" + args.optimizer_bin + "\t" + str(args.batch_size_bin) + "\t")
with open(experiments_db_bin, 'a+') as f:
    f.write(line)

input_shape = x_train_bin.shape[1:]
net1 = bin.Bin_Classifier(input_shape, args.fit_net_bin)
net1.define_parametric_arch(args)
net1.model_compile(args.optimizer_bin, args.loss_bin)
loss = net1.model_fit(x_train_bin, y_train_bin, MODEL_PATH, args)
val_loss = np.asarray(loss[0].history['val_loss'])
train_loss = np.asarray(loss[0].history['loss'])
best_epoch = np.argmin(val_loss)
best_val_loss = val_loss[best_epoch]
best_loss = train_loss[best_epoch]

#TODO LOAD ONLY BEST MODEL (BIN)
output = net1.class_predictions(x_devel, MODEL_PATH)
out_net_filename = path.join(SCORES_PATH, 'network_preds_bin.csv')
np.savetxt(out_net_filename, output, delimiter=';')

A, UAR, ConfMatrix, class_pred_bin = net1.compute_score(output, y_devel_bin, args.threshold)

print("Accuracy (Bin): " + str(A))
print("UAR (Bin): " + str(UAR))

line = "{}_{:.4f}_{:.4f}_{:.3f}_{:.3f}"
line = line.format(best_epoch,best_loss,best_val_loss,A,UAR)
line = line.replace('_', "\t")
with open(experiments_db_bin, 'a+') as f:
    f.write(line)

# CNN CLASSIFIER
line = ("\n" + fold_name + "\t" + path.basename(args.config_filename) + "\t" + str(args.kernel_number) + "\t" +
        str(args.kernel_shape) + "\t" + str(args.m_pool) + "\t" + str(args.dense_layers_inputs) + "\t" +
        args.cnn_conv_activation + "\t" + args.cnn_dense_activation + "\t" + str(args.batch_size) + "\t" +
        args.optimizer + "\t").replace('[','').replace(']','').replace(' ','')
with open(experiments_db_cnn, 'a+') as f:
    f.write(line)
with open(experiments_db_cnn_test, 'a+') as f:
    f.write(line)

input_shape = x_train_cnn.shape[1:]
net2=cnn.Cnn_Classifier([3,3], [16, 8, 8], input_shape, args.fit_net)
#net.define_arch()
net2.define_parametric_arch(args)
#parametri di default anche per compile e fit
net2.model_compile(args.optimizer, args.learning_rate, args.loss)
loss = net2.model_fit(x_train_cnn, y_train_cnn, MODEL_PATH, args)
val_loss = np.asarray(loss[0].history['val_loss'])
train_loss = np.asarray(loss[0].history['loss'])
best_epoch = np.argmin(val_loss)
best_val_loss = val_loss[best_epoch]
best_loss = train_loss[best_epoch]

line = "{}_{:.4f}_{:.4f}"
line = line.format(best_epoch,best_loss,best_val_loss)
line = line.replace('_', "\t")
with open(experiments_db_cnn, 'a+') as f:
    f.write(line)


# ------------------------------------------------------------------------------------
# WARNING: queste predizioni servono solo a vedere quanto è ben
# allenata (o meno) la rete CNN. Le vere predizioni vanno fatte con il
# devel_set in uscita dal classificatore binario!
# (commentare questo blocco quando valuto la rete globale)
#output = net2.class_predictions(x_devel_cnn, MODEL_PATH)
#
#A, UAR, ConfMatrix, class_pred_cnn = net2.compute_score(output, y_devel_cnn_lab)
#
#print("Accuracy (CNN Test "OTE"): " + str(A))
#print("UAR (CNN Test "OTE"): " + str(UAR))
#
#line = "{}_{:.4f}_{:.4f}_{:.3f}_{:.3f}"
#line = line.format(best_epoch,best_loss,best_val_loss,A,UAR)
#line = line.replace('_', "\t")
#with open(experiments_db_cnn_test, 'a+') as f:
#    f.write(line)
# ------------------------------------------------------------------------------------


# BINARY + CNN
tag = arguments
tag = tag[1::2]
tag = reduce(lambda a, b: a + "\t" + b, tag).replace('--resume', '').replace('/', '-').replace('--', ';').replace('True', 'T').replace('False', 'F')
line = "\n" + fold_name + "\t" + path.basename(args.config_filename) + "\t" + tag + "\t"
with open(experiments_db, 'a+') as f:
    f.write(line)

devel_cnn_reduced = develset[:]
i = 0
for c in class_pred_bin:
    if c == 0:
        del devel_cnn_reduced[i]
        continue
    else:
        i += 1

x_devel_cnn, _ = dm.reshape_set(devel_cnn_reduced)

output = net2.class_predictions(x_devel_cnn, MODEL_PATH)
out_net_filename = path.join(SCORES_PATH, 'network_preds_cnn.csv')
np.savetxt(out_net_filename, output, delimiter=';')

output_global = []
i = 0
for c in class_pred_bin:
    if c == 0:
        output_global.append([1,0,0,0])
    else:
        output_global.append([0,output[i][0],output[i][1],output[i][2]])
        i += 1
output_global = np.asarray(output_global)
out_net_filename = path.join(SCORES_PATH, 'network_preds_global.csv')
np.savetxt(out_net_filename, output_global, delimiter=';')

A, UAR, ConfMatrix, class_pred = net2.compute_score_global(output_global, y_devel_lab)

print("Accuracy (Global): " + str(A))
print("UAR (Global): " + str(UAR))

line = "{:.3f}_{:.3f}"
line = line.format(A,UAR)
line = line.replace('_', "\t")
with open(experiments_db, 'a+') as f:
    f.write(line)