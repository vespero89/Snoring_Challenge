import sys
sys.path.append('..')
import numpy as np
import random
from sklearn.externals import joblib
import os
os.environ["KERAS_BACKEND"] = "theano"

from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.utils.np_utils import normalize
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.dataset_manupulation as dm
import utils.utils as utl
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

parser = argparse.ArgumentParser(description="ComParE2017 Snoring Classification")

# Global params
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-it", "--featureset", dest="featureset", default='SCAT')

# Layout params
parser.add_argument('-gs','--gen-shape', dest="gen_layers_shape",  nargs='+', default=[128,128])
parser.add_argument('-ag','--gen-activation', dest="gen_activation", default="tanh", choices = ["tanh","relu"])

parser.add_argument('-ds','--disc-shape', dest="disc_layers_shape",  nargs='+', default=[128,128])
parser.add_argument('-ad','--disc-activation', dest="disc_activation", default="tanh", choices = ["tanh","relu","LeakyReLU"])


parser.add_argument('-i','--init', dest="init", default="glorot_uniform", choices = ["glorot_uniform"])
parser.add_argument('-nb', '--no-bias', dest = "bias", default = True, action = 'store_false')
parser.add_argument('-p', '--end-pool', dest = "pool_only_to_end", default = False, action = 'store_true')
parser.add_argument("-drp", "--dropout", dest="dropout", default=False, action="store_true")
parser.add_argument("-drpr", "--drop-rate", dest="drop_rate", default=0.5, type=float)
parser.add_argument("-bn", "--batch-norm", dest="batch_norm", default=False, action="store_true")

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

#path setup
root_dir = os.path.realpath('/media/fabio/DATA/Work/Snoring/Snore_dist')
targePath = os.path.join(root_dir, 'gmmUbmSvm','snoring_class')
listPath = os.path.join(root_dir, 'dataset')
featPath = os.path.join(root_dir, 'dataset', featureset)

ubmsPath = os.path.join(targePath, featureset, "ubms")
supervecPath = os.path.join(targePath, featureset, "supervectors")
scoresPath = os.path.join(targePath, featureset, "score_best")
#snoreClassPath =os.path.join(targePath, featureset, "score_best","final_score_TEST_ER_t.csv")

#TODO FIX PATHs
#sys.stdout = open(os.path.join(scoresPath,'test.txt'), 'w')   #log to a file
#print "TEST: "+featureset; #to have the reference to experiments in text files
#sys.stderr = open(os.path.join(scoresPath,'test_err.txt'), 'w')   #log to a file

#LOAD DATASET
snoring_dataset = dm.load_ComParE2017(featPath, filetype)
trainset, develset, testset = dm.split_ComParE2017_simple(snoring_dataset)
labels = dm.label_loading(os.path.join(root_dir,'lab','ComParE2017_Snore.tsv'))
trainset_l, develset_l, _ = dm.split_ComParE2017_simple(labels)
del snoring_dataset

y = []
for seq in trainset:
    y.append(seq[0])

y_train, y_train_lab, _ = dm.label_organize(trainset_l, y)

V, O, T, E = dm.label_split(trainset_l, y)

nMixtures = joblib.load(os.path.join(scoresPath,'nmix2'));
Cs = joblib.load(os.path.join(scoresPath,'cBestValues2')); # Best
gammas =joblib.load(os.path.join(scoresPath,'gBestValues2')); # Best
Best_model = joblib.load(os.path.join(scoresPath,'best_model')); # Best
fold = 0;

print("Fold: " + str(fold));
C = Cs[fold];
gamma = gammas[fold];
BM = Best_model

print "Loading Features"
curSupervecPath = os.path.join(supervecPath, "trainset_" + str(fold), str(nMixtures));

V_feat = utl.readfeatures(curSupervecPath, V)
O_feat = utl.readfeatures(curSupervecPath, O)
T_feat = utl.readfeatures(curSupervecPath, T)
E_feat = utl.readfeatures(curSupervecPath, E)

X_t = np.concatenate((T_feat,E_feat),axis=0)
X_train = normalize(X_t)

input_shape = X_train.shape[1]
dropout_rate = 0.25
opt = Adam(lr=1e-4) #Generator optimizer
dopt = Adam(lr=1e-3) #Discriminator optimizer

# Build Generative model ...

g_input = Input(shape=(input_shape,))
x = g_input
for i in range(len(args.gen_layers_shape)):
    x = Dense(args.gen_layers_shape[i],
              init=args.init,
              activation=args.gen_activation,
              bias=args.bias)(x)
    print("dense[" + str(i) + "] -> (" + str(args.gen_layers_shape[i]) + ")")
    if args.dropout:
        x = Dropout(args.drop_rate)(x)
    if args.batch_norm:
        x = BatchNormalization(mode=1)(x)
g_V = Dense(input_shape, activation='sigmoid')(x)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

# Build Discriminative model ...
d_input = Input(shape=(input_shape,))
z = d_input
for i in range(len(args.disc_layers_shape)):
    z = Dense(args.disc_layers_shape[i],
              init=args.init,
              activation=args.disc_activation,
              bias=args.bias)(z)
    print("dense[" + str(i) + "] -> (" + str(args.disc_layers_shape[i]) + ")")
    if args.dropout:
        z = Dropout(args.drop_rate)(z)
    if args.batch_norm:
        z = BatchNormalization(mode=1)(z)
d_V = Dense(2, activation='softmax')(z)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=(input_shape,))
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


# def plot_loss(losses):
#     plt.figure(figsize=(10, 8))
#     plt.plot(losses["d"], label='discriminative loss')
#     plt.plot(losses["g"], label='generative loss')
#     plt.legend()
#     plt.show()

#TODO CREATE FUNCTION TO GENERATE SAMPLES OF DESIRED CLASS AFTER THE GAN TRAINING
# def plot_gen(n_ex=16, size=(4, 4)):
#     noise = np.random.uniform(0, 1, size=[n_ex, 100])
#     generated_images = generator.predict(noise)



ntrain = 25
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
XT = X_train[trainidx, :]

# Pre-train the discriminator network ...
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], input_shape])
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))
# Create binary target vector: first half random selected samples from trainset, second half samples generated from noise
n = XT.shape[0]
y = np.zeros([2 * n, 2])
y[:n, 1] = 1
y[n:, 0] = 1

make_trainable(discriminator, True)
discriminator.fit(X, y, nb_epoch=1, batch_size=1)
y_hat = discriminator.predict(X)

# Measure accuracy of pre-trained discriminator network
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx
n_tot = y.shape[0]
n_rig = (diff == 0).sum()
acc = n_rig * 100.0 / n_tot
print "Accuracy: %0.02f %% (%d of %d) vect right" % (acc, n_rig, n_tot)

# set up loss storage vector
losses = {"d": [], "g": []}
#TODO PRINT-SAVE in a file LOSSES

# Set up our main training loop
def train_for_n(nb_epoch=5000, BATCH_SIZE=1):

    for e in tqdm(range(nb_epoch)):

        # Make generative images
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, input_shape])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        # make_trainable(discriminator,True)
        d_loss = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, input_shape])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        # make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)



# # Train for 6000 epochs at original learning rates
train_for_n(nb_epoch=5000, BATCH_SIZE=1)
#
# # Train for 2000 epochs at reduced learning rates
# opt.lr.set_value(1e-5)
# dopt.lr.set_value(1e-4)
# train_for_n(nb_epoch=2000, BATCH_SIZE=1)
#
# # Train for 2000 epochs at reduced learning rates
# opt.lr.set_value(1e-6)
# dopt.lr.set_value(1e-5)
# train_for_n(nb_epoch=2000, BATCH_SIZE=1)
#
#
# # Plot some generated vectors from our GAN - TODO
#