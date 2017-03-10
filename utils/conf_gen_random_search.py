import numpy as np
import scipy
import scipy.stats
from os import path, makedirs
from tqdm import trange
from shutil import copyfile
from random import shuffle

from custom_distributions import *

root_dir = path.realpath('../')
CONF_DIR = 'CNN_snoring/Configs_RandomSearch'
CONF_DIR = path.join(root_dir, CONF_DIR)
if not path.exists(CONF_DIR):
    makedirs(CONF_DIR)

# Global ranges
nConfigurations = 1000

# Discriminator ranges

cnn_n_layers = [1, 2, 3]
dense_n_layers = [1, 2, 3, 4]

kern_n_low = np.log2(4)
kern_n_high = np.log2(32)

dense_n_units_low = np.log2(64)
dense_n_units_high = np.log2(2048)

kernel_shape = ['1,1','3,3','5,5']

activation = ['relu','tanh']  # Keras basic activations or LeakyReLU

batch_size_low =  np.log2(32)
batch_size_high = np.log2(128)



# params  =               {'
#                         'n_layers': scipy.stats.randint(low=dis_n_layers_low, high=dis_n_layers_high + 1),
#                         'n_units': loguniform_gen(low=dis_n_units_low, high=dis_n_units_high, round_exponent=False,
#                                                   round_output=True),
#                         'lr': loguniform_gen(base=10, low=lr_low, high=lr_high, round_exponent=False,
#                                              round_output=False)
#                         }


for i in trange(nConfigurations):
    configuration_name = str(i).zfill(4) + '.cfg'

    k_num = choices(cnn_n_layers).rvs()
    k_shape = []
    pool_shape = []
    kern_n = []
    for i in (xrange(k_num)):
        kn = loguniform_gen(low=kern_n_low, high=kern_n_high, round_exponent=False, round_output=True).rvs()
        ks = choices(kernel_shape).rvs()
        ps = choices(kernel_shape).rvs()
        kern_n.append(str(int(kn)))
        k_shape.append(ks)
        pool_shape.append(ps)


    d_num = choices(dense_n_layers).rvs()
    d_unit = []
    for j in (xrange(d_num)):
        du = loguniform_gen(low=dense_n_units_low, high=dense_n_units_high, round_exponent=False, round_output=True).rvs()
        d_unit.append(str(int(du)))

    cnn_activ = choices(activation).rvs()
    dense_activ = choices(activation).rvs()

    batch_sz = int(loguniform_gen(low=batch_size_low, high=batch_size_high, round_exponent=False, round_output=True).rvs())

    line = "--kernels-number\t" + ','.join(kern_n) + "\n"\
        + "--kernel-shape\t" + ';'.join(k_shape) + "\n"\
        + "--max-pool-shape\t" + ';'.join(pool_shape) + "\n"\
        + "--dense-shape\t" + ','.join(d_unit) + "\n"\
        + "--cnn-conv-activation\t" + cnn_activ+ "\n"\
        + "--cnn-dense-activation\t" + dense_activ + "\n"\
        + "--batch-size\t" + str(batch_sz)

    with open(path.join(CONF_DIR,configuration_name), 'w+') as f:
        f.write(line)
