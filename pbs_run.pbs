#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -W x=GRES:gpu@1

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

source ~/virtualenvs/snoring/bin/activate

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python /media/a3lab/Data/Projects/Snore_dist/CNN_snoring/main_experiment_SNORING.py --config-file /media/a3lab/Data/Projects/Snore_dist/CNN_snoring/$CFGNAME
