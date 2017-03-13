#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:36:52 2017

@author: buckler
"""
import os
import numpy as np
import csv
from os import path
import keras.utils.np_utils as nputils
import htkutils

def load_ComParE2017(featuresPath, filetype):
    '''
    Carica tutto il dataset (spettri) in una lista di elementi [filename , matrix ]
    '''
    print("Loading ComParE2017_Snoring dataset");
    snoring=list()
    for root, dirnames, filenames in os.walk(featuresPath):
        i=0;
        if (filetype == 'csv'):
            for file in filenames:
                num_lines = sum(1 for line in open(os.path.join(featuresPath, file)))
                with open(os.path.join(featuresPath, file), 'r') as f:
                    line = f.readline()
                    line = line.rstrip('\n')
                    line = line.split(',')
                    line = [float(s) for s in line]
                    matrix = np.zeros((num_lines, len(line)))
                    startIndex = 0
                    for line in f:
                        line = line.rstrip('\n')
                        line = line.split(',')
                        line = [float(s) for s in line]
                        matrix[startIndex, :] = line
                        startIndex += 1
                # matrix = np.asarray(matrix)
                data = [file, matrix]
                snoring.append(data)
                i += 1
        elif (filetype == 'npy'):
            for file in filenames:
                matrix=np.load(os.path.join(root,file))
                if matrix.ndim != 2:
                    shape = matrix.shape[1:]
                    matrix = matrix.reshape(shape)
                data=[file,matrix]
                snoring.append(data)
                i+=1
        elif (filetype == 'htk'):
            for file in filenames:
                matrix=htkutils.readhtk(os.path.join(root,file))
                if matrix.ndim != 2:
                    shape = matrix.shape[1:]
                    matrix = matrix.reshape(shape)
                data=[file,matrix]
                snoring.append(data)
                i+=1
    return snoring


def awgn_padding_set( set_to_pad, dim_pad, loc=0.0, scale=1.0):
    print("awgn_padding_set")
    #dim_pad=np.amax([len(k[1][2]) for k in set_to_pad])
    awgn_padded_set = []
    for e in set_to_pad:
        row, col = e[1].shape;
        # crete an rowXcol matrix with awgn samples
        awgn_matrix = np.random.normal(loc, scale, size=(row,dim_pad-col));
        awgn_padded_set.append([e[0],np.hstack((e[1],awgn_matrix))]);
    return awgn_padded_set 

def reshape_set(set_to_reshape, channels=1):
    '''
    '''
    print("reshape_set")
    n_sample=len(set_to_reshape);
    row, col = set_to_reshape[0][1].shape;
    label = []
    shaped_matrix = np.empty((n_sample,channels,row,col));
    for i in range(len(set_to_reshape)):
        label.append(set_to_reshape[i][0]);
        shaped_matrix[i][0]=set_to_reshape[i][1]
    return shaped_matrix,  label

def split_ComParE2017_simple(data,train_tag=None, devel_tag=None, test_tag=None):
    '''
    Splitta il dataset in train, devel e test set
    (da amplicare in modo che consenta lo split per la validation)
    '''
    print("split_ComParE2017simple")

    if train_tag==None:
        train_tag=['train']
    if test_tag==None:
        test_tag=['test']
    if devel_tag==None:
        devel_tag = ['devel']
    
    data_train=[d for d in data if any(word in d[0] for word in train_tag)] #controlla se uno dei tag è presente nnel nome del file e lo assegna al trainset
    data_test = [d for d in data if any(word in d[0] for word in test_tag)]
    data_devel = [d for d in data if any(word in d[0] for word in devel_tag)]
    #data_test=[d for d in data if d not in data_train]#tutto cioò che non è train diventa test
    
    return data_train, data_devel, data_test

def split_ComParE2017_from_lists(data, listpath, namelist):
    '''
    Richede in inglesso la cartella dove risiedono i file di testo che elencano i vari segnali che farano parte di un voluto set di dati.
    Inltre in namelist vanno specificati i nomi dei file di testo da usare.
    Ritorna una lista contentete le liste dei dataset di shape: (len(namelist),data.shape)
    '''
    print("split_ComParE2017_from_lists")

    sets=list();
    for name in namelist:
        sets.append(select_list(os.path.join(listpath,name),data));
    return sets
    
def select_list(filename,dataset):
    '''
    Dato in ingesso un file di testo, resituisce una array contenete i dati corrispondenti elencati nel file
    '''
    print("select_list")

    subset=list()
    with open(filename) as f:
        content = f.readlines();
        content = [x.strip().replace('.wav','.npy') for x in content] #remove the '\n' at the end of string
        subset = [s for s in dataset if any(name in s[0] for name in content)] #select all the data presetn in the list
    return subset        
    
def normalize_data(data,mean=None,std=None):
    '''
    normalizza media e varianza del dataset passato
    se data=None viene normalizzato tutto il dataset A3FALL
    se mean e variance = None essi vengono calcolati in place sui data
    '''
    print("normalize_data")

    if bool(mean) ^ bool(std):#xor operator
        raise("Error!!! Provide both mean and variance")
    elif mean==None and std==None: #compute mean and variance of the passed data
        data_conc= concatenate_matrix(data)
        mean=np.mean(data_conc)
        std=np.std(data_conc)   
                                        
    data_std= [[d[0],((d[1]-mean)/std)] for d in data]#normalizza i dati: togle mean e divide per std
    
    return data_std, mean , std
    
def concatenate_matrix(data):
    '''
    concatena gli spettri in un unica matrice: vule una lista e restituisce un array
    '''
    print("concatenate_matrix")
    shapes=[]
    data_= data[:]
    #data_.pop(0) ??
    matrix=data[0][1]
    for d in data_:
        np.append(matrix,d[1], axis=1)
    return matrix

def label_loading(label_file=None):
    filenames = []
    labels = []
    with open(label_file, 'r') as csvfile:
        labfile = csv.reader(csvfile, delimiter='\t')
        rownum = 0
        for row in labfile:
            if rownum == 0:
                header = row
            else:
                #filename = row[0]
                #label = row[1]

                #filenames.append(filename)
                labels.append(row)
            rownum += 1

        return labels

def label_organize(label_set,namelist):
    y_set = []
    for i in range(len(namelist)):
        a = path.splitext(namelist[i])[0]
        for j in range(len(label_set)):
            b = path.splitext(label_set[j][0])[0]
            if a == b:
                y_set.append(label_set[j][1])

    y_set_int = []
    for k in y_set:
        if k == 'V':
            i = 0
        elif k == 'O':
            i = 1
        elif k == 'T':
            i = 2
        elif k == 'E':
            i = 3
        y_set_int = np.append(y_set_int, i)

    y_set_cat= nputils.to_categorical(y_set_int,nb_classes=4)
    return y_set_cat, y_set_int

def dim_to_pad(set_to_pad):
    dim_pad = np.amax([len(k[1][2]) for k in set_to_pad])
    return dim_pad

def data_bin_organize(label_set,namelist):
    y_set = []
    for i in range(len(namelist)):
        a = path.splitext(namelist[i])[0]
        for j in range(len(label_set)):
            b = path.splitext(label_set[j][0])[0]
            if a == b:
                y_set.append(label_set[j][1])

    y_set_int = []
    for k in y_set:
        if k == 'V':
            i = 0
        else:
            i = 1
        y_set_int = np.append(y_set_int, i)

    return y_set_int

def data_class_organize(label_set,namelist,data_set):
    y_set = []
    for i in range(len(namelist)):
        a = path.splitext(namelist[i])[0]
        for j in range(len(label_set)):
            b = path.splitext(label_set[j][0])[0]
            if a == b:
                y_set.append(label_set[j][1])

    y_set_int = []
    x_set = data_set[:]
    j=0
    k_to_remove = []
    for k in range(0,len(y_set)):
        if y_set[k] == 'V':
            k_to_remove.append(k)
        elif y_set[k] == 'O':
            i = 0
            y_set_int.append(i)
        elif y_set[k] == 'T':
            i = 1
            y_set_int.append(i)
        elif y_set[k] == 'E':
            i = 2
            y_set_int.append(i)
        j+=1

    x_set = np.delete(x_set, k_to_remove, axis=0)
    y_set_cat = nputils.to_categorical(y_set_int,nb_classes=3)
    return x_set, y_set_cat, y_set_int