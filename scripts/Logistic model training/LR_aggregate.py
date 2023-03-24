#!/usr/bin/env python

#System tools 
import pickle as pkl
import os,sys,csv,re

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
# from Modeling.gen_features import *


# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2

def onehotencoder(seq):
    '''
    Encodes a sequence into one-hot encoding.

    For each position there are 4 values, representing the 4 nucleotides. Only the nucleotide that is in that position
    gets value 1, the others are 1. There will be as many 1-values as there are nucleotides in the sequence.

    For each dinucleotide there are 16 values, representing 4*4=16 possible combinations of two nucleotides. The one
    in this position gets value 1, others are 0.
    '''
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode
# Load data
#threshold = sys.argv[1]

"""

''' Load the counts for all targets. '''
target_to_count = {}
with open("../../data_course/combi_totalfrequency.txt", 'r') as file:
    for line in file:
        target_to_count[line.split(",")[0]] = int(line.split(",")[1])
"""

"""
''' This part loads the training data. '''
workdir  = "C:/Users/celin/PycharmProjects/L1-Lindel/data/"
fname    = "Lindel_training.txt"

label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
feature_size = len(features) + 384 
data     = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)
Seqs = data[:,0]
data = data[:,1:].astype('float32')

# Sum up deletions and insertions to 
''' The first 'feature_size' values are inputs, all others are output. The Aggregate Model will output the average
frequency of each class, independently of the input. The frequencies are stored to a numpy array file. '''

X = data[:,:feature_size]
y = data[:, feature_size:]

'''
# Here we scale the frequencies by the number of counts for a target
y_scaled = []
for i in range(y.shape[0]):
    counts = target_to_count[Seqs[i]] if (Seqs[i] in target_to_count) else 0
    y_scaled.append(y[i,:]*counts)

y_scaled = np.array(y_scaled)
y_summed = np.sum(y_scaled, axis=0) # sum over all targets to find counts per class

agg_output_scaled = y_summed / np.sum(y_summed) #
np.save("agg_output_scaled", agg_output_scaled)
'''

agg_output = np.mean(y, axis=0)
np.save("agg_output", agg_output)

''' This part computes the MSE for the Aggregate Model.
We load the test data in "Lindel_test.txt". The first 'feature_size' values are inputs, all others are output. For each
of the input items in the data, we compute the MSE between the output and the output of the Aggregate Model. These
values are collected and visualized in a histogram. '''

workdir  = "C:/Users/celin/PycharmProjects/L1-Lindel/data/"
fname    = "Lindel_test.txt"

label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
feature_size = len(features) + 384
data     = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)
Seqs = data[:,0]
data = data[:,1:].astype('float32')

X = data[:,:feature_size]
y = data[:, feature_size:]
print('y.shape = ',y.shape)

agg_output = np.load("agg_output.npy")
MSE = np.zeros(y.shape[0])
for i in range(y.shape[0]):
    MSE[i] = mse(agg_output,y[i,:])

np.save("MSE_aggregate", MSE)
"""

''' This part makes a histogram of the MSEs. '''
MSE = np.load("MSE_aggregate.npy")
fig, ax = plt.subplots()
ax.hist(MSE*1000, bins=25, alpha=0.5)
ax.set_xlabel ("MSE (x10^-3)")
ax.set_ylabel("Counts")
ax.set_title("Performance of the Aggregate Model on the test set")
plt.show()
plt.savefig('../../Figures/Aggregate.eps', format='eps')
