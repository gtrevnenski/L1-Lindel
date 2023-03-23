#!/usr/bin/env python

#System tools 
import pickle as pkl
import os,sys,csv,re

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1


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
workdir  = sys.argv[1]
fname    = sys.argv[2]

label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
feature_size = len(features) + 384 
data     = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)
Seqs = data[:,0]
data = data[:,1:].astype('float32')

# Sum up deletions and insertions to
''' The first 'feature_size' values are inputs, all others are output.'''
X = data[:,:feature_size]
y = data[:, feature_size:]

''' The next few lines shuffle the data, to randomize which items are training and which ar validation data.
The size of train and validation data is defined. '''
np.random.seed(121)
idx = np.arange(len(y))
np.random.shuffle(idx)
X, y = X[idx], y[idx]
train_size = round(len(data) * 0.9) if 'ForeCasT' in fname else 3900
valid_size = round(len(data) * 0.1) if 'ForeCasT' in fname else 450 

''' Each sequence is encoded and stored in the array for either training or validation data.'''
Seq_train = Seqs[idx]
x_train,x_valid = [],[]
y_train,y_valid = [],[]
for i in range(train_size):
    x_train.append(onehotencoder(Seq_train[i]))
    y_train.append((sum(y[i][:-21]),sum(y[i][-21:])))
for i in range(train_size,len(Seq_train)):
    x_valid.append(onehotencoder(Seq_train[i]))
    y_valid.append((sum(y[i][:-21]),sum(y[i][-21:])))

x_train,x_valid = np.array(x_train),np.array(x_valid)
y_train,y_valid = np.array(y_train),np.array(y_valid)

# Train model
lambdas = 10 ** np.arange(-10, -1, 0.1)
errors_l1, errors_l2 = [], []
''' tqdm just shows a progress bar while it loops over values of lambda (penalty strength)
Train model with both L1 and L2 regularization and compute the errors.'''
for l in tqdm(lambdas):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(2,  activation='softmax', input_shape=(384,), kernel_regularizer=l2(l)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
              callbacks=[EarlyStopping(patience=1)], verbose=0)
    y_hat = model.predict(x_valid)
    errors_l2.append(mse(y_hat, y_valid))

for l in tqdm(lambdas):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(2,  activation='softmax', input_shape=(384,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
              callbacks=[EarlyStopping(patience=1)], verbose=0)
    y_hat = model.predict(x_valid)
    errors_l1.append(mse(y_hat, y_valid))


np.save(workdir+'mse_l1_indel.npy',errors_l1)
np.save(workdir+'mse_l2_indel.npy',errors_l2)

# final model
'''Choose the lambda for L1 and L2 separately that gives the smallest error.'''
l = lambdas[np.argmin(errors_l1)]
np.random.seed(0)
model = Sequential()
model.add(Dense(2, activation='softmax', input_shape=(384,), kernel_regularizer=l1(l)))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save(workdir+'L1_indel.h5')


l = lambdas[np.argmin(errors_l2)]
np.random.seed(0)
model = Sequential()
model.add(Dense(2, activation='softmax', input_shape=(384,), kernel_regularizer=l2(l)))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save(workdir+'L2_indel.h5')
