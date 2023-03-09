import pandas as pd
import pickle as pkl
import numpy as np

# Tab delimited files which consist of training and test samples. 
# First column is the guide sequence, the next 3033 columns are the features(2649 MH binary + 384 one-hot encoded features) 
# and the last 557 columns are the observed outcome(class) frequencies.

Lindel_training = pd.read_csv("data_course/Lindel_training.txt", sep='\t')

label, rev_index, features = pkl.load(open('data_course/feature_index_all.pkl','rb'))


matrix1 = pkl.load(open('data_course/NHEJ_rep1_final_matrix.pkl','rb'))
matrix2 = pkl.load(open('data_course/NHEJ_rep2_final_matrix.pkl','rb'))
matrix3 = pkl.load(open('data_course/NHEJ_rep3_final_matrix.pkl','rb'))

### FIGURE 2B - VIOLIN PLOT PROCESSING ###

# idea is to get an array of allele frequencies per target and use that with https://realpython.com/numpy-scipy-pandas-correlation-python/
# to calculate the correlation between two target sequences. What if we observe non-TW alleles that are present in 1 replicate but not in the other?

frequency_repl1 = {}
frequency_repl2 = {}
frequency_repl3 = {}

# initialize frequency array
for i in range(len(matrix1)):
    frequency_repl1[matrix1[i][2]] = {}

for i in range(len(matrix2)):
    frequency_repl2[matrix2[i][2]] = {}

for i in range(len(matrix3)):
    frequency_repl3[matrix3[i][2]] = {}

# count frequencies of different alleles per target
for i in range(len(matrix1)):
    if matrix1[i][2] != matrix1[i][3]:
        if matrix1[i][3] not in frequency_repl1[matrix1[i][2]]:
            frequency_repl1[matrix1[i][2]][matrix1[i][3]] = 1
        else:
            frequency_repl1[matrix1[i][2]][matrix1[i][3]] += 1

for i in range(len(matrix2)):
    if matrix2[i][2] != matrix2[i][3]:
        if matrix2[i][3] not in frequency_repl2[matrix2[i][2]]:
            frequency_repl2[matrix2[i][2]][matrix2[i][3]] = 1
        else:
            frequency_repl2[matrix2[i][2]][matrix2[i][3]] += 1

for i in range(len(matrix3)):
    if matrix3[i][2] != matrix3[i][3]:
        if matrix3[i][3] not in frequency_repl3[matrix3[i][2]]:
            frequency_repl3[matrix3[i][2]][matrix3[i][3]] = 1
        else:
            frequency_repl3[matrix3[i][2]][matrix3[i][3]] += 1

with open("data_course/repl1_frequency_data.txt", 'w') as file:
    for key in frequency_repl1:
        file.write(key + ": " + str(frequency_repl1[key]) + "\n")

with open("data_course/repl2_frequency_data.txt", 'w') as file:
    for key in frequency_repl2:
        file.write(key + ": " + str(frequency_repl2[key]) + "\n")

with open("data_course/repl3_frequency_data.txt", 'w') as file:
    for key in frequency_repl3:
        file.write(key + ": " + str(frequency_repl3[key]) + "\n")