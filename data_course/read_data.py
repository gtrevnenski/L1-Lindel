import pandas as pd
import pickle as pkl
import numpy as np

# Tab delimited files which consist of training and test samples. 
# First column is the guide sequence, the next 3033 columns are the features(2649 MH binary + 384 one-hot encoded features) 
# and the last 557 columns are the observed outcome(class) frequencies.

Lindel_training = pd.read_csv("Lindel_training.txt", sep='\t')

label, rev_index, features = pkl.load(open('feature_index_all.pkl','rb'))


matrix1 = pkl.load(open('NHEJ_rep1_final_matrix.pkl','rb'))
matrix2 = pkl.load(open('NHEJ_rep2_final_matrix.pkl','rb'))
matrix3 = pkl.load(open('NHEJ_rep3_final_matrix.pkl','rb'))

### FIGURE 2B - VIOLIN PLOT PROCESSING ###

# idea is to get an array of allele frequencies per target and use that with https://realpython.com/numpy-scipy-pandas-correlation-python/
# to calculate the correlation between two target sequences. What if we observe non-TW alleles that are present in 1 replicate but not in the other?

frequency_repl1 = {}
frequency_repl2 = {}
frequency_repl3 = {}

frequency_comb = {}
# initialize frequency array
for i in range(len(matrix1)):
    frequency_repl1[matrix1[i][2]] = {}
    frequency_comb[matrix1[i][2]] = {}

for i in range(len(matrix2)):
    frequency_repl2[matrix2[i][2]] = {}
    frequency_comb[matrix2[i][2]] = {}

for i in range(len(matrix3)):
    frequency_repl3[matrix3[i][2]] = {}
    frequency_comb[matrix3[i][2]] = {}

# count frequencies of different alleles per target
for i in range(len(matrix1)):
    if matrix1[i][2] != matrix1[i][3]:
        if matrix1[i][3] not in frequency_repl1[matrix1[i][2]]:
            frequency_repl1[matrix1[i][2]][matrix1[i][3]] = 1
        else:
            frequency_repl1[matrix1[i][2]][matrix1[i][3]] += 1

        if matrix1[i][3] not in frequency_comb[matrix1[i][2]]:
            frequency_comb[matrix1[i][2]][matrix1[i][3]] = [1, 0, 0]
        else:
            frequency_comb[matrix1[i][2]][matrix1[i][3]][0] += 1

for i in range(len(matrix2)):
    if matrix2[i][2] != matrix2[i][3]:
        if matrix2[i][3] not in frequency_repl2[matrix2[i][2]]:
            frequency_repl2[matrix2[i][2]][matrix2[i][3]] = 1
        else:
            frequency_repl2[matrix2[i][2]][matrix2[i][3]] += 1

        if matrix2[i][3] not in frequency_comb[matrix2[i][2]]:
            frequency_comb[matrix2[i][2]][matrix2[i][3]] = [0, 1, 0]
        else:
            frequency_comb[matrix2[i][2]][matrix2[i][3]][1] += 1

for i in range(len(matrix3)):
    if matrix3[i][2] != matrix3[i][3]:
        if matrix3[i][3] not in frequency_repl3[matrix3[i][2]]:
            frequency_repl3[matrix3[i][2]][matrix3[i][3]] = 1
        else:
            frequency_repl3[matrix3[i][2]][matrix3[i][3]] += 1

        if matrix3[i][3] not in frequency_comb[matrix3[i][2]]:
            frequency_comb[matrix3[i][2]][matrix3[i][3]] = [0, 0, 1]
        else:
            frequency_comb[matrix3[i][2]][matrix3[i][3]][2] += 1


with open("combi_frequency_data.txt", 'w') as file:
    for key in frequency_comb:
        file.write(key + ": " + str(frequency_comb[key]) + "\n")

with open("combi_frequency_arr2.txt", 'w') as file:
    for target, outcomes in frequency_comb.items():
        for i in range(3):
            for freq in outcomes.values():
                file.write(str(freq[i]) + ",")
            file.write("\n")
        file.write("---\n")

