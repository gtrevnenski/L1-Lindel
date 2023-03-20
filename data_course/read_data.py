import pandas as pd
import pickle as pkl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# # Tab delimited files which consist of training and test samples. 
# # First column is the guide sequence, the next 3033 columns are the features(2649 MH binary + 384 one-hot encoded features) 
# # and the last 557 columns are the observed outcome(class) frequencies.

# # Lindel_training = pd.read_csv("data_course/Lindel_training.txt", sep='\t')

# # label, rev_index, features = pkl.load(open('feature_index_all.pkl','rb'))


matrix1 = pkl.load(open('NHEJ_rep1_final_matrix.pkl','rb'))
matrix2 = pkl.load(open('NHEJ_rep2_final_matrix.pkl','rb'))
matrix3 = pkl.load(open('NHEJ_rep3_final_matrix.pkl','rb'))

# types_of_mutations = set()
# for i in range(len(matrix1)):
#     types_of_mutations.add(matrix1[i][9])

# print(types_of_mutations) # you will see that there's a mutation called 'complex' which most probably represents a combination of substitutions, insertions, and deletions that they do not take into account for their analysis.

### FIGURE 2B - VIOLIN PLOT PROCESSING ###

# idea is to get an array of allele frequencies per target and use that with https://realpython.com/numpy-scipy-pandas-correlation-python/
# to calculate the correlation between two target sequences. What if we observe non-TW alleles that are present in 1 replicate but not in the other?

frequency_comb = {}
# initialize frequency array 
for i in range(len(matrix1)):
    frequency_comb[matrix1[i][2]] = {}

for i in range(len(matrix2)):
    frequency_comb[matrix2[i][2]] = {}

for i in range(len(matrix3)):
    frequency_comb[matrix3[i][2]] = {}

# count frequencies of different alleles per target
for i in range(len(matrix1)):
    if matrix1[i][2] != matrix1[i][3] and matrix1[i][9] != 'complex':
        if matrix1[i][3] not in frequency_comb[matrix1[i][2]]:
            frequency_comb[matrix1[i][2]][matrix1[i][3]] = [1, 0, 0]
        else:
            frequency_comb[matrix1[i][2]][matrix1[i][3]][0] += 1

for i in range(len(matrix2)):
    if matrix2[i][2] != matrix2[i][3] and matrix2[i][9] != 'complex':
        if matrix2[i][3] not in frequency_comb[matrix2[i][2]]:
            frequency_comb[matrix2[i][2]][matrix2[i][3]] = [0, 1, 0]
        else:
            frequency_comb[matrix2[i][2]][matrix2[i][3]][1] += 1

for i in range(len(matrix3)):
    if matrix3[i][2] != matrix3[i][3] and matrix3[i][9] != 'complex':
        if matrix3[i][3] not in frequency_comb[matrix3[i][2]]:
            frequency_comb[matrix3[i][2]][matrix3[i][3]] = [0, 0, 1]
        else:
            frequency_comb[matrix3[i][2]][matrix3[i][3]][2] += 1

''' Save for each target the total number of outcomes observed, in order to find the Aggregate model. We sum over the 
three replicates, as indicated in the caption of figure 2a in the figure. '''
with open("combi_totalfrequency2.txt", 'w') as file:
    for target, outcomes in frequency_comb.items():
        counts = 0
        for freq in outcomes.values():
            counts += sum(freq)
        file.write(target+","+str(counts)+"\n")


# Remove outcomes with less than 10 UMIs.
for target, outcome_dicts in frequency_comb.items():
    to_remove = []
    for outcome, freqs in outcome_dicts.items():
        if sum(freqs) < 10:
            to_remove.append(outcome)
    for item in to_remove:
        outcome_dicts.pop(item)

# Remove targets with less than 3 outcomes
to_remove = []
for target, outcome_dicts in frequency_comb.items():
    if len(outcome_dicts) < 3:
        to_remove.append(target)
for item in to_remove:
    frequency_comb.pop(item)

with open("combi_frequency_data.txt", 'w') as file:
    for key in frequency_comb:
        file.write(key + ": " + str(frequency_comb[key]) + "\n")

with open("combi_frequency_arr3.txt", 'w') as file:
    for target, outcomes in frequency_comb.items():
        if len(outcomes.values()) > 0:
            for i in range(3):
                for freq in outcomes.values():
                    file.write(str(freq[i]) + ",")
                file.write("\n")
            file.write("---\n")


repl_1_2_correlation_coefficients = []
repl_2_3_correlation_coefficients = []
repl_1_3_correlation_coefficients = [] 

repl_1Normal_2Permuted_correlation_coefficients = []
repl_1Normal_3Permuted_correlation_coefficients = []
repl_2Normal_1Permuted_correlation_coefficients = []
repl_2Normal_3Permuted_correlation_coefficients = [] 
repl_3Normal_1Permuted_correlation_coefficients = []
repl_3Normal_2Permuted_correlation_coefficients = []

for target in frequency_comb:
    frequency_repl1 = []
    frequency_repl2 = []
    frequency_repl3 = []
    
    for outcome in frequency_comb[target]:
        frequency_repl1.append(frequency_comb[target][outcome][0])
        frequency_repl2.append(frequency_comb[target][outcome][1])
        frequency_repl3.append(frequency_comb[target][outcome][2])

    permuted_replc1 = np.random.permutation(frequency_repl1)
    permuted_replc2 = np.random.permutation(frequency_repl2)
    permuted_replc3 = np.random.permutation(frequency_repl3)

    if len(frequency_repl1) > 1:
        
        # normal corr coefficient (i.e. frequencies not permuted (Fig. 2A in the paper))
        if np.var(frequency_repl1) > 0 and np.var(frequency_repl2) > 0:      
            repl_1_2_correlation_coefficients.append(np.corrcoef(frequency_repl1, frequency_repl2)[0][1])
            if(np.corrcoef(frequency_repl1, frequency_repl2)[0][1] < 0):
                print(np.corrcoef(frequency_repl1, frequency_repl2)[0][1], frequency_repl1, frequency_repl2)
        if np.var(frequency_repl1) > 0 and np.var(frequency_repl3) > 0:      
            repl_1_3_correlation_coefficients.append(np.corrcoef(frequency_repl1, frequency_repl3)[0][1])
        if np.var(frequency_repl2) > 0 and np.var(frequency_repl3) > 0:      
            repl_2_3_correlation_coefficients.append(np.corrcoef(frequency_repl2, frequency_repl3)[0][1])

        # permuted corr coefficient (i.e. frequencies not permuted (Fig. 2B in the paper))
        if np.var(frequency_repl1) > 0 and np.var(permuted_replc2) > 0:      
            repl_1Normal_2Permuted_correlation_coefficients.append(np.corrcoef(frequency_repl1, permuted_replc2)[0][1])
            if(np.corrcoef(frequency_repl1, permuted_replc2)[0][1] < 0):
                print(np.corrcoef(frequency_repl1, permuted_replc2)[0][1], frequency_repl1, permuted_replc2)
        if np.var(frequency_repl1) > 0 and np.var(permuted_replc3) > 0:      
            repl_1Normal_3Permuted_correlation_coefficients.append(np.corrcoef(frequency_repl1, permuted_replc3)[0][1])
        if np.var(frequency_repl2) > 0 and np.var(permuted_replc1) > 0:      
            repl_2Normal_1Permuted_correlation_coefficients.append(np.corrcoef(frequency_repl2, permuted_replc1)[0][1])
        if np.var(frequency_repl2) > 0 and np.var(permuted_replc3) > 0:      
            repl_2Normal_3Permuted_correlation_coefficients.append(np.corrcoef(frequency_repl2, permuted_replc3)[0][1])
        if np.var(frequency_repl3) > 0 and np.var(permuted_replc1) > 0:      
            repl_3Normal_1Permuted_correlation_coefficients.append(np.corrcoef(frequency_repl3, permuted_replc1)[0][1])
        if np.var(frequency_repl3) > 0 and np.var(permuted_replc2) > 0:      
            repl_3Normal_2Permuted_correlation_coefficients.append(np.corrcoef(frequency_repl3, permuted_replc2)[0][1])


repl_1_2_correlation_coefficients_dataframe = pd.DataFrame([['rep 1 vs. rep 2', x] for x in repl_1_2_correlation_coefficients if x >= 0], columns=['index', 'corr_coef'])

repl_1_3_correlation_coefficients_dataframe = pd.DataFrame([['rep 1 vs. rep 3', x] for x in repl_1_3_correlation_coefficients if x >= 0], columns=['index', 'corr_coef'])

repl_2_3_correlation_coefficients_dataframe = pd.DataFrame([['rep 2 vs. rep 3', x] for x in repl_2_3_correlation_coefficients if x >= 0], columns=['index', 'corr_coef'])

permuted_concatenated = repl_1Normal_2Permuted_correlation_coefficients + repl_1Normal_3Permuted_correlation_coefficients + repl_2Normal_1Permuted_correlation_coefficients + repl_2Normal_3Permuted_correlation_coefficients + repl_3Normal_1Permuted_correlation_coefficients + repl_3Normal_2Permuted_correlation_coefficients

repl_permuted_correlation_coefficients_dataframe = pd.DataFrame([['permuted comparison', x] for x in permuted_concatenated if x >= 0], columns=['index', 'corr_coef'])

repl_combined = [repl_1_2_correlation_coefficients_dataframe, repl_1_3_correlation_coefficients_dataframe, repl_2_3_correlation_coefficients_dataframe, repl_permuted_correlation_coefficients_dataframe]
repl_combined_dataframe = pd.concat(repl_combined)
# dirty_corr_coef = [ele for ele in repl_1_2_correlation_coefficients if ele > 0]
sns.set(style='whitegrid')
sns.violinplot(x="index", y="corr_coef", data=repl_combined_dataframe)
plt.show()

with open("pearson_corr_1_2.txt", 'w') as file:
    file.write(str(repl_1_2_correlation_coefficients))

with open("pearson_corr_1_3.txt", 'w') as file:
    file.write(str(repl_1_3_correlation_coefficients))

with open("pearson_corr_2_3.txt", 'w') as file:
    file.write(str(repl_2_3_correlation_coefficients))


# with open("repl1_frequency_data.txt", 'w') as file:
#     for key in frequency_repl1:
#         file.write(key + ": " + str(frequency_repl1[key]) + "\n")

# with open("repl2_frequency_data.txt", 'w') as file:
#     for key in frequency_repl2:
#         file.write(key + ": " + str(frequency_repl2[key]) + "\n")

# with open("repl3_frequency_data.txt", 'w') as file:
#     for key in frequency_repl3:
#         file.write(key + ": " + str(frequency_repl3[key]) + "\n")
