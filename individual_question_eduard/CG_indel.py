from analysis_helper import *
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.colors as mcolors

# obtaining sequence data
targets_training = read_data("data/Lindel_training.txt")
targets_test = read_data("data/Lindel_test.txt")
targets_algient_NHEJ = read_data("data/algient_NHEJ_guides_final.txt")
long_replicate1_targets, replicate1_targets = read_pkl_raw_data('data/NHEJ_rep1_final_matrix.pkl') 
long_replicate2_targets, replicate2_targets = read_pkl_raw_data('data/NHEJ_rep2_final_matrix.pkl') 
long_replicate3_targets, replicate3_targets = read_pkl_raw_data('data/NHEJ_rep3_final_matrix.pkl') 
# checking how many target sequences are CG dense

# training and test data
filtered_targets_1 = observed_to_expected_CpG_filter_1(targets_training, 0.6, 0.5)
filtered_targets_1_test = observed_to_expected_CpG_filter_1(targets_test, 0.6, 0.5)

targets_training_with_features = read_data_with_features("data/Lindel_training.txt")
targets_test_with_features = read_data_with_features("data/Lindel_test.txt")

# # DISTRIBUTION OF FREQUENCIES SPARSE VS. DENSE

# # Training data

filtered_targets_1_sequences = [i[0] for i in filtered_targets_1]

filtered_targets_1_sequences_test = [i[0] for i in filtered_targets_1_test]

ins_del_ratio_df = pd.DataFrame(columns=("tr/te", "insertion ratio", "deletion ratio", "dense/sparse"))
ins_del_ratio_combined_df = pd.DataFrame(columns=("tr/te", "ins/del ratio", "CpG density: ", "GC content"))
ins_del_ratio_dense_df = pd.DataFrame(columns=("tr/te", "ins/del ratio", "GC content"))
ins_del_ratio_sparse_df = pd.DataFrame(columns=("tr/te", "ins/del ratio", "GC content"))
filtered_targets_1_np = np.array(filtered_targets_1)

count = 0
count2 = 0
def gc_content(seq):
    C_occurrences = seq.count('C')
    G_occurrences = seq.count('G')
    return (C_occurrences + G_occurrences) / len(seq)

for i, data_point in enumerate(targets_training_with_features):
    seq = data_point[0] 
    ins_r = np.sum(data_point[-21:])
    del_r = np.sum(data_point[3033+1:3033+1+536])
    CG_density = gc_content(seq)
    if seq in filtered_targets_1_sequences:   
        ins_del_ratio_df.loc[i] = ["Training", ins_r, del_r, "dense"]
        if del_r != 0:
            ins_del_ratio_combined_df.loc[i] = ["Training", ins_r/del_r, "dense", CG_density]
            ins_del_ratio_dense_df.loc[count] = ["Training", ins_r/del_r, CG_density]
            count += 1
    else:
        ins_del_ratio_df.loc[i] = ["Training", ins_r, del_r, "sparse"]
        if del_r != 0:
            ins_del_ratio_combined_df.loc[i] = ["Training", ins_r/del_r, "sparse", CG_density]
            ins_del_ratio_sparse_df.loc[count2] = ["Training", ins_r/del_r, CG_density]
            count2 += 1
            
for i, data_point in enumerate(targets_test_with_features):
    seq = data_point[0] 
    ins_r = np.sum(data_point[-21:])
    del_r = np.sum(data_point[3033+1:3033+1+536])
    CG_density = gc_content(seq)
    if seq in filtered_targets_1_sequences_test:   
        ins_del_ratio_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r, del_r, "dense"]
        if del_r != 0:
            ins_del_ratio_combined_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r/del_r, "dense", CG_density]
            ins_del_ratio_dense_df.loc[count] = ["Training", ins_r/del_r, CG_density]
            count += 1
    else:
        ins_del_ratio_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r, del_r, "sparse"]
        if del_r != 0:
            ins_del_ratio_combined_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r/del_r, "sparse", CG_density]
            ins_del_ratio_sparse_df.loc[count2] = ["Test", ins_r/del_r, CG_density]
            count2 += 1

plot_GC_indel = sns.lmplot( x="GC content", y="ins/del ratio", data=ins_del_ratio_combined_df, x_estimator=np.mean, col="CpG density: ", height=4)
for ax in plot_GC_indel.axes.flatten():
    ax.grid() 
plt.show()