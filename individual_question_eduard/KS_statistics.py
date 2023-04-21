from analysis_helper import *
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.colors as mcolors

# obtaining sequence data
targets_training = read_data("Project_Eduard/data/Lindel_training.txt")
targets_test = read_data("Project_Eduard/data/Lindel_test.txt")
targets_algient_NHEJ = read_data("Project_Eduard/data/algient_NHEJ_guides_final.txt")
long_replicate1_targets, replicate1_targets = read_pkl_raw_data('data/NHEJ_rep1_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched
long_replicate2_targets, replicate2_targets = read_pkl_raw_data('data/NHEJ_rep2_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched
long_replicate3_targets, replicate3_targets = read_pkl_raw_data('data/NHEJ_rep3_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched

# checking how many target sequences are CG dense

# training and test data
filtered_targets_1 = observed_to_expected_CpG_filter_1(targets_training, 0.6, 0.5)
filtered_targets_1_test = observed_to_expected_CpG_filter_1(targets_test, 0.6, 0.5)

targets_training_with_features = read_data_with_features("Project_Eduard/data/Lindel_training.txt")
targets_test_with_features = read_data_with_features("Project_Eduard/data/Lindel_test.txt")

filtered_targets_1_sequences = [i[0] for i in filtered_targets_1]
targets_training_features_dense = [row for row in targets_training_with_features if row[0] in filtered_targets_1_sequences]
targets_training_features_sparse = [row for row in targets_training_with_features if row[0] not in filtered_targets_1_sequences]

filtered_targets_1_sequences_test = [i[0] for i in filtered_targets_1_test]
targets_test_features_dense = [row for row in targets_test_with_features if row[0] in filtered_targets_1_sequences_test]
targets_test_features_sparse = [row for row in targets_test_with_features if row[0] not in filtered_targets_1_sequences_test]

# 2-sample Kolmogorov-Smirnov test for each column

# sample1 and sample2 are your two arrays of shape (m, 536)

targets_training_outcomes_dense_np = np.array(targets_training_features_dense)[:,-557:]
targets_training_outcomes_sparse_np = np.array(targets_training_features_sparse)[:,-557:]
targets_test_outcomes_dense_np = np.array(targets_test_features_dense)[:,-557:]
targets_test_outcomes_sparse_np = np.array(targets_test_features_sparse)[:,-557:]

print(targets_training_outcomes_dense_np.shape)
print(targets_test_outcomes_dense_np.shape)

targets_combined_outcomes_dense_np = np.concatenate((targets_training_outcomes_dense_np, targets_test_outcomes_dense_np))
targets_training_outcomes_sparse_np = np.concatenate((targets_training_outcomes_sparse_np, targets_test_outcomes_sparse_np))

ks_stats = []
p_values = []
mutational_outcomes = []
for i in range(targets_training_outcomes_dense_np.shape[1]):
    ks_stat, p_value = ks_2samp(targets_training_outcomes_dense_np[:, i], targets_training_outcomes_sparse_np[:, i], alternative='two-sided', mode='auto')
    ks_stats.append(ks_stat)
    p_values.append(p_value)

# print the test statistics and p-values for each column
for i in range(targets_training_outcomes_dense_np.shape[1]):
    print(f"Column {i+1} - Kolmogorov-Smirnov test statistic: {ks_stats[i]}, p-value: {p_values[i]}")
    if p_values[i] < 0.05:
        print(f"LOOK AT THIS ONE: {p_values[i]}")
        mutational_outcomes.append([i+1, p_values[i]])

print(mutational_outcomes)
fig, axes = plt.subplots(1,2)
binwidth1 = (max(ks_stats)-min(ks_stats)) / 50
axes[0].hist(ks_stats, bins=np.arange(min(ks_stats), max(ks_stats) + binwidth1, binwidth1), color=mcolors.CSS4_COLORS['wheat'])
axes[0].set_xlabel("KS test value")
axes[0].set_ylabel("Frequency")
axes[0].set_title("CpG dense training data KS value distribution")

binwidth2 = (max(p_values)-min(p_values)) / 50
axes[1].hist(p_values, bins=np.arange(min(p_values), max(p_values) + binwidth2, binwidth2), color=mcolors.CSS4_COLORS['wheat'])
axes[1].set_xlabel("p-value")
axes[1].set_ylabel("Frequency")
axes[1].set_title("CpG dense training data p-value distribution")
plt.show()
