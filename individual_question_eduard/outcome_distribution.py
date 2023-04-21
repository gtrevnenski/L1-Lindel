
from analysis_helper import *
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.colors as mcolors

targets_training = read_data("Project_Eduard/data/Lindel_training.txt")
targets_test = read_data("Project_Eduard/data/Lindel_test.txt")
training_test_combined = targets_training + targets_test

targets_training_with_features = read_data_with_features("Project_Eduard/data/Lindel_training.txt")
targets_test_with_features = read_data_with_features("Project_Eduard/data/Lindel_test.txt")
targets_test_with_features_combined = targets_training_with_features + targets_test_with_features
filtered_combined = observed_to_expected_CpG_filter_1(training_test_combined, 0.6, 0.5)
# Training data

targets_combined_features_dense = [row for row in targets_test_with_features_combined if row[0] in np.array(filtered_combined)[:,0]]
targets_combined_features_sparse = [row for row in targets_test_with_features_combined if row[0] not in np.array(filtered_combined)[:,0]]

targets_combined_features_dense_np = np.array(targets_combined_features_dense)[:,1:]
targets_combined_features_sparse_np = np.array(targets_combined_features_sparse)[:,1:]

targets_combined_outcomes_dense_np_deletions = targets_combined_features_dense_np[:, 3033:3033+536].astype(float)
targets_combined_outcomes_sparse_np_deletions = targets_combined_features_sparse_np[:, 3033:3033+536].astype(float)

targets_combined_outcomes_dense_np_insertions = targets_combined_features_dense_np[:, -21:].astype(float)
targets_combined_outcomes_sparse_np_insertions = targets_combined_features_sparse_np[:, -21:].astype(float)

targets_combined_dense_mean_outcome_frequencies_deletions = np.mean(targets_combined_outcomes_dense_np_deletions, axis=0)
targets_combined_sparse_mean_outcome_frequencies_deletions = np.mean(targets_combined_outcomes_sparse_np_deletions, axis=0)

targets_combined_dense_mean_outcome_frequencies_insertions = np.mean(targets_combined_outcomes_dense_np_insertions, axis=0)
targets_combined_sparse_mean_outcome_frequencies_insertions = np.mean(targets_combined_outcomes_sparse_np_insertions, axis=0)

# # # Test data
# filtered_targets_1_sequences_test = [i[0] for i in filtered_targets_1_test]
# targets_test_features_dense = [row for row in targets_test_with_features if row[0] in filtered_targets_1_sequences_test]
# targets_test_features_sparse = [row for row in targets_test_with_features if row[0] not in filtered_targets_1_sequences_test]

# targets_test_features_dense_np = np.array(targets_test_features_dense)[:,1:]
# targets_test_features_sparse_np = np.array(targets_test_features_sparse)[:,1:]

# targets_test_outcomes_dense_np_deletions = targets_test_features_dense_np[:, 3033:3033+536].astype(float)
# targets_test_outcomes_sparse_np_deletions = targets_test_features_sparse_np[:, 3033:3033+536].astype(float)

# targets_test_outcomes_dense_np_insertions = targets_test_features_dense_np[:, -21:].astype(float)
# targets_test_outcomes_sparse_np_insertions = targets_test_features_sparse_np[:, -21:].astype(float)

# targets_test_dense_mean_outcome_frequencies_deletions = np.mean(targets_test_outcomes_dense_np_deletions, axis=0)
# targets_test_sparse_mean_outcome_frequencies_deletions = np.mean(targets_test_outcomes_sparse_np_deletions, axis=0)

# targets_test_dense_mean_outcome_frequencies_insertions = np.mean(targets_test_outcomes_dense_np_insertions, axis=0)
# targets_test_sparse_mean_outcome_frequencies_insertions = np.mean(targets_test_outcomes_sparse_np_insertions, axis=0)


xticks = [0] + [i*139 for i in range(5) if i != 0]
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
axes[0].bar(np.arange(21), targets_combined_dense_mean_outcome_frequencies_insertions)
axes[0].bar(np.arange(536), targets_combined_dense_mean_outcome_frequencies_deletions)
axes[0].set_title("CpG dense sequences average outcome distribution")
axes[0].set_ylim(0, 0.13)
axes[0].grid(axis = 'y')
axes[0].legend(["deletions", "insertions"])
# axes[0].set_ylabel("Average relative frequency")


axes[1].bar(np.arange(21), targets_combined_sparse_mean_outcome_frequencies_insertions)
axes[1].bar(np.arange(536), targets_combined_sparse_mean_outcome_frequencies_deletions)
axes[1].set_title("CpG sparse sequences average outcome distribution")
axes[1].set_xticks(xticks)
axes[1].set_xlabel("mutational outcome")
axes[1].legend(["deletions", "insertions"])
axes[1].grid(axis = 'y')
axes[1].set_ylim(0, 0.13)
# axes[1].set_ylabel("")
fig.text(0.04, 0.5, 'Average relative frequency', va='center', rotation='vertical')
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.ylabel("Average relative frequency")
plt.show()

# # xticks = [0] + [i*139 for i in range(5) if i != 0]
# # fig, axes = plt.subplots(2, 1, sharex=True)
# # plt.ylabel("Average frequency")
# # axes[0].bar(np.arange(21), targets_test_dense_mean_outcome_frequencies_insertions)
# # axes[0].bar(np.arange(536), targets_test_dense_mean_outcome_frequencies_deletions)
# # axes[0].set_title("Test set dense sequences average outcome distribution")
# # axes[0].set_ylim(0, 0.12)
# # axes[0].grid(axis = 'y')
# # axes[0].legend(["deletions", "insertions"])
# # axes[0].set_ylabel("Average relative frequency")


# # axes[1].bar(np.arange(21), targets_test_sparse_mean_outcome_frequencies_insertions)
# # axes[1].bar(np.arange(536), targets_test_sparse_mean_outcome_frequencies_deletions)
# # axes[1].set_title("Test set sparse sequences average outcome distribution")
# # axes[1].set_xticks(xticks)
# # axes[1].set_xlabel("mutational outcome")
# # axes[1].legend(["deletions", "insertions"])
# # axes[1].grid(axis = 'y')
# # axes[1].set_ylim(0, 0.12)
# # axes[1].set_ylabel("Average relative frequency")

# # plt.show()