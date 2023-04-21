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
long_replicate1_targets, replicate1_targets = read_pkl_raw_data('data/NHEJ_rep1_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched
long_replicate2_targets, replicate2_targets = read_pkl_raw_data('data/NHEJ_rep2_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched
long_replicate3_targets, replicate3_targets = read_pkl_raw_data('data/NHEJ_rep3_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched

# checking how many target sequences are CG dense

# training and test data
filtered_targets_1 = observed_to_expected_CpG_filter_1(targets_training, 0.6, 0.5)
filtered_targets_1_test = observed_to_expected_CpG_filter_1(targets_test, 0.6, 0.5)

# replicates 20 bp sequences
replicate1_CG_dense1 = observed_to_expected_CpG_filter_1(replicate1_targets, 0.6, 0.5)
replicate2_CG_dense1 = observed_to_expected_CpG_filter_1(replicate2_targets, 0.6, 0.5)
replicate3_CG_dense1 = observed_to_expected_CpG_filter_1(replicate3_targets, 0.6, 0.5)


## STACKED BAR CHART ###
total_amount_of_reads_in_replicates = len(replicate1_targets) + len(replicate2_targets) + len(replicate3_targets)
non_dense = ((len(replicate1_targets) - len(replicate1_CG_dense1))/(total_amount_of_reads_in_replicates), (len(replicate2_targets) - len(replicate2_CG_dense1))/(total_amount_of_reads_in_replicates), (len(replicate3_targets) - len(replicate3_CG_dense1))/total_amount_of_reads_in_replicates)
print(non_dense)
dense = ((len(replicate1_CG_dense1))/total_amount_of_reads_in_replicates, (len(replicate2_CG_dense1))/total_amount_of_reads_in_replicates, (len(replicate3_CG_dense1))/total_amount_of_reads_in_replicates)
print(dense)
ind = np.arange(3)  
 
fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax1.bar(ind, dense)
ax1.bar(ind, non_dense,
             bottom = dense)
ax1.set_title('Proportions per replicate')
ax1.set_ylabel('Relative frequency')
ax1.set_xticks([0,1,2],['replicate 1', 'replicate 2', 'replicate 3'])
ax1.legend(('CpG dense', 'CpG sparse'), loc='best')
ax1.set_ylim(0,1)

total_amount_of_reads_in_training_test = len(targets_training) + len(targets_test)
non_dense_training_test = ((len(targets_training) - len(filtered_targets_1))/total_amount_of_reads_in_training_test, (len(targets_test) - len(filtered_targets_1_test))/total_amount_of_reads_in_training_test)
print(non_dense_training_test)
dense_training_tests = (len(filtered_targets_1)/total_amount_of_reads_in_training_test, len(filtered_targets_1_test)/total_amount_of_reads_in_training_test)
print(dense_training_tests)
ind = np.arange(2)  
 
ax2 = axes[1]
ax2.bar(ind, dense_training_tests)
ax2.bar(ind, non_dense_training_test,
             bottom = dense_training_tests)
ax2.set_title('Proportions for training and test data')
ax2.set_ylabel('Relative frequency')
ax2.set_xticks([0,1],['Training', 'Test'])
ax2.legend(('CpG dense', 'CpG sparse'), loc='best')
ax2.set_ylim(0,1)

plt.show()

 