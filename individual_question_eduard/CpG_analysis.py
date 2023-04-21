from analysis_helper import *
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.colors as mcolors

### SCRIPT FOR QUICK CHECK UPS AND NOTES --> CAN BE IGNORED ###
















# read meta data about the features of the training and test data
label, rev_index, features = pkl.load(open('data/feature_index_all.pkl','rb'))

# rev_index is a dict {column (0-556): type of event outcome} where deletions are solely defined by their start/end points (0-535 (including 535)), 
# single nucleotide insertions and dinucleotide insertions (536-555) are defined by 'number + nucleotide(s)' and insertions greater than 2 bp 
# in length by '3' (556).
print(rev_index[425])
print(rev_index[458])
print(rev_index[486])
print(rev_index[538])
print(rev_index[539])
print(rev_index[540])
# label is dict which has the keys and values inverted of rev_index so that you can find the column that corresponds to a certain event outcome
# class of interest. 

# features corresponding only to MH tracts (the 384 sequence features are one-hot encoded and not described here)
# every deletion event outcome class has it's own MH tracts features. The format is as follows: 'starting position, (starting position + number = end position), length of MH tract.'

# # obtaining sequence data
# targets_training = read_data("data/Lindel_training.txt")
# targets_test = read_data("data/Lindel_test.txt")
# targets_algient_NHEJ = read_data("data/algient_NHEJ_guides_final.txt")
# long_replicate1_targets, replicate1_targets = read_pkl_raw_data('data/NHEJ_rep1_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched
# long_replicate2_targets, replicate2_targets = read_pkl_raw_data('data/NHEJ_rep2_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched
# long_replicate3_targets, replicate3_targets = read_pkl_raw_data('data/NHEJ_rep3_final_matrix.pkl') # is is true that the 6th item is the target sequence? I think so since it's 20 nucleotides long and untouched

# # checking how many target sequences are CG dense

# # training and test data
# filtered_targets_1 = observed_to_expected_CpG_filter_1(targets_training, 0.6, 0.5)
# filtered_targets_1_test = observed_to_expected_CpG_filter_1(targets_test, 0.6, 0.5)

# # long sequences from algient_NHEJ_guides_final.txt file
# filtered_algient = observed_to_expected_CpG_filter_1(targets_algient_NHEJ, 0.6, 0.5) # find out what this data is exactly? 

# # replicates 20 bp sequences
# replicate1_CG_dense1 = observed_to_expected_CpG_filter_1(replicate1_targets, 0.6, 0.5)
# replicate2_CG_dense1 = observed_to_expected_CpG_filter_1(replicate2_targets, 0.6, 0.5)
# replicate3_CG_dense1 = observed_to_expected_CpG_filter_1(replicate3_targets, 0.6, 0.5)

# # replicates long (how many bp?) bp sequences
# long_replicate1_CG_dense1 = observed_to_expected_CpG_filter_1(long_replicate1_targets, 0.6, 0.5)
# long_replicate2_CG_dense1 = observed_to_expected_CpG_filter_1(long_replicate2_targets, 0.6, 0.5)
# long_replicate3_CG_dense1 = observed_to_expected_CpG_filter_1(long_replicate3_targets, 0.6, 0.5)

# targets_training_with_features = read_data_with_features("data/Lindel_training.txt")
# targets_test_with_features = read_data_with_features("data/Lindel_test.txt")

# # # DISTRIBUTION OF FREQUENCIES SPARSE VS. DENSE

# # # Training data

# filtered_targets_1_sequences = [i[0] for i in filtered_targets_1]
# targets_training_features_dense = [row for row in targets_training_with_features if row[0] in filtered_targets_1_sequences]
# targets_training_features_sparse = [row for row in targets_training_with_features if row[0] not in filtered_targets_1_sequences]

# targets_training_features_dense_np = np.array(targets_training_features_dense)[:,1:]
# targets_training_features_sparse_np = np.array(targets_training_features_sparse)[:,1:]

# targets_training_outcomes_dense_np_deletions = targets_training_features_dense_np[:, 3033:3033+536].astype(float)
# targets_training_outcomes_sparse_np_deletions = targets_training_features_sparse_np[:, 3033:3033+536].astype(float)

# targets_training_outcomes_dense_np_insertions = targets_training_features_dense_np[:, -21:].astype(float)
# targets_training_outcomes_sparse_np_insertions = targets_training_features_sparse_np[:, -21:].astype(float)
# # print(np.sum(targets_training_features_dense_np[0][3033:3033+557].astype(float))) # =  1!

# targets_training_dense_mean_outcome_frequencies_deletions = np.mean(targets_training_outcomes_dense_np_deletions, axis=0)
# # print(np.sum(targets_training_dense_mean_outcome_frequencies_deletions))
# # print(len(targets_training_dense_mean_outcome_frequencies_deletions))
# # print(ks_2samp(targets_training_outcomes_dense_np_deletions[0], targets_training_outcomes_sparse_np_deletions[0]))
# targets_training_sparse_mean_outcome_frequencies_deletions = np.mean(targets_training_outcomes_sparse_np_deletions, axis=0)

# # using batches of samples else too many calculations

# # targets_training_features_dense_all_outcomes = targets_training_features_dense_np[:, -557:].astype(float)
# # targets_training_features_sparse_all_outcomes = targets_training_features_sparse_np[:, -557:].astype(float)

# # av_KS = 0
# # p_value = 0 
# # for sparse_distribution in targets_training_features_sparse_all_outcomes:
# #     subset_dense_distributions = targets_training_features_dense_all_outcomes[np.random.choice(targets_training_features_dense_all_outcomes.shape[0], 20, replace=False), :]
# #     for dense_distribution in subset_dense_distributions:
# #         result = ks_2samp(sparse_distribution, dense_distribution)
# #         av_KS += result[0]
# #         p_value += result[1]

# #         if np.sum(dense_distribution) < 0.99:
# #             print('dense wrong')
# #             print(np.sum(dense_distribution))
# #         elif np.sum(sparse_distribution) < 0.99:
# #             print('sparse wrong')
# #             print(np.sum(sparse_distribution))

# # print(av_KS / (len(targets_training_features_sparse_all_outcomes)*20))
# # print(p_value / (len(targets_training_features_sparse_all_outcomes)*20))

# # def average_smirnov(distribution1, distribution2, batch_size=20):
# #     av_KS = 0
# #     p_value = 0
# #     for sparse_distribution in distribution1:
# #         print(distribution2.shape)
# #         subset_dense_distributions = distribution2[np.random.choice(distribution2.shape[0], batch_size, replace=False), :]
# #         for dense_distribution in subset_dense_distributions:
# #             result = ks_2samp(sparse_distribution, dense_distribution)
# #             av_KS += result[0]
# #             p_value += result[1]
# #     av_KS = av_KS / (len(distribution1)*batch_size)
# #     p_value = p_value / (len(distribution1)*batch_size)

# #     return av_KS, p_value

# targets_training_with_features_all_outcomes = np.array(targets_training_with_features)[:, -557:].astype(float)
# targets_test_with_features_all_outcomes = np.array(targets_test_with_features)[:, -557:].astype(float)
# # print(targets_test_with_features_all_outcomes.shape)

# # KS_values = []
# # p_values = []
# # for i in range(200):
# #     subset_1 = np.array(targets_test_with_features_all_outcomes[np.random.choice(targets_test_with_features_all_outcomes.shape[0], len(filtered_targets_1_test), replace=False), :])
# #     print(subset_1.shape)
# #     subset_2 = []
# #     for seq in targets_test_with_features_all_outcomes:
# #         print(seq)
# #         if seq not in subset_1:
# #             subset_2.append(seq)
# #             print('here')
# #     print(subset_2.shape)
# #     KS, p = average_smirnov(subset_1, subset_2)
# #     KS_values.append(KS)
# #     p_values.append(p)

# # print('KS_values:')
# # print(KS_values)
# # print('p_values:')
# # print(p_values)

# # targets_training_dense_mean_outcome_frequencies_insertions = np.mean(targets_training_outcomes_dense_np_insertions, axis=0)
# # targets_training_sparse_mean_outcome_frequencies_insertions = np.mean(targets_training_outcomes_sparse_np_insertions, axis=0)

# # # Test data
# filtered_targets_1_sequences_test = [i[0] for i in filtered_targets_1_test]
# targets_test_features_dense = [row for row in targets_test_with_features if row[0] in filtered_targets_1_sequences_test]
# targets_test_features_sparse = [row for row in targets_test_with_features if row[0] not in filtered_targets_1_sequences_test]

# # targets_test_features_dense_np = np.array(targets_test_features_dense)[:,1:]
# # targets_test_features_sparse_np = np.array(targets_test_features_sparse)[:,1:]

# # targets_test_outcomes_dense_np_deletions = targets_test_features_dense_np[:, 3033:3033+536].astype(float)
# # targets_test_outcomes_sparse_np_deletions = targets_test_features_sparse_np[:, 3033:3033+536].astype(float)

# # targets_test_outcomes_dense_np_insertions = targets_test_features_dense_np[:, -21:].astype(float)
# # targets_test_outcomes_sparse_np_insertions = targets_test_features_sparse_np[:, -21:].astype(float)

# # targets_test_dense_mean_outcome_frequencies_deletions = np.mean(targets_test_outcomes_dense_np_deletions, axis=0)
# # targets_test_sparse_mean_outcome_frequencies_deletions = np.mean(targets_test_outcomes_sparse_np_deletions, axis=0)

# # targets_test_dense_mean_outcome_frequencies_insertions = np.mean(targets_test_outcomes_dense_np_insertions, axis=0)
# # targets_test_sparse_mean_outcome_frequencies_insertions = np.mean(targets_test_outcomes_sparse_np_insertions, axis=0)

# # 2-sample Kolmogorov-Smirnov test for each column

# # sample1 and sample2 are your two arrays of shape (m, 536)

# # # calculate the empirical distribution function (ECDF) for each column separately
# # ecdf1 = lambda x: sum(targets_training_outcomes_dense_np <= x, axis=0) / len(targets_test_outcomes_dense_np)
# # ecdf2 = lambda x: sum(targets_test_outcomes_sparse_np <= x, axis=0) / len(targets_test_outcomes_sparse_np)

# # perform the 2-sample Kolmogorov-Smirnov test

# # targets_training_outcomes_dense_np = np.array(targets_training_features_dense)[:,-557:]
# # targets_training_outcomes_sparse_np = np.array(targets_training_features_sparse)[:,-557:]
# # targets_test_outcomes_dense_np = np.array(targets_test_features_dense)[:,-557:]
# # targets_test_outcomes_sparse_np = np.array(targets_training_features_sparse)[:,-557:]

# # ks_stats = []
# # p_values = []
# # mutational_outcomes = []
# # for i in range(targets_training_outcomes_dense_np.shape[1]):
# #     ks_stat, p_value = ks_2samp(targets_training_outcomes_dense_np[:, i], targets_training_outcomes_sparse_np[:, i], alternative='two-sided', mode='auto')
# #     ks_stats.append(ks_stat)
# #     p_values.append(p_value)

# # # print the test statistics and p-values for each column
# # for i in range(targets_training_outcomes_dense_np.shape[1]):
# #     print(f"Column {i+1} - Kolmogorov-Smirnov test statistic: {ks_stats[i]}, p-value: {p_values[i]}")
# #     if p_values[i] < 0.05:
# #         print(f"LOOK AT THIS ONE: {p_values[i]}")
# #         mutational_outcomes.append([i+1, p_values[i]])

# # print(mutational_outcomes)
# # fig, axes = plt.subplots(1,2)
# # binwidth1 = (max(ks_stats)-min(ks_stats)) / 50
# # axes[0].hist(ks_stats, bins=np.arange(min(ks_stats), max(ks_stats) + binwidth1, binwidth1), color=mcolors.CSS4_COLORS['wheat'])
# # axes[0].set_xlabel("KS test value")
# # axes[0].set_ylabel("Frequency")
# # axes[0].set_title("CpG dense training data KS value distribution")

# # binwidth2 = (max(p_values)-min(p_values)) / 50
# # axes[1].hist(p_values, bins=np.arange(min(p_values), max(p_values) + binwidth2, binwidth2), color=mcolors.CSS4_COLORS['wheat'])
# # axes[1].set_xlabel("p-value")
# # axes[1].set_ylabel("Frequency")
# # axes[1].set_title("CpG dense training data p-value distribution")
# # plt.show()

# # xticks = [0] + [i*139 for i in range(5) if i != 0]
# # fig, axes = plt.subplots(2, 1, sharex=True)
# # plt.ylabel("Average relative frequency")
# # axes[0].bar(np.arange(21), targets_training_dense_mean_outcome_frequencies_insertions)
# # axes[0].bar(np.arange(536), targets_training_dense_mean_outcome_frequencies_deletions)
# # axes[0].set_title("Training set dense sequences average outcome distribution")
# # axes[0].set_ylim(0, 0.12)
# # axes[0].grid(axis = 'y')
# # axes[0].legend(["deletions", "insertions"])
# # axes[0].set_ylabel("Average relative frequency")


# # axes[1].bar(np.arange(21), targets_training_sparse_mean_outcome_frequencies_insertions)
# # axes[1].bar(np.arange(536), targets_training_sparse_mean_outcome_frequencies_deletions)
# # axes[1].set_title("Training set sparse sequences average outcome distribution")
# # axes[1].set_xticks(xticks)
# # axes[1].set_xlabel("mutational outcome")
# axes[1].legend(["deletions", "insertions"])
# axes[1].grid(axis = 'y')
# axes[1].set_ylim(0, 0.12)
# axes[1].set_ylabel("Average relative frequency")

# plt.show()

# xticks = [0] + [i*139 for i in range(5) if i != 0]
# fig, axes = plt.subplots(2, 1, sharex=True)
# plt.ylabel("Average frequency")
# axes[0].bar(np.arange(21), targets_test_dense_mean_outcome_frequencies_insertions)
# axes[0].bar(np.arange(536), targets_test_dense_mean_outcome_frequencies_deletions)
# axes[0].set_title("Test set dense sequences average outcome distribution")
# axes[0].set_ylim(0, 0.12)
# axes[0].grid(axis = 'y')
# axes[0].legend(["deletions", "insertions"])
# axes[0].set_ylabel("Average relative frequency")


# axes[1].bar(np.arange(21), targets_test_sparse_mean_outcome_frequencies_insertions)
# axes[1].bar(np.arange(536), targets_test_sparse_mean_outcome_frequencies_deletions)
# axes[1].set_title("Test set sparse sequences average outcome distribution")
# axes[1].set_xticks(xticks)
# axes[1].set_xlabel("mutational outcome")
# axes[1].legend(["deletions", "insertions"])
# axes[1].grid(axis = 'y')
# axes[1].set_ylim(0, 0.12)
# axes[1].set_ylabel("Average relative frequency")

# plt.show()
    

# # BOX PLOT
# print(np.array(filtered_targets_1)[:,1])
# tips = pd.DataFrame({"day": [1]*len(filtered_targets_1), "total_bill": np.array(filtered_targets_1)[:,1]})
# sns.catplot(data=tips, x="day", y="total_bill", kind="box")
# plt.show()


# INSERTION/DEL RATIO
# print(targets_training_with_features[0][3033+1:3033+557+1]) 
# print(len(targets_training_with_features[0][3033+1:3033+557+1])) # outcome classes 

# ins_del_ratio_df = pd.DataFrame(columns=("tr/te", "insertion ratio", "deletion ratio", "dense/sparse"))
# ins_del_ratio_combined_df = pd.DataFrame(columns=("tr/te", "ins/del ratio", "CpG density: ", "GC content"))
# ins_del_ratio_dense_df = pd.DataFrame(columns=("tr/te", "ins/del ratio", "GC content"))
# ins_del_ratio_sparse_df = pd.DataFrame(columns=("tr/te", "ins/del ratio", "GC content"))
# filtered_targets_1_np = np.array(filtered_targets_1)

# count = 0
# count2 = 0
# def gc_content(seq):
#     C_occurrences = seq.count('C')
#     G_occurrences = seq.count('G')
#     return (C_occurrences + G_occurrences) / len(seq)

# for i, data_point in enumerate(targets_training_with_features):
#     seq = data_point[0] 
#     ins_r = np.sum(data_point[-21:])
#     del_r = np.sum(data_point[3033+1:3033+1+536])
#     CG_density = gc_content(seq)
#     if seq in filtered_targets_1_sequences:   
#         ins_del_ratio_df.loc[i] = ["Training", ins_r, del_r, "dense"]
#         if del_r != 0:
#             ins_del_ratio_combined_df.loc[i] = ["Training", ins_r/del_r, "dense", CG_density]
#             ins_del_ratio_dense_df.loc[count] = ["Training", ins_r/del_r, CG_density]
#             count += 1
#     else:
#         ins_del_ratio_df.loc[i] = ["Training", ins_r, del_r, "sparse"]
#         if del_r != 0:
#             ins_del_ratio_combined_df.loc[i] = ["Training", ins_r/del_r, "sparse", CG_density]
#             ins_del_ratio_sparse_df.loc[count2] = ["Training", ins_r/del_r, CG_density]
#             count2 += 1
            
# for i, data_point in enumerate(targets_test_with_features):
#     seq = data_point[0] 
#     ins_r = np.sum(data_point[-21:])
#     del_r = np.sum(data_point[3033+1:3033+1+536])
#     CG_density = gc_content(seq)
#     if seq in filtered_targets_1_sequences_test:   
#         ins_del_ratio_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r, del_r, "dense"]
#         if del_r != 0:
#             ins_del_ratio_combined_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r/del_r, "dense", CG_density]
#             ins_del_ratio_dense_df.loc[count] = ["Training", ins_r/del_r, CG_density]
#             count += 1
#     else:
#         ins_del_ratio_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r, del_r, "sparse"]
#         if del_r != 0:
#             ins_del_ratio_combined_df.loc[i + len(ins_del_ratio_df)] = ["Test", ins_r/del_r, "sparse", CG_density]
#             ins_del_ratio_sparse_df.loc[count2] = ["Test", ins_r/del_r, CG_density]
#             count2 += 1

# # sns.catplot(data=ins_del_ratio_df, x="tr/te", y="deletion ratio", hue="dense/sparse", kind="swarm")
# # sns.catplot(data=ins_del_ratio_df, x="tr/te", y="deletion ratio", hue="dense/sparse", kind="box")
# # sns.catplot(data=ins_del_ratio_combined_df, x="tr/te", y="ins/del ratio", hue="dense/sparse", kind="box")

# # fig, axes = plt.subplots(1,2)
# plot_GC_indel = sns.lmplot( x="GC content", y="ins/del ratio", data=ins_del_ratio_combined_df, x_estimator=np.mean, col="CpG density: ", height=4)
# for ax in plot_GC_indel.axes.flatten():
#     ax.grid() 
# plt.show()
# # axes[0].grid()
# # axes[0].set_title("CpG dense sequences")

# # sns.lmplot(ax=axes[1], x="GC content", y="ins/del ratio", data=ins_del_ratio_sparse_df, x_estimator=np.mean)
# # axes[1].grid()
# axes[1].set_title("CpG dense sequences")

# print(ins_del_ratio_training_dense)

# print(len(long_replicate1_targets))
# print(len(long_replicate1_CG_dense1))
# print(len(long_replicate2_targets))
# print(len(long_replicate2_CG_dense1))
# print(len(long_replicate3_targets))
# print(len(long_replicate3_CG_dense1)) 

### STACKED BAR CHART ###
# total_amount_of_reads_in_replicates = len(replicate1_targets) + len(replicate2_targets) + len(replicate3_targets)
# non_dense = ((len(replicate1_targets) - len(replicate1_CG_dense1))/(total_amount_of_reads_in_replicates), (len(replicate2_targets) - len(replicate2_CG_dense1))/(total_amount_of_reads_in_replicates), (len(replicate3_targets) - len(replicate3_CG_dense1))/total_amount_of_reads_in_replicates)
# print(non_dense)
# dense = ((len(replicate1_CG_dense1))/total_amount_of_reads_in_replicates, (len(replicate2_CG_dense1))/total_amount_of_reads_in_replicates, (len(replicate3_CG_dense1))/total_amount_of_reads_in_replicates)
# print(dense)
# ind = np.arange(3)  
 
# fig, axes = plt.subplots(1, 2, figsize=(15,5))
# ax1 = axes[0]
# ax1.bar(ind, dense)
# ax1.bar(ind, non_dense,
#              bottom = dense)
# ax1.set_title('Proportions per replicate')
# ax1.set_ylabel('frequency')
# ax1.set_xticks([0,1,2],['replicate 1', 'replicate 2', 'replicate 3'])
# ax1.legend(('CpG dense', 'CpG sparse'), loc='best')
# ax1.set_ylim(0,1)

# total_amount_of_reads_in_training_test = len(targets_training) + len(targets_test)
# non_dense_training_test = ((len(targets_training) - len(filtered_targets_1))/total_amount_of_reads_in_training_test, (len(targets_test) - len(filtered_targets_1_test))/total_amount_of_reads_in_training_test)
# print(non_dense_training_test)
# dense_training_tests = (len(filtered_targets_1)/total_amount_of_reads_in_training_test, len(filtered_targets_1_test)/total_amount_of_reads_in_training_test)
# print(dense_training_tests)
# ind = np.arange(2)  
 
# ax2 = axes[1]
# ax2.bar(ind, dense_training_tests)
# ax2.bar(ind, non_dense_training_test,
#              bottom = dense_training_tests)
# ax2.set_title('Proportions for training and test data')
# ax2.set_ylabel('frequency')
# ax2.set_xticks([0,1],['Training', 'Test'])
# ax2.legend(('CpG dense', 'CpG sparse'), loc='best')
# ax2.set_ylim(0,1)

# plt.show()

 
# plt.ylabel('Counts')
# plt.title('CG dense vs. CG sparse sequences per replicate')
# plt.xticks(ind, ('replicate 1', 'replicate 2', 'replicate 3'))
# plt.legend((p1[0], p2[0]), ('CG dense', 'CG sparse'), loc='best')
# plt.show()

# # VIOLIN PLOT
# replicate1_CG_dense1 = [lst + ['replicate 1'] for lst in replicate1_CG_dense2]
# replicate2_CG_dense1 = [lst + ['replicate 2'] for lst in replicate2_CG_dense2]
# replicate3_CG_dense1 = [lst + ['replicate 3'] for lst in replicate3_CG_dense2]
# print(replicate3_CG_dense1[0:3])
# all_replicates_dense = replicate1_CG_dense1 + replicate2_CG_dense1 + replicate3_CG_dense1
# all_replicates_dense = pd.DataFrame(all_replicates_dense, columns =['sequence', 'obs/exp ratio', 'GC content', 'replicate version']) 
# print(all_replicates_dense[0:3])
# sns.displot(all_replicates_dense, x="obs/exp ratio")
# plt.show()
# print(len(filtered_targets_1))
# print(len(filtered_targets_1_test))

## LEFT OVER CODE IF NEEDED ##
# replicate1_CG_dense2 = observed_to_expected_CpG_filter_2(replicate1_targets, 0.6, 0.5)
# replicate2_CG_dense2 = observed_to_expected_CpG_filter_2(replicate2_targets, 0.6, 0.5)
# replicate3_CG_dense2 = observed_to_expected_CpG_filter_2(replicate3_targets, 0.6, 0.5)

# long_replicate1_CG_dense2 = observed_to_expected_CpG_filter_2(long_replicate1_targets, 0.6, 0.5)
# long_replicate2_CG_dense2 = observed_to_expected_CpG_filter_2(long_replicate2_targets, 0.6, 0.5)
# long_replicate3_CG_dense2 = observed_to_expected_CpG_filter_2(long_replicate3_targets, 0.6, 0.5)

# filtered_targets_2 = observed_to_expected_CpG_filter_2(targets_training, 0.6, 0.5)
# filtered_targets_2_test = observed_to_expected_CpG_filter_2(targets_test, 0.6, 0.5)


