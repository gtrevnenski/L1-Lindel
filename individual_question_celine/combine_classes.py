import pickle as pkl
import numpy as np
import sys

# Find labels per index
workdir  = sys.argv[1]
fname = "Lindel_training.txt"

label, rev_index, features = pkl.load(open(workdir + 'feature_index_all.pkl', 'rb'))

"""
# Construct index file for outcome classes
with open("index_to_label.txt", 'w') as f:
    for ind, label in rev_index.items():
        f.write(str(ind)+": "+label+"\n")
        
sorted_features = sorted(features.items(), key=lambda x:x[1])

# Construct index file for features
for item in sorted_features:
    classname = item[0].split('+')
    start = int(classname[0])
    end = int(classname[1])
    print(end+start)

with open("index_to_feature.txt", "w") as f:
    for item in sorted_features:
        f.write(str(item[1])+": "+item[0]+"\n")
"""

def load_data(workdir):
    """
    This part loads the training data and test data, and combines them into 2 matrices. One has all input features and
    the other has all outcome values.
    The input matrix is saved as 'input_all.npy' and output is saved as 'output_all.npy'
    """
    feature_size = 3033

    fname = "Lindel_training.txt"
    data = np.loadtxt(workdir + fname, delimiter="\t", dtype=str)
    data = data[:, 1:].astype('float32')
    X = data[:, :feature_size]
    Y = data[:, feature_size:]

    fname = "Lindel_test.txt"
    data = np.loadtxt(workdir + fname, delimiter="\t", dtype=str)
    data = data[:, 1:].astype('float32')

    X_test = np.array(data[:, :feature_size], dtype=np.float32)
    Y_test = np.array(data[:, feature_size:], dtype=np.float32)
    np.save("input_all", X)
    np.save("output_all", Y)
    np.save("input_test", X_test)
    np.save("output_test", Y_test)


def combine_deletions():
    """
    Combines the deletion classes by length, resulting in 29 classes representing deletions of lengths 1 to 29.
    Construct a dictionary with deletion length as key and list of indices in data as value.
    For each key, sum the columns of the indices in the list to combine all classes of that length.
    """
    Y = np.load("output_all.npy")
    len_to_index = {}
    maxdel = 29
    for i in range(1, maxdel + 1):
        len_to_index[i] = []
    for ind, label in rev_index.items():
        if ind >= 536:
            break
        length = int(label.split('+')[-1])
        len_to_index[length].append(ind)

    output_deletionsize = np.zeros((Y.shape[0], maxdel))
    for i in range(maxdel):
        indices = len_to_index[i + 1]
        output_deletionsize[:, i] = np.sum(Y[:, indices], axis=1)

    output_combined = np.concatenate((output_deletionsize, Y[:, -21:]), axis=1)
    np.save("output_insertiondeletionsize", output_combined)
    np.save("output_deletionsize", output_deletionsize)

def split_outputs():
    Y = np.load("output_all.npy")
    insertions = Y[:,-21:]
    deletions = Y[:, :-21]

    np.save("output_insertion", np.nan_to_num(insertions))
    np.save("output_deletion", np.nan_to_num(deletions))

def split_inputs():
    X = np.load("input_all.npy")
    sequence = X[:,-384:]
    MHtracts = X[:,:-384]

    np.save("input_sequence", np.nan_to_num(sequence))
    np.save("input_MHtracts", np.nan_to_num(MHtracts))


load_data("../data/")
split_inputs()
split_outputs()
combine_deletions()
