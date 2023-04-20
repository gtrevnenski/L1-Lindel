import numpy as np
from sklearn.model_selection import KFold
from evaluate_cca import evaluate_CCA

# Load data, choose right files! They can be created by running "combine_classes.py"
X = np.load("input_sequence.npy")
Y = np.load("output_deletion.npy")

n_max = 8  # number of n_comp values, max number of components to consider
ns = np.arange(1, n_max + 1)
splits = 4  # number of splits for the k-fold cross validation
kf = KFold(n_splits=splits)

means_train = np.zeros((n_max, splits))  # array to store mean canonical correlations for training data
means_test = np.zeros((n_max, splits))  # array to store mean canonical correlations for validation data

for n_comp in ns:
    """
    For each number of components, perform cross validation. Shuffle the observations and split into k groups.
    Fit a CCA model on k-1 groups and find the correlations between canonical variates. Repeat this k times
    """
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    for i, (train, test) in enumerate(kf.split(idx)):
        print(n_comp, i)
        X_train, X_test = X[idx[train], :], X[idx[test], :]
        Y_train, Y_test = Y[idx[train], :], Y[idx[test], :]

        corr_train, corr_test, cca = evaluate_CCA(n_comp, X_train, Y_train, X_test, Y_test)

        means_train[n_comp - 1, i] = corr_train
        means_test[n_comp - 1, i] = corr_test
        print (corr_train)
        print(corr_test)

np.save("crossval5_train", means_train)
np.save("crossval5_test", means_test)

