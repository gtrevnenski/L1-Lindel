import numpy as np
from sklearn.cross_decomposition import CCA


def evaluate_CCA(n, X_train, Y_train, X_test, Y_test):
    """
    Fits a CCA model with n components to training data X_train and Y_train.
    Transforms both training and validation data and computes the canonical correlation for both.
    Returns the mean correlation for the training data, and mean correlation for validation data.
    """
    cca = CCA(n_components=n)
    cca.fit(X_train, Y_train)
    X_trt, Y_trt = cca.transform(X_train, Y_train)
    X_tt, Y_tt = cca.transform(X_test, Y_test)
    corrs_train = [np.corrcoef(X_trt[:, i], Y_trt[:, i])[0, 1] for i in range(n)]
    corrs_test = [np.corrcoef(X_tt[:, i], Y_tt[:, i])[0, 1] for i in range(n)]

    """print("correlations on training data: ", corrs_train)
    print("mean correlation on training data: ", np.mean(corrs_train))
    print("correlations on test data: ", corrs_test)
    print("mean correlation on test data: ", np.mean(corrs_test))"""

    return np.mean(corrs_train), np.mean(corrs_test), cca



