import pickle as pkl
import shap
from mca import MCA
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def weight_analysis(history):
    """
        Print the weights corresponding to the features analysed in the study.
        Compute the average absolute weight over all features.
        Plot a histogram of the absolute values of the weights.

        Args:
            history: (keras.callbacks.History): object obtained from keras after fitting the model.

        Returns:
            None.
    """

    features_of_interest = {64: "weights of 'A' at position 17: ",
                            65: "weights of 'T' at position 17: ",
                            343: "weights of 'TG' at position 17/18: ",
                            331: "weights of 'CG' at position 16/17: ",
                            348: "weights of 'GA' at position 17/18: "}

    for i in features_of_interest:
        print(features_of_interest[i] + str(history.model.weights[0][i]))

    value_array = np.concatenate([var.numpy() for var in history.model.weights[0]])

    # flatten the NumPy array and convert it to a flattened list of floats
    weights_list = [abs(float(val)) for val in value_array.ravel()]

    print("Average weight: " + str(sum(weights_list) / len(weights_list)))

    plt.hist(weights_list, bins=10)

    # set the x and y axis labels
    plt.xlabel('Weight absolute value')
    plt.ylabel('Frequency')
    plt.show()

def compute_shap_values(dataset, model):
    """
        Create summary plot of the first 20 SHAP values.

        Args:
            dataset: (numpy.ndarray): training set (x_train).
            model: (keras.engine.sequential.Sequential): the neural network model.

        Returns:
            None.
    """

    feature_names = [str(i) for i in range(384)]

    # Summarize the background dataset using shap.sample()
    background = shap.sample(dataset, 400)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(dataset[:200])

    shap.summary_plot(shap_values, feature_names=feature_names)
    plt.show()

def mca_correlations(dataset):
    """
        Create correlation matrix with size 431x431 (384 features + 47 PCs). Every entry of the matrix is Pearson
        correlation between the row and the column variables.
        Print average summed-up correlation of original feature with the 47 PCs.
        Print the summed-up correlations of the 5 features of interest with the 47 PCs

        Args:
            dataset: (numpy.ndarray): training set (x_train).

        Returns:
            None.
    """
    x_train_df = pd.DataFrame(dataset)
    mca_model = MCA(x_train_df)
    fs = mca_model.fs_r(0.7)
    corr_matrix = np.corrcoef(dataset.T,
                              fs.T)  # the dimensions are 431 by 431 (these include original features and principal components at the end)
    total_correlations = []
    for i in range(384):
        feature_correlations = np.abs(corr_matrix[383:, i])
        total_correlations.append(np.sum(feature_correlations))

    average_correlation = np.mean(total_correlations)
    print("Average correlation: " + str(average_correlation))

    features_of_interest = {64: "'A' at position 17: ",
                            65: "'T' at position 17: ",
                            343: "'TG' at position 17/18: ",
                            331: "'CG' at position 16/17: ",
                            348: "'GA' at position 17/18: "}
    for i in features_of_interest:
        feature_correlations = np.abs(corr_matrix[383:, i])
        print(features_of_interest[i] + str(np.sum(feature_correlations)))



