import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

data = [["sequence", "classes for insertions"],
        ["sequence", "classes for insertions and deletions by size"],
        ["microhomology", "classes for insertions and deletions by size"],
        ["all", "classes for insertions and deletions by size"],
        ["microhomology", "classes for deletions grouped by size"],
        ["all", "all classes"]]
names = [6, 1, 2, 5, 3, 4]

"""
for i in range(6):
    means_train = np.load("crossval" + str(i + 1) + "_train.npy")
    means_test = np.load("crossval" + str(i + 1) + "_test.npy")
    if i==2:
        means_tr_a = np.load("crossval" + str(i + 1) + "a_train.npy")
        means_te_a = np.load("crossval" + str(i + 1) + "a_test.npy")

        means_train = np.concatenate((means_tr_a, means_train[4:, :]))
        means_test = np.concatenate((means_te_a, means_test[4:, :]))
        np.save("crossval" + str(i + 1) + "_train.npy", means_train)
        np.save("crossval" + str(i + 1) + "_test.npy", means_test)

    ns = np.arange(1, means_test.shape[0] + 1)

    fig, ax = plt.subplots()
    ax.plot(ns, np.mean(means_train, axis=1), label="training data")
    ax.plot(ns, np.mean(means_test, axis=1), label="test data")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Average canonical correlation")
    ax.set_title("Average canonical correlations for different numbers of CCA components\non " + data[i][0] +
                 " features and " + data[i][1])
    if i < 4:
        ax.set_ylim([0.5, 1])
    ax.legend()
    plt.show()
    fig.savefig("crossval"+str(i+1)+".png")
"""

fig, axs = plt.subplots(2,3, figsize=(14,8))
fig.suptitle("Average canonical correlations for different numbers of CCA components", fontsize=20)
plt.subplots_adjust(hspace=0.4)
labels = ["A)","B)","C)","D)","E)", "F)"]

for j, ax in enumerate(axs.flat):
    i = names[j]-1
    means_train = np.load("crossval" + str(i + 1) + "_train.npy")
    means_test = np.load("crossval" + str(i + 1) + "_test.npy")
    ns = np.arange(1, means_test.shape[0] + 1)

    ax.plot(ns, np.mean(means_train, axis=1), label="training data")
    ax.plot(ns, np.mean(means_test, axis=1), label="test data")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Correlation")
    ax.set_title(data[i][0] + " features and\n" + data[i][1], fontsize=12)
    if i<5:
        ax.set_ylim([0.5, 1])

    trans = mtransforms.ScaledTranslation(-20 / 72, 18 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, labels[j], transform=ax.transAxes + trans,
            fontsize='12', va='bottom')
    a=0
plt.figlegend(['Training data', 'Test data'], loc='outside right center')
plt.show()
fig.savefig("crossval_all2.png")