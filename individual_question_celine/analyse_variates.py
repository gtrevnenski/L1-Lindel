import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

''' Plot canonical loads for sequence features and insertion classes only.'''
X = np.load("input_sequence.npy")
Y = np.load("output_insertion.npy")
n_comp = 3

cca = CCA(n_components=n_comp)
cca.fit(X, Y)

xloadings = cca.x_loadings_
yloadings = cca.y_loadings_

ypos = np.arange(yloadings.shape[0])  # the label locations
width = 0.25  # the width of the bars

fig, axs = plt.subplots(2,2, figsize=(10,6))
fig.suptitle("Canonical loads for sequence features and insertion classes", fontsize=18)
plt.subplots_adjust(hspace=0.4)
axs[-1, -1].axis('off')

for i in range(n_comp):
    axs[0,0].bar(range(xloadings.shape[0]), xloadings[:,i], alpha=0.4, linewidth=1)
axs[0,0].set_xlabel("Feature index")
axs[0,0].set_ylabel("Canonical load")
axs[0,0].set_title("Sequence features", fontsize=14)

axs[0,1].bar(ypos-width, yloadings[:,0], width, alpha=0.5)
axs[0,1].bar(ypos, yloadings[:,1], width, alpha=0.5)
axs[0,1].bar(ypos+width, (yloadings[:,2]), width, alpha=0.5)
axs[0,1].set_xticks(ypos, np.arange(yloadings.shape[0]))
axs[0,1].set_xlabel("Class index")
axs[0,1].set_ylabel("Canonical load")
axs[0,1].set_title("Outcome classes", fontsize=14)

# This adds a plot focusing on sequence features of nucleotide 17
xpos = np.arange(xloadings.shape[0])[60:70]
axs[1,0].bar(xpos-width, xloadings[60:70,0], width, alpha=0.5)
axs[1,0].bar(xpos, xloadings[60:70,1], width, alpha=0.5)
axs[1,0].bar(xpos+width, (xloadings[60:70,2]), width, alpha=0.5)
axs[1,0].set_xticks(xpos, np.arange(xloadings.shape[0])[60:70])
axs[1,0].set_xlabel("Feature index")
axs[1,0].set_ylabel("Canonical load")
axs[1,0].set_title("Sequence features around position 17", fontsize=14)

trans = mtransforms.ScaledTranslation(-25 / 72, 7 / 72, fig.dpi_scale_trans)
axs[0,0].text(0.0, 1.0, "A)", transform=axs[0,0].transAxes + trans,
            fontsize='14', va='bottom')
axs[0,1].text(0.0, 1.0, "B)", transform=axs[0,1].transAxes + trans,
            fontsize='14', va='bottom')
axs[1,0].text(0.0, 1.0, "C)", transform=axs[1,0].transAxes + trans,
            fontsize='14', va='bottom')

axs[1,0].legend(["Variate 1", "Variate 2", "Variate 3"], loc='upper left', bbox_to_anchor=(1.2, 1.05))

plt.show()
fig.savefig("Variates_1.png")


''' Plot canonical loads for sequence features and insertion and deletion size classes.'''
X = np.load("input_sequence.npy")
Y = np.load("output_insertiondeletionsize.npy")
n_comp = 3

cca = CCA(n_components=n_comp)
cca.fit(X, Y)

xloadings = cca.x_loadings_
yloadings = cca.y_loadings_

ypos = np.arange(yloadings.shape[0])  # the label locations
width = 0.25  # the width of the bars
fig, axs = plt.subplots(2,1, figsize=(10,6))
fig.suptitle("Canonical loads for sequence features and classes for insertions and deletion size", fontsize=16)
plt.subplots_adjust(hspace=0.4)

for i in range(n_comp):
    axs[0].bar(range(xloadings.shape[0]), (xloadings[:,i]), alpha=0.4, linewidth=1)
axs[0].set_xlabel("Feature index")
axs[0].set_ylabel("Canonical load")
axs[0].set_title("Sequence features", fontsize=14)

yloadings = yloadings/(np.max(np.abs(yloadings), axis=0)[np.newaxis,:])

axs[1].bar(ypos - width, yloadings[:,0], width, alpha=0.5)
axs[1].bar(ypos, yloadings[:,1], width, alpha=0.5)
axs[1].bar(ypos + width, yloadings[:,2], width, alpha=0.5)
axs[1].set_xticks(ypos[::2], np.arange(yloadings.shape[0])[::2])
axs[1].set_xlabel("Class index")
axs[1].set_ylabel("Relative canonical load")
axs[1].set_title("Insertion and deletion size classes", fontsize=14)

trans = mtransforms.ScaledTranslation(-25 / 72, 7 / 72, fig.dpi_scale_trans)
axs[0].text(0.0, 1.0, "A)", transform=axs[0].transAxes + trans,
            fontsize='14', va='bottom')
axs[1].text(0.0, 1.0, "B)", transform=axs[1].transAxes + trans,
            fontsize='14', va='bottom')
fig.legend(["Variate 1", "Variate 2", "Variate 3"], loc='outside center right')

plt.show()
fig.savefig("Variates_2.png")

'''Plot canonical loads for MH features and deletion size classes only.'''
X = np.load("input_MHtracts.npy")
Y = np.load("output_deletionsize.npy")
n_comp = 2

cca = CCA(n_components=n_comp)
cca.fit(X, Y)

xloadings = cca.x_loadings_
yloadings = cca.y_loadings_


ypos = np.arange(yloadings.shape[0])  # the label locations
width = 0.35  # the width of the bars
fig, axs = plt.subplots(2,1, figsize=(10,6))
fig.suptitle("Canonical loads for microhomology features and classes for deletion size", fontsize=16)
plt.subplots_adjust(hspace=0.4)

for i in range(n_comp):
    axs[0].bar(range(xloadings.shape[0]), (xloadings[:,i]), alpha=0.4, linewidth=1)
axs[0].set_xlabel("Feature index")
axs[0].set_ylabel("Canonical load")
axs[0].set_title("Microhomology features", fontsize=14)

axs[1].bar(ypos - width/2, yloadings[:,0], width, alpha=0.5)
axs[1].bar(ypos + width/2, yloadings[:,1], width, alpha=0.5)
axs[1].set_xticks(ypos[::2], np.arange(yloadings.shape[0])[::2])
axs[1].set_xlabel("Class index")
axs[1].set_ylabel("Canonical load")
axs[1].set_title("Deletion size classes", fontsize=14)

trans = mtransforms.ScaledTranslation(-25 / 72, 7 / 72, fig.dpi_scale_trans)
axs[0].text(0.0, 1.0, "A)", transform=axs[0].transAxes + trans,
            fontsize='14', va='bottom')
axs[1].text(0.0, 1.0, "B)", transform=axs[1].transAxes + trans,
            fontsize='14', va='bottom')
fig.legend(["Variate 1", "Variate 2"], loc='outside center right')

plt.show()
fig.savefig("Variates_3.png")