import h5py
import numpy as np
import keras
filename = "L1_ins.h5"

# mod = keras.models.load_model(filename)
# print(mod.get_weights())

lambdas = 10 ** np.arange(-10, -1, 0.1)
for file in ["mse_l1_indel.npy", "mse_l2_indel.npy",
             "mse_l1_ins.npy", "mse_l2_ins.npy",
             "mse_l1_del.npy", "mse_l2_del.npy"]:
    data = np.load(file)
    print(min(data), np.argmin(data), lambdas[np.argmin(data)])

#indel ratio:   l2 has smaller error for lambda 0.010000
#insertion:     l1 has smaller error for lambda 0.00100
#deletion:      l1 has smaller error for lambda 0.000200

"""
Indel
0.03047848088317288 69 0.0007943282347242366
0.029195189104684775 80 0.009999999999999346

Ins
0.0071847765 70 0.0009999999999999428
0.0072082356 61 0.00012589254117941043

Del
0.00019374027 63 0.00019952623149687768
0.00023700343 69 0.0007943282347242366
"""