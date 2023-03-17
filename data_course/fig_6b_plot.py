import pandas as pd
from tensorflow import keras
import Lindel, os, sys
from Lindel.Predictor import * 
import pickle as pkl
import numpy as np 

test_data = pd.read_csv("data_course\Lindel_test.txt", sep='\t')
model_del = keras.models.load_model('data_course\L1_del.h5')
model_del_ins_ratio = keras.models.load_model('data_course\L2_indel.h5')
model_ins = keras.models.load_model('data_course\L2_ins.h5')

model_del_array_weights = model_del.trainable_weights[0].numpy()
model_del_array_bias = model_del.trainable_weights[1].numpy()
model_ratio_array_weights = model_del_ins_ratio.trainable_weights[0].numpy()
model_ratio_array_biases = model_del_ins_ratio.trainable_weights[1].numpy()
model_ins_array_weights = model_ins.trainable_weights[0].numpy()
model_ins_array_biases = model_ins.trainable_weights[1].numpy()

weights_biases = np.array([model_ratio_array_weights, model_ratio_array_biases, model_del_array_weights, model_del_array_bias, model_ins_array_weights, model_ins_array_biases])

# y = test_data.iloc[3, 3034:] 
# print(y)
prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0],'model_prereq.pkl'),'rb'))
MSE = []

def mse(x, y):
    return ((x-y)**2).mean()

# NOTES: the gen_prediction function needs a 60 bp input (in contrast to the 20 bp test that we are given). It needs this for gen_indel and input_del vectors
# to take into account the microhomology. Why do we need that for prediction? How can we generate a prediction with only 20 bp input? Are they using another data set perhaps?

for indx, row in test_data.iterrows():
    target_value = np.array(row[3034:])
    seq = row[0]
    y_prediction = gen_prediction(seq, weights_biases, prerequesites)
    if isinstance(y_prediction, str):
        continue
    print(y_prediction)
    break
    a = mse(y_prediction[0], target_value)
    MSE.append(a)


# filename = sys.argv[2]
# try:
#     y_hat, fs = gen_prediction(seq,weights,prerequesites)
#     filename += '_fs=' + str(round(fs,3))+'.txt'
#     rev_index = prerequesites[1]
#     pred_freq = {}
#     for i in range(len(y_hat)):
#         if y_hat[i]!=0:
#             pred_freq[rev_index[i]] = y_hat[i]
#     pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1],reverse=True)
#     write_file(seq,pred_sorted,pred_freq,filename)
# except ValueError:
#     print ('Error: No PAM sequence is identified.Please check your sequence and try again')
