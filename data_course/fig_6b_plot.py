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

prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0],'model_prereq.pkl'),'rb'))

matrix1 = pkl.load(open('data_course/NHEJ_rep1_final_matrix.pkl','rb'))
matrix2 = pkl.load(open('data_course/NHEJ_rep2_final_matrix.pkl','rb'))
matrix3 = pkl.load(open('data_course/NHEJ_rep3_final_matrix.pkl','rb'))

example_data_point = matrix1[0]
test_data_np = test_data.to_numpy()
filtered_data_test = {}

for target_data_point in test_data_np:
    target = target_data_point[0]
    for raw_data_point in matrix1:
        target_raw_data = raw_data_point[2]
        if target == target_raw_data:
            # encoding the raw data into binary features
            indels = gen_indel(target,30) 
            input_indel = onehotencoder(target)
            label,rev_index,features,frame_shift = prerequesites
            input_del = np.concatenate((create_feature_array(features,indels),input_indel),axis=None)

            if np.array_equal(np.array(input_del), target_data_point[1:3034]):
                print("here!")
                if len(filtered_data_test[target]) == 0:
                    filtered_data_test[target] = [target_data_point[0]]
                else:
                    filtered_data_test[target].append(target_data_point[0])
            # print(filtered_data_test)
            
    break

MSE = []

def mse(x, y):
    return ((x-y)**2).mean()

# NOTES: the gen_prediction function needs a 60 bp input (in contrast to the 20 bp test that we are given). It needs this for gen_indel and input_del vectors
# to take into account the microhomology. Why do we need that for prediction? How can we generate a prediction with only 20 bp input? Are they using another data set perhaps?

# def gen_prediction_adapted(seq,wb,prereq):
#     '''generate the prediction for all classes, redundant classes will be combined'''
#     pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
#     guide = seq[13:33]

#     print("\n----------------------\n")
#     print(guide)
#     print("\n----------------------\n")

#     # if seq[33:36] not in pam:
#     #     return ('Error: No PAM sequence is identified.')
#     w1,b1,w2,b2,w3,b3 = wb
#     label,rev_index,features,frame_shift = prereq
#     indels = gen_indel(seq,30) 
#     print("\n----------------------\n")
#     print(indels)
#     print(len(indels))
#     print("\n----------------------\n")
#     input_indel = onehotencoder(guide)
#     print(len(input_indel))
#     print("\n----------------------\n")
#     print(input_indel) # 384 one hot encoded sequence features (single and dinucleotide features)
#     print("\n----------------------\n")
#     input_ins   = onehotencoder(guide[-6:])
#     print("\n----------------------\n")
#     print(input_ins)
#     print(len(input_ins)) # 104 features -> single and dinucleotide features for the last 6 nucleotides of the sequence
#     print("\n----------------------\n")
#     input_del   = np.concatenate((create_feature_array(features,indels),input_indel),axis=None)
#     print("\n----------------------\n")
#     print(input_del)
#     print(len(input_del)) # 3033 features in total 
#     print("\n----------------------\n")
#     cmax = gen_cmatrix(indels,label) # combine redundant classes
#     print(cmax)
#     dratio, insratio = softmax(np.dot(input_indel,w1)+b1)
#     ds  = softmax(np.dot(input_del,w2)+b2)
#     ins = softmax(np.dot(input_ins,w3)+b3)
#     y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None) * cmax
#     return (y_hat,np.dot(y_hat,frame_shift))


# for indx, row in test_data.iterrows():
#     target_value = np.array(row[3034:])
#     seq = row[0]
#     y_prediction = gen_prediction(seq, weights_biases, prerequesites)
#     if isinstance(y_prediction, str):
#         continue
#     print(y_prediction)
#     break
#     a = mse(y_prediction[0], target_value)
#     MSE.append(a)
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
