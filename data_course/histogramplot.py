import pandas as pd
from tensorflow import keras
import Lindel, os, sys
import Lindel.Predictor
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

def openRawData(file):
    f = open(file, 'r')
    output = []
    for line in f:
        str_line = line.split()
        if 'mh_design_1' in str_line or 'mh_design_2' in str_line or 'mh_design_3' in str_line:
            output.append(str_line[0])
    
    return output

raw_data = openRawData('data_course/algient_NHEJ_guides_final.txt')

matrix1 = pkl.load(open('data_course/NHEJ_rep1_final_matrix.pkl','rb'))
matrix2 = pkl.load(open('data_course/NHEJ_rep2_final_matrix.pkl','rb'))
matrix3 = pkl.load(open('data_course/NHEJ_rep3_final_matrix.pkl','rb'))


test_data_np = test_data.to_numpy()

# test_long_sequence = []

# for test_data_point in test_data_np:
#     test_seq = test_data_point[0]
#     for big_string in raw_data:
#         position = big_string.find(test_seq)
#         if position != -1:
#             start = position + 17 - 30
#             end_position = position + 17 + 30 
#             bp_60_seq = big_string[start:end_position]
#             test_long_sequence.append(bp_60_seq)

# print(len(test_long_sequence))
        
sixty_bp = {}

pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
for test_data_point in test_data_np:
    test = test_data_point[0]
    for raw_data_point in matrix1:
        target_raw_data = raw_data_point[3]
        if test == target_raw_data:
            guide = raw_data_point[0][27:47]
            if guide == test:
                print('yes')
            if raw_data_point[0][47:50] not in pam:
                print('here')
                continue
            # encoding the raw data into binary features
            indels = Lindel.Predictor.gen_indel(guide,27+18) 
            input_indel = Lindel.Predictor.onehotencoder(guide)
            label,rev_index,features,frame_shift = prerequesites
            input_del = np.concatenate((Lindel.Predictor.create_feature_array(features,indels),input_indel),axis=None) # 3033 * 1

            if np.array_equal(np.array(input_del), test_data_point[1:3034]):
                print("here!")
                if len(sixty_bp[test]) == 0:
                    sixty_bp[test] = [raw_data_point[0]]
                else:
                    sixty_bp[test].append(raw_data_point[0])
            # print(filtered_data_test)
            
    break

# MSE = []

# def mse(x, y):
#     return ((x-y)**2).mean()

# NOTES: the gen_prediction function needs a 60 bp input (in contrast to the 20 bp test that we are given). It needs this for gen_indel and input_del vectors
# to take into account the microhomology. Why do we need that for prediction? How can we generate a prediction with only 20 bp input? Are they using another data set perhaps?

# def gen_prediction_adapted(seq,wb,prereq):
#     '''generate the prediction for all classes, redundant classes will be combined'''
#     pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
#     guide = seq[13:33]
