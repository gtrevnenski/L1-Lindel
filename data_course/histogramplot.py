import pandas as pd
from tensorflow import keras
import Lindel, os, sys
import Lindel.Predictor
import pickle as pkl
import numpy as np 

# Load data
test_data = pd.read_csv("data_course\Lindel_test.txt", sep='\t')
model_del = keras.models.load_model('data_course\L1_del.h5')
model_del_ins_ratio = keras.models.load_model('data_course\L2_indel.h5')
model_ins = keras.models.load_model('data_course\L2_ins.h5')

# Load trained model weights
model_del_array_weights = model_del.trainable_weights[0].numpy()
model_del_array_bias = model_del.trainable_weights[1].numpy()
model_ratio_array_weights = model_del_ins_ratio.trainable_weights[0].numpy()
model_ratio_array_biases = model_del_ins_ratio.trainable_weights[1].numpy()
model_ins_array_weights = model_ins.trainable_weights[0].numpy()
model_ins_array_biases = model_ins.trainable_weights[1].numpy()

weights_biases = np.array([model_ratio_array_weights, model_ratio_array_biases, model_del_array_weights, model_del_array_bias, model_ins_array_weights, model_ins_array_biases])

prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0],'model_prereq.pkl'),'rb'))

def openRawData(file):
    """Read raw data to find longer sequences corresponding to target sequences.

    Args:
        file (str): filename

    Returns:
        str[]: longer sequences
    """
    f = open(file, 'r')
    output = []
    for line in f:
        str_line = line.split()
        if 'mh_design_1' in str_line or 'mh_design_2' in str_line or 'mh_design_3' in str_line:
            output.append(str_line[0])
    
    return output

raw_data = openRawData('Icourse/algient_NHEJ_guides_final.txt')

matrix1 = pkl.load(open('data_course/NHEJ_rep1_final_matrix.pkl','rb'))
matrix2 = pkl.load(open('data_course/NHEJ_rep2_final_matrix.pkl','rb'))
matrix3 = pkl.load(open('data_course/NHEJ_rep3_final_matrix.pkl','rb'))


test_data_np = test_data.to_numpy()
        
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
            
    break