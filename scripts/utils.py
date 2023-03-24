import pandas as pd
from tensorflow import keras


def get_weights_biases():
    model_del = keras.models.load_model('../data_course/L1_del.h5')
    model_del_ins_ratio = keras.models.load_model('../data_course/L2_indel.h5')
    model_ins = keras.models.load_model('../data_course/L2_ins.h5')

    model_del_array_weights = model_del.trainable_weights[0].numpy()
    model_del_array_biases = model_del.trainable_weights[1].numpy()
    model_ratio_array_weights = model_del_ins_ratio.trainable_weights[0].numpy()
    model_ratio_array_biases = model_del_ins_ratio.trainable_weights[1].numpy()
    model_ins_array_weights = model_ins.trainable_weights[0].numpy()
    model_ins_array_biases = model_ins.trainable_weights[1].numpy()

    return model_del_array_weights, model_del_array_biases, model_ratio_array_weights, model_ratio_array_biases, model_ins_array_weights, model_ins_array_biases


def read_200bp_sequences(file):
    f = open(file, 'r')
    seventy_k = []
    homing_design = []
    mh_design1 = []
    mh_design2 = []
    mh_design3 = []
    may_data = []
    for line in f.readlines():
        str_line = line.split()
        if '70k' in str_line:
            seventy_k.append(str_line[0])
        elif 'homing_design' in str_line:
            homing_design.append(str_line[0])
        elif 'mh_design_1' in str_line:
            mh_design1.append(str_line[0])
        elif 'mh_design_2' in str_line:
            mh_design2.append(str_line[0])
        elif 'mh_design_3' in str_line:
            mh_design3.append(str_line[0])
        elif 'Maydata' in str_line:
            may_data.append(str_line[0])

    return seventy_k, homing_design, mh_design1, mh_design2, mh_design3, may_data
