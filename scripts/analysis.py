import pandas as pd
import numpy as np
from ast import literal_eval
import pickle as pkl
import utils
import Lindel
from Lindel import Predictor
import os

def find_seq_and_replace(data_set, test_seq, sequences_60bp, test_data_point):
    for seq in data_set:
        pos = seq.find(test_seq, 39)
        if pos >= 0:
            start_pos = pos + 17 - 30
            end_pos = pos + 17 + 30
            seq_60_bp = seq[start_pos:end_pos]
            sequences_60bp.append(seq_60_bp)
            test_data_point[0] = seq_60_bp
            return True

    return False


def get_test_set_60bp():
    test_data = []
    with open('data_course/Lindel_test.txt') as f:
        lines = f.readlines()
        for l in lines:
            line_arr = l.split()
            row = [line_arr[0]]
            for token in line_arr[1:]:
                row.append(float(token))
            test_data.append(row)

    seventy_k, homing_design, mh1_200bp, _, _, _ = utils.read_200bp_sequences(
        'data_course/algient_NHEJ_guides_final.txt')
    sequences_60bp = []

    for test_data_point in test_data:
        test_seq = test_data_point[0]
        if find_seq_and_replace(seventy_k, test_seq, sequences_60bp, test_data_point):
            continue
        elif find_seq_and_replace(homing_design, test_seq, sequences_60bp, test_data_point):
            continue
        else:
            find_seq_and_replace(mh1_200bp, test_seq, sequences_60bp, test_data_point)

    # print("\n-------------\n")
    # print(test_data[0][0])
    # print(len(test_data[0][0]))
    # print(len(sequences_60bp))  # 440 data points in Lindel_test

    return test_data


def mse(x, y):
    return ((x-y)**2).mean()


if __name__ == '__main__':
    get_test_set_60bp()

    test_data = get_test_set_60bp()
    model_del_array_weights, model_del_array_biases, model_ratio_array_weights, model_ratio_array_biases, model_ins_array_weights, model_ins_array_biases = utils.get_weights_biases()
    weights_biases = [model_ratio_array_weights, model_ratio_array_biases, model_del_array_weights, model_del_array_biases,
         model_ins_array_weights, model_ins_array_biases]

    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0],'model_prereq.pkl'),'rb'))

    predictions = []

    for test_point in test_data:
        test_seq = test_point[0]
        frequencies_hat, c = Predictor.gen_prediction(test_seq, weights_biases, prerequesites)
        predictions.append(frequencies_hat)

    np.save('lindel_output', np.array(predictions))



