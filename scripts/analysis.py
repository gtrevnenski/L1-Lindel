from tensorflow import keras
import pandas as pd
import numpy as np
from ast import literal_eval
import pickle as pkl
import utils


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
    with open('../data_course/Lindel_test.txt') as f:
        lines = f.readlines()
        for l in lines:
            line_arr = l.split()
            row = [line_arr[0]]
            for token in line_arr[1:]:
                row.append(float(token))
            test_data.append(row)


    seventy_k, homing_design, mh1_200bp, _, _, _ = utils.read_200bp_sequences('../data_course/algient_NHEJ_guides_final.txt')
    combined_set = seventy_k + mh1_200bp
    print(len(combined_set))
    sequences_60bp = []

    for test_data_point in test_data:
        test_seq = test_data_point[0]
        if find_seq_and_replace(seventy_k, test_seq, sequences_60bp, test_data_point):
            continue
        elif find_seq_and_replace(homing_design, test_seq, sequences_60bp, test_data_point):
            continue
        else:
            find_seq_and_replace(mh1_200bp, test_seq, sequences_60bp, test_data_point)

    print("\n-------------\n")
    print(test_data[0][0])
    print(len(test_data[0][0]))
    print(len(sequences_60bp))#440 data points in Lindel_test


if __name__ == '__main__':
    get_test_set_60bp()
