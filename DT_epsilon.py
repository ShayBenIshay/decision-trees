import time
import pandas as pd
from ID3_algorithm import ID3Alg
from TDIDT_algorithm import TDIDT_algorithm
from Temp_classes import *
import numpy as np
from DT import compute_accuracy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def question_6(train_data_table, test_data_table, x=9):
    ############################ train #############################
    print(f"Start training...")
    start_time = time.time()

    # compute epsilon vector
    # epsilon_vec = [0]*30
    epsilon_vec =[]
    # for key in train_data_table.keys()[1]:
    for key in train_data_table.keys():
        v_i = 0.1 * np.std(train_data_table[key].values)
        epsilon_vec.append(v_i)
    # create dict with key:(std_val*0.1)
    epsilon_vec = dict(zip(train_data_table.keys().tolist(), epsilon_vec))

    # get list of features names, convert to Feature array
    features_list = list(train_data_table.columns)[1:]  # reduce the diagnosis column
    features = np.array([Feature(name=f_name) for f_name in features_list])    # defines an array which contains Feature's type objects

    # use the train_data_table to build an ID3 classifier
    id3_obj = ID3Alg(train_data_table, features, x)
    id3_dt = id3_obj.get_DecisionTree()   # build the
    end_time = time.time()
    print(f"Training had finished. Decision tree had gain. Time: {end_time - start_time}\n\n")

    ############################ test #############################
    print(f"Start testing...")
    start_time = time.time()


    # get classifications for test_data
    class_vector = [0] * int(test_data_table.size / 31)

    # calculate epsilon to 0.1 * v (eps_i = 0.1*v_i)
    for i, obj in test_data_table.iterrows():
        dict_obj_to_class = dict(zip(train_data_table.keys().tolist(), obj))
        class_vector[i] = id3_obj.DT_Epsilon_classify(dict_obj_to_class, id3_dt, epsilon_vec)

    end_time = time.time()
    print(f"Testing had finished. Time: {end_time - start_time}\n\n")

    ###################### accuracy computation ####################
    true_class_vector = test_data_table["diagnosis"].values  # ndarray
    accuracy = compute_accuracy(class_vector, true_class_vector)
    print(f"ID3 decision tree with epsilon-rule accuracy: {accuracy:.4f}\n\n")

    # in this case, the function is called from question3, and the accuracy is used for plot
    return accuracy


def Epsilon(train_data_table, test_data_table):
    question_6(train_data_table, test_data_table)


def main():
    # load the train data
    train_data_table = pd.read_csv("train.csv")
    train_data_table.columns = train_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
                                                                                                                  '').str.replace(')', '')
    # load test train data
    test_data_table = pd.read_csv("test.csv")
    test_data_table.columns = test_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
                                                                                                                '').str.replace(')', '')
    Epsilon(train_data_table, test_data_table) # question6


if __name__ =="__main__":
    main()