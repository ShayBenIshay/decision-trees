from KNN import data_normalization
import pandas as pd
from ID3_algorithm import ID3Alg
from Temp_classes import *
from KNN_algorithm import KNN
from DT import compute_accuracy
from TDIDT_algorithm import majority_class


def question_11(train_data_table, test_data_table, x, k):
    # *****build T9 - epsilon Decision tree******
    # get list of features names, convert to Feature array
    features_list = list(train_data_table.columns)[1:]  # reduce the diagnosis column
    features = np.array([Feature(name=f_name) for f_name in features_list])    # defines an array which contains Feature's type objects

    # use the train_data_table to build an ID3 classifier (with pruning, x=9)
    id3_obj = ID3Alg(train_data_table, features, x)
    T9_decision_tree = id3_obj.get_DecisionTree()   # build the

    # compute epsilon vector
    epsilon_vec = []
    for key in train_data_table.keys():
        v_i = 0.1 * np.std(train_data_table[key].values)
        epsilon_vec.append(v_i)
    # create dict with key:(std_val*0.1)
    epsilon_vec = dict(zip(train_data_table.keys().tolist(), epsilon_vec))

    # ***** get the relevant leaves for each, and classify with KNN. k=9*******
    class_vector = [0] * int(test_data_table.shape[0])
    for i, obj in test_data_table.iterrows():
        # get relevant leaves according to epsilon-decision-rule for obj
        leaves_lst = id3_obj.DT_Epsilon_classify_get_examples(dict(zip(train_data_table.keys().tolist(), obj)), T9_decision_tree, epsilon_vec)

        # get the examples of each leave
        # leaves_examples_lst = pd.DataFrame([item for sublist in leaves_lst for item in sublist])
        leaves_examples_lst = leaves_lst
        # if there are less than k examples, decide according to the majority
        if len(leaves_examples_lst) < k:
            obj_class, _ = majority_class(leaves_examples_lst)
            class_vector[i] = obj_class

        # classify with KNN. train_set = leaves_examples_lst, k=9
        else:
            knn_classifier = KNN(train_data_set=leaves_examples_lst, k=k)
            # test
            obj_class = knn_classifier.KNN_classify(obj)
            class_vector[i] = obj_class


    # accuracy
    true_class_vector = test_data_table["diagnosis"].values  # ndarray
    accuracy = compute_accuracy(class_vector, true_class_vector)
    print(f"KNN_Epsilon model accuracy with k = {k}, eps = {epsilon_vec} is: {accuracy:.4f}")
    return accuracy


def KNN_eps(train_data_table, test_data_table):
    question_11(train_data_table, test_data_table, x=9, k=9)


def main():
    # load the train data
    train_data_table = pd.read_csv("train.csv")
    train_data_table.columns = train_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')', '')

    # load test train data
    test_data_table = pd.read_csv("test.csv")
    test_data_table.columns = test_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')', '')

    data_normalization(train_data_table, test_data_table)
    KNN_eps(train_data_table, test_data_table)


if __name__ =="__main__":
    main()