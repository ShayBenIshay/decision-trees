import numpy as np
import pandas as pd
from KNN_algorithm import KNN
from DT import compute_accuracy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib


# get the data sets (train and test), and normalize the features values according to the test feature values
def data_normalization(train_data_table, test_data_table):
    for col in train_data_table.keys()[1:]:
        # get min and max values
        max_value = max(train_data_table[col].values)
        min_value = min(train_data_table[col].values)

        # normalize the train data
        train_data_table[col] = (train_data_table[col] - min_value) / (max_value - min_value)
        test_data_table[col] = (test_data_table[col] - min_value) / (max_value - min_value)


def question_8(train_data_table, test_data_table, k_val=9):
    if k_val == 9:
        print("*****************QUESTION8****************\n")
    else:
        print(f"****************QUESTION9, k = {k_val}****************\n")

    # build KNN classifier
    knn_classifier = KNN(train_data_set=train_data_table, k=k_val)

    # test
    class_vector = knn_classifier.KNN_get_classification_vec(test_data_table)

    # accuracy
    true_class_vector = test_data_table["diagnosis"].values  # ndarray
    accuracy = compute_accuracy(class_vector, true_class_vector)
    print(f"KNN model accuracy with k = {knn_classifier.k} is: {accuracy:.4f}")
    return accuracy


def question_9(train_data_table, test_data_table):
    accuracy_vec_hist = []
    # accuracy_vec_hist = [0.92, 0.934, 0.955876, 0.9865]
    for k in [1, 3, 9, 27]:
        # get the accuracy of KNN classification with each k value
        accuracy_vec_hist.append(question_8(train_data_table, test_data_table, k))

    # plot View accuracy as x dependent
    # ax = plt.figure()
    plt.subplot(211)
    plt.title("Classification Accuracy for KNN classifier")

    plt.scatter(1, accuracy_vec_hist[0])
    plt.scatter(3, accuracy_vec_hist[1])
    plt.scatter(9, accuracy_vec_hist[2])
    plt.scatter(27, accuracy_vec_hist[3])

    plt.xlabel('k')
    plt.ylabel('accuracy')

    plt.show()


def KNN_alg(train_data_table, test_data_table):
    # question_8(train_data_table, test_data_table)
    question_9(train_data_table, test_data_table)


def main():
    # load the train data
    train_data_table = pd.read_csv("train.csv")
    train_data_table.columns = train_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')', '')

    # load test train data
    test_data_table = pd.read_csv("test.csv")
    test_data_table.columns = test_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')', '')

    data_normalization(train_data_table, test_data_table)
    KNN_alg(train_data_table, test_data_table)


if __name__ =="__main__":
    main()
