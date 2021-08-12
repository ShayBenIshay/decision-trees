import time
import pandas as pd
from ID3_algorithm import ID3Alg
from Temp_classes import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def compute_accuracy(class_vector, true_class_vector):
    true_positive = 0
    true_negative = 0
    N = len(class_vector)
    # evaluate the classifier accuracy. compare th true classification to our classifier results
    for i in range(N):
        if class_vector[i] == 1 and true_class_vector[i] == 1:
            true_positive += 1
        elif class_vector[i] == 0 and true_class_vector[i] == 0:
            true_negative += 1
    accuracy = (true_negative + true_positive) / N
    return accuracy


def question_2(train_data_table, test_data_table, x=0):
    if x == 0:
        print("****************QUESTION2****************\n")
    else:
        print(f"****************QUESTION3, x = {x}****************\n")

    ############################ train #############################
    print(f"Start training...")
    start_time = time.time()

    # print(train_data_table.keys())
    # train_data_table.plot.area()

    # get list of features names, convert to Feature array
    features_list = list(train_data_table.columns)[1:]  # reduce the diagnosis column
    features = np.array([Feature(name=f_name) for f_name in features_list])    # defines an array which contains Feature's type objects

    # use the train_data_table to build an ID3 classifier
    id3_obj = ID3Alg(train_data_table, features, x)
    id3_decision_tree = id3_obj.get_DecisionTree()   # build the

    end_time = time.time()
    print(f"Training had finished. Decision tree had gain. Time: {end_time - start_time}\n\n")

    ############################ test #############################
    print(f"Start testing...")
    start_time = time.time()

    # get classifications for test_data
    class_vector = [0] * int(test_data_table.size / 31)
    for i, obj in test_data_table.iterrows():
            class_vector[i] = id3_obj.DT_classify(dict(zip(train_data_table.keys().tolist(), obj)), id3_decision_tree)

    end_time = time.time()
    print(f"Testing had finished. Time: {end_time - start_time}\n\n")

    ###################### accuracy computation ####################
    true_class_vector = test_data_table["diagnosis"].values  # ndarray
    accuracy = compute_accuracy(class_vector, true_class_vector)
    print(f"ID3 Decision Tree accuracy: {accuracy:.4f}\n\n")

    # in this case, the function is called from question3, and the accuracy is used for plot
    if x:
        return accuracy
    return None


def question_3(train_data_table, test_data_table):
    accuracy_vec_hist = []
    for x in [3, 9, 27]:
        # build the decision tree
        accuracy_vec_hist.append(question_2(train_data_table, test_data_table, x))

    # plot View accuracy as x dependent
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot([3, 9, 27], accuracy_vec_hist)
    plt.xlabel('x')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.show()


def ID3(train_data_table, test_data_table):
    question_2(train_data_table, test_data_table)
    question_3(train_data_table, test_data_table)


# def main():
#     # load the train data
#     train_data_table = pd.read_csv("train.csv")
#     train_data_table.columns = train_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
#                                                                                                                   '').str.replace(')', '')
#     # load test train data
#     test_data_table = pd.read_csv("test.csv")
#     test_data_table.columns = test_data_table.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
#                                                                                                                 '').str.replace(')', '')
#     # ID3(train_data_table, test_data_table) # question2, question3
#
#
# if __name__ =="__main__":
#     main()

# def test():
#     (X_train, X_test, y_train):
#     # Decision tree with entropy
#     clf_entropy = DecisionTreeClassifier(criterion="entropy")
#
#     # Performing training
#     clf_entropy.fit(X_train, y_train)
#     return clf_entropy

