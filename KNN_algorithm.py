import math
import pandas as pd
import functools


class KNN:
    def __init__(self, train_data_set, k=1):
        self.k = k
        self.train_data_set: pd.DataFrame = train_data_set

    def KNN_classify(self, obj_to_classify):
        # get all distances between objects in train_data_set to obj_to_classify
        # distance list contains tuples of (class, distance)
        dist_and_class_list = []
        for i, obj in self.train_data_set.iterrows():
            difference_between_obj = obj[1:] - obj_to_classify[1:]
            distance = math.sqrt(sum(difference_between_obj ** 2))
            obj_class = obj[0]
            dist_and_class_list.append((obj_class, distance))

        # sort distance_list
        dist_and_class_list.sort(key=lambda tup: tup[1])

        # k_distance_list = k nearest neighbors
        k_dist_and_class_list = dist_and_class_list[:self.k]
        k_diagnosis_list = [tup[0] for tup in k_dist_and_class_list]

        # compute the classification of most neighbours
        positive_class = functools.reduce(lambda a, b: a + b, k_diagnosis_list)
        negative_class = self.k - positive_class

        # return the majority class of k neighbors
        return False if negative_class > positive_class else True

    # returns classification vector for the test_data_table
    def KNN_get_classification_vec(self, test_data_table) -> list:
        class_vector = []
        for i, obj in test_data_table.iterrows():
            class_vector.append(self.KNN_classify(obj))

        return class_vector
