from abc import ABC, abstractmethod
from Temp_classes import *
import pandas as pd
import numpy as np
import functools


# return tuple:
# the majority of classifying value for the given data_set
# the number of positive classifications
def majority_class(data_set: pd.DataFrame) -> (bool, int):
    data_classes = data_set["diagnosis"].values.tolist()    # get the diagnosis column in the table
    positive_class = functools.reduce(lambda a, b: a+b, data_classes)
    negative_class = len(data_classes) - positive_class
    return (False if negative_class > positive_class else True), positive_class


class TDIDT_algorithm(ABC):
    def __init__(self, train_data_set, features, default_val, select_feature, x=0):
        self.train_data_set: pd.DataFrame = train_data_set
        self.features: np.array = features
        self.default_val = default_val
        self.select_feature = select_feature    # function
        self.output_classifier = None    # the founded DecisionTree
        self.x = x
        self.discrete_features = []

    def get_DecisionTree(self):
        return self.TDIDT(self.train_data_set, self.features, self.default_val, self.select_feature,[])

    @abstractmethod
    def select_feature(self, features, data_set) -> Feature:
        pass

    def TDIDT(self, train_data_set: pd.DataFrame, features, default_val, select_feature,features_discrete_vals_list):
        if train_data_set.size == 0:
            return DecisionTree(None, [], default_val), features_discrete_vals_list

        c, positive_num = majority_class(train_data_set)  # get the default_val classification
        # negative_num = len(train_data_set)-positive_num

        if train_data_set.shape[0] <= self.x:
            # if positive_num*2 == train_data_set.shape[0]:
            return DecisionTree(None, [], c, train_data_set)

        same_class = (positive_num == train_data_set.shape[0]) or (positive_num == 0)
        if same_class or len(features) == 0:
            return DecisionTree(None, [], c, train_data_set), features_discrete_vals_list

        selected_f: DiscreteFeature = select_feature(features, train_data_set)     # feature to evaluate for split
        sub_trees = []
        original_continues_f = selected_f.original_f_name
        for val in selected_f.vals:
            features_discrete_vals_list.append(val)
            new_train_data_set = train_data_set.query(f'({original_continues_f} > {float(selected_f.name)}) == {val}')
            DT = self.TDIDT(new_train_data_set, features, c, select_feature,features_discrete_vals_list)
            sub_trees.append((val, DT))

        return DecisionTree(original_continues_f+":"+selected_f.name, sub_trees, c, train_data_set), features_discrete_vals_list

    def DT_classify(self, obj, tree: DecisionTree):
        if not tree.children:
            return tree.classification

        for child_f_val, child_subtree in tree.children:
            feature, threshold = tree.feature.split(':')
            obj_value = not (obj[feature] <= float(threshold))
            if obj_value == child_f_val:
                return self.DT_classify(obj, child_subtree)

    def DT_Epsilon_recursive(self, obj, tree, epsilon, leaves_list):
        if not tree.children:
            return []

        feature, threshold = tree.feature.split(':')
        for child_f_val, child_subtree in tree.children:
            # if not child_subtree.children:
                # this child is a leaf
                # return leaves_list.append(child_subtree)
            # if this condition is true - it will be for booth childrens
            if abs(obj[feature] - float(threshold)) <= epsilon[feature]:
                return leaves_list.append(self.DT_Epsilon_recursive(obj, child_subtree, epsilon, []))
            else:
                obj_value = not (obj[feature] <= float(threshold))
                if obj_value == child_f_val:
                    return leaves_list.append(self.DT_Epsilon_recursive(obj, child_subtree, epsilon, []))

    def DT_Epsilon_classify(self, obj, tree: DecisionTree, epsilon):
        leaves_list = self.DT_Epsilon_recursive(obj, tree, epsilon, [])
        leaves_examples_lst = pd.DataFrame([example for data_frame in leaves_list for example in data_frame])
        data_classes = leaves_examples_lst["diagnosis"].values.tolist()  # get the diagnosis column in the table

        negative_num = sum(data_classes)
        positive_num = leaves_examples_lst.shape[0] - negative_num
        return positive_num > negative_num  # return the classification. if equal return's True.
