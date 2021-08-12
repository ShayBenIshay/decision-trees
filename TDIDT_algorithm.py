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

    def get_DecisionTree(self):
        return self.TDIDT(self.train_data_set, self.features, self.default_val, self.select_feature)

    @abstractmethod
    def select_feature(self, features, data_set) -> Feature:
        pass

    def TDIDT(self, train_data_set: pd.DataFrame, features, default_val, select_feature) -> DecisionTree:
        if train_data_set.size == 0:
            return DecisionTree(None, [], default_val, train_data_set)
        c, positive_num = majority_class(train_data_set)  # get the default_val classification
        negative_num = len(train_data_set)-positive_num

        if train_data_set.shape[0] <= self.x:
            # if positive_num*2 == train_data_set.shape[0]:
            return DecisionTree(None, [], c, train_data_set)

        same_class = (positive_num == train_data_set.shape[0]) or (positive_num == 0)
        if same_class or len(features) == 0:
            return DecisionTree(None, [], c, train_data_set)

        selected_f: DiscreteFeature = select_feature(features, train_data_set)     # feature to evaluate for split

        sub_trees = []
        original_continues_f = selected_f.original_f_name
        for val in selected_f.vals:
            new_train_data_set = train_data_set.query(f'({original_continues_f} > {float(selected_f.name)}) == {val}')
            DT = self.TDIDT(new_train_data_set, features, c, select_feature)
            sub_trees.append((val, DT))

        return DecisionTree(original_continues_f+":"+selected_f.name, sub_trees, c, train_data_set)

    def DT_classify(self, obj, tree: DecisionTree):
        if not tree.children:
            return tree.classification

        feature, threshold = tree.feature.split(':')
        for child_f_val, child_subtree in tree.children:
            obj_value = not (obj[feature] <= float(threshold))
            if obj_value == child_f_val:
                return self.DT_classify(obj, child_subtree)

    # return all of the leaves' examples which are relevant for obj
    def DT_Epsilon_classify_get_examples(self, obj, tree: DecisionTree, epsilon):
        if not tree.children:
            return tree.examples

        feature, threshold = tree.feature.split(':')
        # all_leaves_flag = True if abs(obj[feature] - threshold) <= epsilon[feature] else False
        all_leaves_flag = True if abs(obj[feature] - float(threshold)) <= epsilon[feature] else False

        examples_of_leaves = pd.DataFrame([])
        for child_f_val, child_subtree in tree.children:
            obj_value = not (obj[feature] <= float(threshold))
            if all_leaves_flag:
                examples_of_leaves = pd.concat([examples_of_leaves, self.DT_Epsilon_classify_get_examples(obj, child_subtree, epsilon)])
            elif obj_value == child_f_val:
                examples_of_leaves = self.DT_Epsilon_classify_get_examples(obj, child_subtree, epsilon)

        return examples_of_leaves

    # return the classification of obj according to the majority of leaves' relevant examples
    def DT_Epsilon_classify(self, obj, tree: DecisionTree, epsilon):
        all_relevant_examples = self.DT_Epsilon_classify_get_examples(obj, tree, epsilon)
        c,_ = majority_class(all_relevant_examples)
        return c;
