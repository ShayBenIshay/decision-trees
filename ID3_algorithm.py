import math
import pandas as pd
from TDIDT_algorithm import TDIDT_algorithm, majority_class
from Temp_classes import *


class ID3Alg(TDIDT_algorithm):
    def __init__(self, train_data_set, features, x=0):
        default_val, _ = majority_class(train_data_set)
        TDIDT_algorithm.__init__(self, train_data_set, features, default_val, self.select_feature, x)

    # return the entropy of the data_set
    def get_entropy(self, data_set):
        entropy = 0
        for c in range(2):
            if (data_set.shape[0]) == 0:
                p_c = 0
            else:
                p_c = len([data for data in data_set['diagnosis'] if data == c]) /(data_set.shape[0])
                # (data_set[data_set.diagnosis == c]).size / data_set.size
            entropy -= p_c * (0 if p_c == 0 else math.log(p_c, 2))
        return entropy

    def max_IG(self, discrete_features, examples) -> (DiscreteFeature, float):
        f_with_max_ig = None
        max_ig = -1

        examples_entropy = self.get_entropy(examples)
        for discrete_f in discrete_features:
            # compute the entropy of leaves created by splitting by f feature
            original_continues_f = discrete_f.original_f_name

            examples_f_val_0 = examples.query(f'({original_continues_f} <= {float(discrete_f.name)}) == 1')
            all__examples_below = examples_f_val_0.query(f'{original_continues_f} == {float(discrete_f.name)}').shape[0]

            examples_f_val_1 = examples.query(f'({original_continues_f} <= {float(discrete_f.name)}) == 0')
            # all__examples_above = examples_f_val_1.query(f'{original_continues_f} == {float(discrete_f.name)}').size

            if all__examples_below == 0:# or all__examples_above == 0:
                new_threshold = (examples_f_val_0.max()[original_continues_f] + examples_f_val_1.min()[
                    original_continues_f]) / 2
                discrete_f.name = str(new_threshold)

            ex_0_entropy = (examples_f_val_0.size / examples.size) * self.get_entropy(examples_f_val_0) if examples_f_val_0.shape[0] else 0
            ex_1_entropy = (examples_f_val_1.size / examples.size) * self.get_entropy(examples_f_val_1) if examples_f_val_1.shape[0] else 0
            examples_i_entropy_weighted_sum = ex_0_entropy + ex_1_entropy

            ig_f = examples_entropy - examples_i_entropy_weighted_sum
            if max_ig < ig_f:
                max_ig = ig_f
                f_with_max_ig = discrete_f

        return f_with_max_ig, max_ig

    # get selected_f which is a continues feature,
    # and return an
    # appropriate list of k-1 binary features- which are defined by threshold values
    def compute_discrete_features(self, selected_f, data_set) -> np.array:
        # get the f(sample) for each sample, and sort the values
        sample_f_vals = np.sort(data_set[selected_f.name].values)

        # compute k-1 threshold values, define k-1 binary features
        threshold_features = []
        for val1, val_2 in zip(sample_f_vals[:-1], sample_f_vals[1:]):
            t_j = (val1 + val_2) / 2
            threshold_features.append(
                DiscreteFeature(name=str(t_j), vals=[0, 1], original_f_name=selected_f.name))

        return np.array(threshold_features)

    def select_feature(self, features, data_set):
        f_with_max_ig = None
        max_ig = float('-inf')

        for f in features:
            new_discrete_features = self.compute_discrete_features(f, data_set)
            discrete_f, f_ig = self.max_IG(new_discrete_features, data_set)

            if max_ig < f_ig:
                f_with_max_ig = discrete_f
                max_ig = f_ig

        return f_with_max_ig  # discrete feature

    def get_DecisionTree(self):
        return super().get_DecisionTree()
