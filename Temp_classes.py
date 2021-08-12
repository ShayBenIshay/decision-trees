import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, feature, children, classification, examples=np.array([])):
        self.feature = feature
        self.children = children
        self.classification = classification
        self.examples = examples


class Feature:
    def __init__(self, name="", vals=pd.array(data=[])):
        self.name = name
        self.vals = vals


class DiscreteFeature(Feature):
    def __init__(self, name="", vals=pd.array(data=[]), original_f_name=""):
        Feature.__init__(self, name, vals)
        self.original_f_name = original_f_name
