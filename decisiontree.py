import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statistics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load data
original_data = pd.read_csv('datasets/credit_approval.csv')
original_attributes = original_data.columns.to_list()
original_attributes.remove('label')

# definitionssss... so many functions ;_;

# create Node class for constructing decision trees
class Node:
    def __init__(self):
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.threshold = None
        self.children = {}


# calculate entropy of given set
def calculate_entropy(set):
    # proportions = set.value_counts(normalize=True)
    _, counts = np.unique(set, return_counts=True)
    proportions = counts / len(set)
    return -np.sum(proportions * np.log2(proportions))

# calculate average entropy of data given an attribute to split by
def calculate_split_entropy(data, split_attribute):

    total_len = len(data)
    avg_entropy = 0

    if split_attribute.endswith('num'):
        # get all values and sort
        unique_vals = np.sort(data[split_attribute].unique())
        
        # if there is only 1 unique value left, we can't split it further
        if len(unique_vals) <= 1:
            return calculate_entropy(data['label']), None
            
        best_entropy = float('inf')
        best_threshold = None

        attribute_vals = data[split_attribute].values
        labels = data['label'].values
        
        # loop through consecutive pairs to find midpoints, check entropies
        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0

            subset_less = labels[attribute_vals <= threshold]
            subset_great = labels[attribute_vals > threshold]
            
            entropy_less = calculate_entropy(subset_less)
            entropy_great = calculate_entropy(subset_great)
            
            weight_less = len(subset_less) / total_len
            weight_great = len(subset_great) / total_len
            
            current_avg_entropy = (entropy_less * weight_less) + (entropy_great * weight_great)

            if current_avg_entropy < best_entropy:
                best_entropy = current_avg_entropy
                best_threshold = threshold
                
        return best_entropy, best_threshold
    # if categorical, extract unique values and respective counts from given data
    else:
        values, counts = np.unique(data[split_attribute], return_counts=True)

    for value, count in zip(values, counts):
        # extract subset from data of all given value types
        subset = data[data[split_attribute] == value]
        # calculate entropy of subset, update weighted avg
        entropy = calculate_entropy(subset['label'])
        weight = count / len(data)
        avg_entropy += entropy * weight

    return avg_entropy, None


# recursive decision tree algorithm following pseudocode from class
def create_decision_tree(data, attributes, max_depth=None, current_depth=0):

    # create new node to add to tree
    N = Node()
    labels = data['label']

    # base case
    if max_depth is not None and current_depth >= max_depth:
        N.is_leaf = True
        N.label = statistics.mode(labels)
        return N

    # all label values are the same, assign to label value
    if (len(labels.unique()) == 1):
        N.is_leaf = True
        N.label = labels.iloc[0]
        return N
    # no more attributes to be tested, give to majority of label value
    if (len(attributes) == 0):
        N.is_leaf = True
        N.label = statistics.mode(labels)
        return N

    # recursive step
    
    # find the best splitting attribute (highest info gain)
    current_entropy = calculate_entropy(labels)
    best_gain = 0
    best_splitting_attribute = None
    best_threshold = None

    for attribute in attributes:
        split_entropy, threshold = calculate_split_entropy(data, attribute)
        gain = current_entropy - split_entropy
        if gain > best_gain: 
            best_gain = gain
            best_splitting_attribute = attribute
            best_threshold = threshold

    # set Node attribute as best split attribute
    N.attribute = best_splitting_attribute
    N.threshold = best_threshold
    # set label to majority just in case feature was not seen in training data
    N.label = statistics.mode(labels)

    # stopping criteria - no useful split found, no information gain added, add leaf
    if best_splitting_attribute is None:
        N.is_leaf = True
        return N

    remaining_attributes = attributes.copy()

    # numerical vs categorical handling
    if best_splitting_attribute.endswith('num'):
        # split by threshold
        partitions = {
            '<= threshold': data[data[best_splitting_attribute] <= best_threshold],
            '> threshold': data[data[best_splitting_attribute] > best_threshold]
        }
        for key, partition in partitions.items():
            if partition.empty:
                leaf = Node()
                leaf.is_leaf = True
                leaf.label = statistics.mode(labels)
                N.children[key] = leaf
            else:
                sub_tree = create_decision_tree(partition, remaining_attributes, max_depth, current_depth + 1)
                N.children[key] = sub_tree
    else:
        remaining_attributes.remove(best_splitting_attribute)
        best_attribute_data_values = data[best_splitting_attribute].unique()
        # create leaves & sub-trees
        for value in best_attribute_data_values:
            partition = data[data[best_splitting_attribute] == value]

            # if partition is empty, create leaf
            if partition.empty:
                leaf = Node()
                leaf.is_leaf = True
                leaf.label = statistics.mode(labels)
                N.children[value] = leaf
            # else create sub_tree
            else:
                sub_tree = create_decision_tree(partition, remaining_attributes, max_depth, current_depth + 1)
                N.children[value] = sub_tree
    return N

# traverse tree to make prediction for given instance
def make_prediction(node, instance):
    # return leaf node label
    if node.is_leaf:
        return node.label
    
    # get value for given attribute in instance
    value = instance[node.attribute]

    # find child node with this value

    if node.attribute.endswith('num'):
        # numerical, use threshold
        child_key = '<= threshold' if value <= node.threshold else '> threshold'
        if child_key in node.children:
            return make_prediction(node.children[child_key], instance)
        else:
            return node.label
    else:
        # categorical
        if value in node.children:
            child = node.children[value]
            return make_prediction(child, instance)
        # edge case - not leaf node but no value found, get majority label value
        else:
            return node.label

def predict_batch(tree, data):
    return [make_prediction(tree, row) for _, row in data.iterrows()]

# stratified k fold index finder
def get_fold_indices(data, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    attributes = data.drop(columns=['label'])
    labels = data['label']

    train_folds = []
    test_folds = []
    for train, test in skf.split(attributes, labels):
        train_folds.append(train)
        test_folds.append(test)

    return train_folds, test_folds