import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statistics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# definitionssss... so many functions ;_;

# create Node class for constructing decision trees
class Node:
    def __init__(self):
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.threshold = None
        self.children = {}

# create bootstrap (sample instances with replacement from og data)
def make_bootstrap(data):
    return data.sample(frac=1.0, replace=True)

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

    # if split_attribute.endswith('num'):
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

        # subset_less = data[data[split_attribute] <= threshold]
        # subset_great = data[data[split_attribute] > threshold]

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
    # else:
    #     values, counts = np.unique(data[split_attribute], return_counts=True)

    for value, count in zip(values, counts):
        # extract subset from data of all given value types
        subset = data[data[split_attribute] == value]
        # calculate entropy of subset, update weighted avg
        entropy = calculate_entropy(subset['label'])
        weight = count / len(data)
        avg_entropy += entropy * weight

    return avg_entropy, None


# recursive decision tree algorithm following pseudocode from class
def create_decision_tree(data, attributes):

    # create new node to add to tree
    N = Node()
    labels = data['label']

    # base case
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

    # select sqrt m attributes to use
    m = len(original_attributes)
    selected_attributes = np.random.choice(attributes, size=int(np.sqrt(m)), replace=False)
    selected_attributes = selected_attributes.tolist()
    # take smaller array, either sqrt(m) from attributes, or all remaining attributes
    if m > len(attributes): selected_attributes = attributes

    for attribute in selected_attributes:
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
        N.label = statistics.mode(labels)
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
                sub_tree = create_decision_tree(partition, remaining_attributes)
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
                sub_tree = create_decision_tree(partition, remaining_attributes)
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

    # if node.attribute.endswith('num'):
    # numerical, use threshold
    child_key = '<= threshold' if value <= node.threshold else '> threshold'
    if child_key in node.children:
        return make_prediction(node.children[child_key], instance)
    else:
        return node.label
    # else:
    #     # categorical
    #     if value in node.children:
    #         child = node.children[value]
    #         return make_prediction(child, instance)
    #     # edge case - not leaf node but no value found, get majority label value
    #     else:
    #         return node.label
        
def ensemble_prediction(ensemble, instance):
    predicted_labels = []

    for tree in ensemble:
        predicted_labels.append(make_prediction(tree, instance))
    
    return statistics.mode(predicted_labels)

# create forest
def create_tree_ensemble(data, n_trees):
    ensemble = []

    for tree in range(n_trees):

        # create bag
        bag = make_bootstrap(data)

        decision_tree = create_decision_tree(bag, original_attributes)

        ensemble.append(decision_tree)
    
    return ensemble


# kNN section - copied from kNN.py, modified to return k votes

def normalize_datasets(training, testing):

    normalized_training = training.copy()
    normalized_testing = testing.copy()
    
    feature_cols = training.columns[:-1]
    
    train_min = normalized_training[feature_cols].min()
    train_max = normalized_training[feature_cols].max()
    
    normalized_training[feature_cols] = (normalized_training[feature_cols] - train_min) / (train_max - train_min)
    normalized_testing[feature_cols] = (normalized_testing[feature_cols] - train_min) / (train_max - train_min)
    
    return normalized_training, normalized_testing

#training_data, testing_data = normalize_datasets(training_data, testing_data)

#Simple function that calculate euclidean distance
def compute_euclidean_distance(point1, point2):

    p1 = np.array(point1)
    p2 = np.array(point2)
    
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    
    return distance

#This function performs the KNN algorithm on one particular instance
def kNN(dataset, k, point):
    
    k_nearest = []
    
    for row in dataset.values:
        features = row[:-1]
        label = row[-1]
        dist = compute_euclidean_distance(point[:-1], features)
        
        if len(k_nearest) < k:
            k_nearest.append([dist, label])
        else:
            max_dist_idx = 0
            max_dist = k_nearest[0][0]
            
            for i in range(1, k):
                if k_nearest[i][0] > max_dist:
                    max_dist = k_nearest[i][0]
                    max_dist_idx = i
                    
            if dist < max_dist:
                k_nearest[max_dist_idx] = [dist, label]
                
    counts = {}
    for n in k_nearest:
        neighbor_label = n[1]
        counts[neighbor_label] = counts.get(neighbor_label, 0) + 1
    
    best_label = None
    max_votes = -1
    for label, count in counts.items():
        if count > max_votes:
            max_votes = count
            best_label = label
            
    return best_label

    
#This function performs KNN on an entire dataset
def KNN_on_dataset(train_set, eval_set, k):
    
    total_correct = 0
    total_incorrect = 0
    
    for row in eval_set.values:

        features = row[:-1]
        curr_correct_label = row[-1]
        
        predicted_label = kNN(train_set, k, features)
        
        if predicted_label == curr_correct_label:
            total_correct += 1
        else:
            total_incorrect += 1
            
    accuracy = total_correct / (total_correct + total_incorrect)
    print(accuracy)
    return accuracy

def create_knn_ensemble(data, n_knns):
    knn_ensemble = []
    for _ in range(n_knns):
        bag = make_bootstrap(data)
        knn_ensemble.append(bag)
    return knn_ensemble 


# ensemle management functions

def ensemble_prediction(rf_ensemble, knn_ensemble, k, instance):
    votes = []
    
    # get n_tree votes from random forest
    for tree in rf_ensemble:
        votes.append(make_prediction(tree, instance))
        
    # get k votes from the kNN 
    for knn_bag in knn_ensemble:
        votes.append(kNN(knn_bag, k, instance))
    
    # most votes wins, or random winner if equal
    winner = statistics.mode(votes)
    if votes.count(winner) == len(votes) / 2:
        return np.random.choice(votes)
    return winner


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


def evaluate_hybrid(df, num_trees_and_knns, k, splits=10):
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    X_df = df.drop(columns=['label'])
    y_series = df['label']
    
    # dictionary to track results for every round of n_trees and n_knns
    results = {(n): {'acc': [], 'f1': []} for n in num_trees_and_knns}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y_series)):        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        norm_train, norm_test = normalize_datasets(train_df, test_df)
        
        for n in num_trees_and_knns:
            forest = create_tree_ensemble(norm_train, n)
            kNNs = create_knn_ensemble(norm_train, n)
            
            y_pred = []
            
            # take vote for each instance
            for _, row in norm_test.iterrows():
                pred = ensemble_prediction(forest, kNNs, k, row)
                y_pred.append(pred)
            
            y_true = test_df['label'].values
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            results[n]['acc'].append(acc)
            results[n]['f1'].append(f1)
            
    print("--- Hybrid Ensemble (Decision Trees + kNN) Results ---")
    avg_results = {}
    for n in num_trees_and_knns:
        avg_acc = np.mean(results[n]['acc'])
        avg_f1 = np.mean(results[n]['f1'])
        avg_results[n] = (avg_acc, avg_f1)
        print(f"ntrees_and_knns={n}, k={k} => Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}")
            
    return avg_results

n_trees_and_knns = [1, 3, 5, 7, 9, 11]

df = pd.read_csv('datasets/parkinsons.csv')
df = df.rename(columns={'Diagnosis': 'label'})
X = df.drop(columns=['label'])
original_attributes = X.columns.to_list()

evaluate_hybrid(df, n_trees_and_knns, k=3, splits=10)