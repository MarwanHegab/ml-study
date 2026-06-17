import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_prepare_data(filepath, target_col=None):
    """
    Loads the dataset, identifies the target column, extracts attributes, 
    and maps string class labels (e.g., 'Cammeo', 'Osmancik') to integers.
    """
    data = pd.read_csv(filepath)
    
    # If target_col is not specified, assume it is the last column
    if target_col is None:
        target_col = data.columns[-1]
        
    attributes = data.columns.to_list()
    attributes.remove(target_col)
    
    # Map string labels to integers if necessary
    if data[target_col].dtype == 'object' or data[target_col].dtype.name == 'category':
        unique_classes = data[target_col].unique()
        class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
        data[target_col] = data[target_col].map(class_mapping)
        print(f"Mapped classes for '{target_col}': {class_mapping}")
        
    return data, attributes, target_col

def is_numeric_attribute(data, attribute):
    """Safely checks if a given column in the dataframe is numeric."""
    return pd.api.types.is_numeric_dtype(data[attribute])


class Node:
    def __init__(self):
        self.is_leaf = False
        self.label = None
        self.attribute = None
        self.threshold = None
        self.is_numeric = False # Added to remember attribute type during prediction
        self.children = {}

def make_bootstrap(data):
    return data.sample(frac=1.0, replace=True)

def calculate_entropy(subset_labels):
    _, counts = np.unique(subset_labels, return_counts=True)
    proportions = counts / len(subset_labels)
    return -np.sum(proportions * np.log2(proportions))

def calculate_split_entropy(data, split_attribute, target_col):
    total_len = len(data)
    avg_entropy = 0

    if is_numeric_attribute(data, split_attribute):
        unique_vals = np.sort(data[split_attribute].unique())
        
        if len(unique_vals) <= 1:
            return calculate_entropy(data[target_col]), None
            
        best_entropy = float('inf')
        best_threshold = None

        attribute_vals = data[split_attribute].values
        labels = data[target_col].values
        
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
    else:
        values, counts = np.unique(data[split_attribute], return_counts=True)

        for value, count in zip(values, counts):
            subset = data[data[split_attribute] == value]
            entropy = calculate_entropy(subset[target_col])
            weight = count / len(data)
            avg_entropy += entropy * weight

        return avg_entropy, None

def create_decision_tree(data, attributes, original_attributes, target_col):
    N = Node()
    labels = data[target_col]

    if (len(labels.unique()) == 1):
        N.is_leaf = True
        N.label = int(labels.iloc[0])
        return N

    if (len(attributes) == 0):
        N.is_leaf = True
        N.label = statistics.mode(labels)
        return N
    
    current_entropy = calculate_entropy(labels)
    best_gain = 0
    best_splitting_attribute = None
    best_threshold = None

    m = len(original_attributes)
    selected_attributes = np.random.choice(attributes, size=int(np.sqrt(m)), replace=False).tolist()
    if m > len(attributes): 
        selected_attributes = attributes

    for attribute in selected_attributes:
        split_entropy, threshold = calculate_split_entropy(data, attribute, target_col)
        gain = current_entropy - split_entropy
        if gain > best_gain: 
            best_gain = gain
            best_splitting_attribute = attribute
            best_threshold = threshold

    N.attribute = best_splitting_attribute
    N.threshold = best_threshold
    N.label = statistics.mode(labels)

    if best_splitting_attribute is None:
        N.is_leaf = True
        return N

    # Store whether the chosen attribute was numeric so the prediction phase knows
    N.is_numeric = is_numeric_attribute(data, best_splitting_attribute)
    remaining_attributes = attributes.copy()

    if N.is_numeric:
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
                N.children[key] = create_decision_tree(partition, remaining_attributes, original_attributes, target_col)
    else:
        remaining_attributes.remove(best_splitting_attribute)
        best_attribute_data_values = data[best_splitting_attribute].unique()
        for value in best_attribute_data_values:
            partition = data[data[best_splitting_attribute] == value]

            if partition.empty:
                leaf = Node()
                leaf.is_leaf = True
                leaf.label = statistics.mode(labels)
                N.children[value] = leaf
            else:
                N.children[value] = create_decision_tree(partition, remaining_attributes, original_attributes, target_col)
    return N

def make_prediction(node, instance):
    if node.is_leaf:
        return node.label
    
    value = instance[node.attribute]

    if node.is_numeric:
        child_key = '<= threshold' if value <= node.threshold else '> threshold'
        if child_key in node.children:
            return make_prediction(node.children[child_key], instance)
        else:
            return node.label
    else:
        if value in node.children:
            return make_prediction(node.children[value], instance)
        else:
            return node.label
        
def ensemble_prediction(ensemble, instance):
    predicted_labels = [make_prediction(tree, instance) for tree in ensemble]
    try:
        return statistics.mode(predicted_labels)
    except statistics.StatisticsError:
        # Fallback if there's a tie in voting
        return predicted_labels[0]

def create_ensemble(data, n_trees, original_attributes, target_col):
    ensemble = []
    for _ in range(n_trees):
        bag = make_bootstrap(data)
        decision_tree = create_decision_tree(bag, original_attributes, original_attributes, target_col)
        ensemble.append(decision_tree)
    return ensemble

def get_fold_indices(data, k, target_col):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    attributes = data.drop(columns=[target_col])
    labels = data[target_col]

    train_folds = []
    test_folds = []
    for train, test in skf.split(attributes, labels):
        train_folds.append(train)
        test_folds.append(test)

    return train_folds, test_folds

def evaluate_forest(filepath, target_col=None, k=5, num_trees_list=[1, 5, 10, 20]):
    """
    Main pipeline function to load data, train the forest across k-folds, 
    and evaluate metrics for a given list of tree quantities.
    """
    data, original_attributes, target_col = load_and_prepare_data(filepath, target_col)
    train_indices, test_indices = get_fold_indices(data, k, target_col)

    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for n_tree in num_trees_list:
        metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

        for i in range(k):
            train_fold = data.iloc[train_indices[i]]
            test_fold = data.iloc[test_indices[i]]
            test_labels = test_fold[target_col]

            forest = create_ensemble(train_fold, n_tree, original_attributes, target_col)
            predicted_test_labels = test_fold.apply(lambda instance: ensemble_prediction(forest, instance), axis=1)

            # Note: average='weighted' used here to support multi-class datasets gracefully
            metrics['acc'].append(accuracy_score(test_labels, predicted_test_labels))
            metrics['prec'].append(precision_score(test_labels, predicted_test_labels, average='weighted', zero_division=0))
            metrics['rec'].append(recall_score(test_labels, predicted_test_labels, average='weighted', zero_division=0))
            metrics['f1'].append(f1_score(test_labels, predicted_test_labels, average='weighted', zero_division=0))
            
        avg_acc = statistics.mean(metrics['acc'])
        avg_prec = statistics.mean(metrics['prec'])
        avg_rec = statistics.mean(metrics['rec'])
        avg_f1 = statistics.mean(metrics['f1'])
        
        print(f"Trees: {n_tree:2d} | Avg Acc: {avg_acc:.4f} | Avg Prec: {avg_prec:.4f} | Avg Rec: {avg_rec:.4f} | Avg F1: {avg_f1:.4f}")
        
        results['accuracy'].append(avg_acc)
        results['precision'].append(avg_prec)
        results['recall'].append(avg_rec)
        results['f1'].append(avg_f1)

    return num_trees_list, results
