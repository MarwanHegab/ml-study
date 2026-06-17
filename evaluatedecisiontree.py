import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import StratifiedKFold

def preprocess_data(df, num_bins=5):

    df_processed = df.copy()
    
    target_col = df_processed.columns[-1]
    features = df_processed.columns[:-1]
    
    for col in features:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = pd.qcut(df_processed[col], q=num_bins, duplicates='drop').astype(str)
            
    return df_processed

def find_i(dataset):
    num_columns = len(dataset[0])
    all_unique_values = []
    for col_index in range(num_columns):
        unique_in_col = []
        for row in dataset:
            value = row[col_index]
            if value not in unique_in_col:
                unique_in_col.append(value)
        all_unique_values.append(unique_in_col)
    return all_unique_values

def calculate_probability(dataset, target_label):
    occurrences = 0
    total = len(dataset)
    if total == 0: return 0
    for row in dataset:
        if row[-1] == target_label: 
            occurrences += 1
    return occurrences / total

def calculate_entropy(dataset):
    entropy = 0
    unique_labels = list(set([row[-1] for row in dataset]))
    for label in unique_labels:
        prob = calculate_probability(dataset, label)
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

def calculate_average_entropy(dataset, col_index, attribute_categories):
    partitions = {cat: [] for cat in attribute_categories}
    for row in dataset:
        value = row[col_index]
        if value in partitions:
            partitions[value].append(row)
            
    total_rows = len(dataset)
    average_entropy = 0
    partitions_list = []
    
    for cat in attribute_categories:
        subset = partitions[cat]
        partitions_list.append(subset)
        subset_size = len(subset)
        if subset_size == 0:
            continue
        weight = subset_size / total_rows
        average_entropy += weight * calculate_entropy(subset)
        
    return average_entropy, partitions_list

def Gain(entropy_dataset, entropy_partitions):
    return entropy_dataset - entropy_partitions

def calculate_best_attribute_for_partitioning(dataset, available_columns, all_unique_values):
    base_entropy = calculate_entropy(dataset)
    most_info = -1.0 
    most_info_attribute = None
    best_partitions = None     
    best_categories = None      
    
    for col_index in available_columns:
        attribute_categories = all_unique_values[col_index]
        avg_entropy, partitions = calculate_average_entropy(dataset, col_index, attribute_categories)
        current_gain = Gain(base_entropy, avg_entropy)
        
        if current_gain > most_info:
            most_info = current_gain
            most_info_attribute = col_index
            best_partitions = partitions          
            best_categories = attribute_categories 
            
    return most_info_attribute, best_partitions, best_categories

def find_majority_class(dataset):
    counts = {}
    for row in dataset:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get) if counts else None

def decision_tree(dataset, available_columns, all_unique_values, max_depth, current_depth=0):
    labels_in_dataset = [row[-1] for row in dataset]
    unique_labels = list(set(labels_in_dataset))
    
    if len(unique_labels) == 1:
        return {'is_leaf': True, 'prediction': unique_labels[0]}
    if len(available_columns) == 0 or current_depth >= max_depth:
        return {'is_leaf': True, 'prediction': find_majority_class(dataset)}
        
    majority_label = find_majority_class(dataset)
    majority_count = labels_in_dataset.count(majority_label)
    if majority_count / len(dataset) >= 0.85:
        return {'is_leaf': True, 'prediction': majority_label}
        
    best_attr, best_partitions, best_categories = calculate_best_attribute_for_partitioning(dataset, available_columns, all_unique_values)
    
    if best_attr is None:
        return {'is_leaf': True, 'prediction': majority_label}
        
    node = {'is_leaf': False, 'attribute': best_attr, 'branches': {}}
    L_new = [col for col in available_columns if col != best_attr]
    
    for i in range(len(best_categories)):
        v = best_categories[i]    
        D_v = best_partitions[i] 
        
        if len(D_v) == 0:
            node['branches'][v] = {'is_leaf': True, 'prediction': majority_label}
        else:
            node['branches'][v] = decision_tree(D_v, L_new, all_unique_values, max_depth, current_depth + 1)
            
    return node

def predict(tree, row):
    if tree['is_leaf']:
        return tree['prediction']
    attribute_index = tree['attribute']
    feature_value = row[attribute_index]
    
    if feature_value in tree['branches']:
        return predict(tree['branches'][feature_value], row)
    else:
        first_available_branch = list(tree['branches'].values())[0]
        return predict(first_available_branch, row)

def calculate_metrics(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / len(y_true)
    
    classes = list(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall == 0: f1 = 0
        else: f1 = 2 * (precision * recall) / (precision + recall)
        f1s.append(f1)
        
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    return accuracy, macro_f1

def evaluate_decision_tree(tree, eval_set):
    y_true = [row[-1] for row in eval_set]
    y_pred = [predict(tree, row) for row in eval_set]
    return calculate_metrics(y_true, y_pred)

def run_experiment(file_path):
    print(f"\n--- Running Experiment for {file_path} ---")
    df = pd.read_csv(file_path, header=0)
    df = preprocess_data(df, num_bins=5)
    dataset = df.values
    
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(str) 
    
    available_columns = list(range(len(X[0])))
    all_unique_values = find_i(dataset)
 
    depths_to_test = [2, 4, 6, 8, 10, 15]
    
    mean_accuracies = []
    mean_f1_scores = []
    
    print(f"{'Max Depth':<12} | {'CV Accuracy':<12} | {'CV F1-Score':<12}")
    print("-" * 42)
    
    for depth in depths_to_test:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_acc = []
        fold_f1 = []
        
        for train_index, test_index in skf.split(X, y):
            train_set = dataset[train_index]
            test_set = dataset[test_index]
            
            tree = decision_tree(train_set, available_columns, all_unique_values, max_depth=depth)
            
            acc, f1 = evaluate_decision_tree(tree, test_set)
            fold_acc.append(acc)
            fold_f1.append(f1)
            
        avg_acc = np.mean(fold_acc)
        avg_f1 = np.mean(fold_f1)
        mean_accuracies.append(avg_acc)
        mean_f1_scores.append(avg_f1)
        
        print(f"{depth:<12} | {avg_acc:<12.4f} | {avg_f1:<12.4f}")

    best_idx = np.argmax(mean_f1_scores)
    print(f"\nBest Hyperparameter Setting: max_depth = {depths_to_test[best_idx]}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(depths_to_test, mean_accuracies, marker='o', label='Accuracy', color='blue')
    plt.plot(depths_to_test, mean_f1_scores, marker='s', label='F1-Score', color='orange')
    plt.title(f'Learning Curve: Hyperparameter Tuning ({file_path})')
    plt.xlabel('Max Depth (Hyperparameter)')
    plt.ylabel('Performance Metric')
    plt.xticks(depths_to_test)
    plt.legend()
    plt.grid(True)
    plt.show()

try:
    #run_experiment('rice.csv')
    run_experiment('parkinsons.csv')
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure datasets are in the same directory.")
