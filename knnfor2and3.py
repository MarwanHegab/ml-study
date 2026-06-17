import pandas as pd
import numpy as np

def encode_categorical_and_normalize(training, testing, target_col):

    train_y = training[target_col].copy()
    test_y = testing[target_col].copy()
    
    train_X = training.drop(columns=[target_col])
    test_X = testing.drop(columns=[target_col])

    train_X = pd.get_dummies(train_X)
    test_X = pd.get_dummies(test_X)
    train_X, test_X = train_X.align(test_X, join='left', axis=1, fill_value=0)

    feature_cols = train_X.columns
    train_min = train_X.min()
    train_max = train_X.max()
    
    range_vals = train_max - train_min
    range_vals = range_vals.replace(0, 1)
    
    train_X[feature_cols] = (train_X - train_min) / range_vals
    test_X[feature_cols] = (test_X - train_min) / range_vals

    train_final = train_X.copy()
    train_final[target_col] = train_y
    
    test_final = test_X.copy()
    test_final[target_col] = test_y
    
    return train_final, test_final

# Simple function that calculates euclidean distance (unchanged)
def compute_euclidean_distance(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    return distance

# Performs the KNN algorithm on one particular instance (unchanged)
def kNN(dataset, k, point):
    k_nearest = []
    
    for row in dataset.values:
        features = row[:-1]
        label = row[-1]
        dist = compute_euclidean_distance(point, features)
        
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
                
    votes_0 = 0
    votes_1 = 0
    
    for n in k_nearest:
        neighbor_label = n[1]
        if neighbor_label == 0:
            votes_0 += 1
        elif neighbor_label == 1:
            votes_1 += 1
            
    if votes_0 > votes_1:
        return 0
    else:
        return 1

# Performs KNN on an entire dataset (updated to return accuracy & f1_score)
def KNN_on_dataset(train_set, eval_set, k):
    total_correct = 0
    total_incorrect = 0
    
    tp = 0 # True Positives
    fp = 0 # False Positives
    fn = 0 # False Negatives
    
    for row in eval_set.values:
        features = row[:-1]
        curr_correct_label = row[-1]
        predicted_label = kNN(train_set, k, features)
        
        # Track for Accuracy
        if predicted_label == curr_correct_label:
            total_correct += 1
            if predicted_label == 1:
                tp += 1
        else:
            total_incorrect += 1
            if predicted_label == 1:
                fp += 1
            elif curr_correct_label == 1 and predicted_label == 0:
                fn += 1
                
    accuracy = total_correct / (total_correct + total_incorrect)
    
    # Calculate F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, f1_score
