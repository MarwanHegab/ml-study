
import pandas as pd
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#original_file = pd.read_csv('wdbc.csv')

#shuffled_file = shuffle(original_file, random_state= 67)
#training_data, testing_data = train_test_split(shuffled_file, test_size=0.2, random_state=67)

#This function normalizes training and testing sets using the algorithm described in lecture
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

#KNN_on_dataset(training_data, training_data, 51)
#KNN_on_dataset(training_data, testing_data, 51)

