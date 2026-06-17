import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from kNN import encode_categorical_and_normalize, KNN_on_dataset

k_values = [1, 3, 5, 7, 9, 11]

def evaluate_dataset(file_path, target_col, dataset_name):
    print(f"\n{'='*40}")
    print(f"Evaluating {dataset_name} Dataset...")
    print(f"{'='*40}")
    
    df = pd.read_csv(file_path)
    
    unique_labels = df[target_col].unique()
    df[target_col] = df[target_col].map({unique_labels[0]: 0, unique_labels[1]: 1})
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=67)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    k_accuracies = []
    k_f1_scores = []
    
    for k in k_values:
        print(f"  Testing k = {k}...")
        fold_accuracies = []
        fold_f1s = []
        
        for train_idx, test_idx in skf.split(X, y):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            norm_train, norm_test = encode_categorical_and_normalize(train_data, test_data, target_col)
            
            accuracy, f1 = KNN_on_dataset(norm_train, norm_test, k)
            fold_accuracies.append(accuracy)
            fold_f1s.append(f1)
            
        mean_acc = np.mean(fold_accuracies)
        mean_f1 = np.mean(fold_f1s)
        
        k_accuracies.append(mean_acc)
        k_f1_scores.append(mean_f1)
        
        print(f"    -> Mean Accuracy: {mean_acc:.4f} | Mean F1-Score: {mean_f1:.4f}")

    best_idx = np.argmax(k_accuracies)
    print(f"\n[RESULTS] Best setting for {dataset_name}: k = {k_values[best_idx]}")
    print(f"Best Accuracy: {k_accuracies[best_idx]:.4f}")
    print(f"Best F1-Score: {k_f1_scores[best_idx]:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, k_accuracies, marker='o', label='Accuracy', color='green')
    plt.plot(k_values, k_f1_scores, marker='s', label='F1-Score', color='blue')
    plt.title(f'k-NN Performance on {dataset_name} Dataset')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Score')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.show()


evaluate_dataset('rice.csv', 'label', 'rice')
evaluate_dataset('parkinsons.csv', 'Diagnosis', 'Parkinsons')
