import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

# local path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import kNN
import randomforest
import NN

def load_digits_data():
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]
    
    # end with '_num' so Random Forest treats them as numerical
    col_names = [f"pixel_{i}_num" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=col_names)
    df['label'] = y
    return df, X, y

def evaluate_knn(df, k_values, splits=10):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    X_df = df.drop(columns=['label'])
    y_series = df['label']
    
    results = {k: {'acc': [], 'f1': []} for k in k_values}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y_series)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        for k in k_values:
            y_pred = []
            for row in test_df.values:
                features = row[:-1]
                pred = kNN.kNN(train_df, k, features)
                y_pred.append(pred)
            
            y_true = test_df['label'].values
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            results[k]['acc'].append(acc)
            results[k]['f1'].append(f1)
            
    print("k-NN Results:")
    avg_results = {}
    for k in k_values:
        avg_acc = np.mean(results[k]['acc'])
        avg_f1 = np.mean(results[k]['f1'])
        avg_results[k] = (avg_acc, avg_f1)
        print(f"k={k} => Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}")
    return avg_results

def evaluate_rf(df, ntree_values, splits=10):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    X_df = df.drop(columns=['label'])
    y_series = df['label']
    
    results = {n: {'acc': [], 'f1': []} for n in ntree_values}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y_series)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        randomforest.original_attributes = train_df.columns.to_list()[:-1]
        
        for n in ntree_values:
            forest = randomforest.create_ensemble(train_df, n)
            
            y_pred = []
            for _, row in test_df.iterrows():
                pred = randomforest.ensemble_prediction(forest, row)
                y_pred.append(pred)
            
            y_true = test_df['label'].values
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            results[n]['acc'].append(acc)
            results[n]['f1'].append(f1)
            
    print("Random Forest Results:")
    avg_results = {}
    for n in ntree_values:
        avg_acc = np.mean(results[n]['acc'])
        avg_f1 = np.mean(results[n]['f1'])
        avg_results[n] = (avg_acc, avg_f1)
        print(f"ntrees={n} => Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}")
    return avg_results

def evaluate_nn(X, y, hyperparams, splits=10):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    num_classes = len(np.unique(y))
    
    results = {i: {'acc': [], 'f1': []} for i in range(len(hyperparams))}
    
    # One hot encode y for NN training
    y_oh = np.zeros((y.size, num_classes))
    y_oh[np.arange(y.size), y] = 1
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_oh, y_test_oh = y_oh[train_idx], y_oh[test_idx]
        y_test = y[test_idx]
        
        for i, hp in enumerate(hyperparams):
            epochs, LN, alpha, lambda_reg, mb_size = hp['epochs'], hp['LN'], hp['alpha'], hp['lambda_reg'], hp['mb_size']
            
            num_attributes = X_train.shape[1]
            weights = NN.weights_generation(LN, num_attributes)
            
            train_dataset = np.hstack((X_train, y_train_oh))
            
            for epoch in range(epochs):
                minibatches = NN.create_minibatch(train_dataset, mb_size)
                for batch in minibatches:
                    minibatch_X = batch[:, :-num_classes]
                    labels = batch[:, -num_classes:]
                    
                    final_prediction, cache = NN.forward_propagation(minibatch_X, weights)
                    deltas = NN.find_deltas(final_prediction, cache, weights, labels)
                    gradients = NN.compute_gradients(deltas, cache, weights, lambda_reg)
                    weights = NN.update_weights(weights, gradients, alpha)
            
            test_predictions, _ = NN.forward_propagation(X_test, weights)
            predicted_classes = np.argmax(test_predictions, axis=1)
            
            acc = accuracy_score(y_test, predicted_classes)
            f1 = f1_score(y_test, predicted_classes, average='weighted')
            
            results[i]['acc'].append(acc)
            results[i]['f1'].append(f1)

    print("Neural Network Results:")
    avg_results = {}
    for i, hp in enumerate(hyperparams):
        avg_acc = np.mean(results[i]['acc'])
        avg_f1 = np.mean(results[i]['f1'])
        avg_results[i] = (hp, avg_acc, avg_f1)
        print(f"HP {i}: {hp} => Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}")
    return avg_results

def plot_results(knn_res, rf_res, nn_res, nn_hyperparams):
    os.makedirs('figures', exist_ok=True)
    
    # k-NN plot
    k_vals = list(knn_res.keys())
    k_accs = [knn_res[k][0] for k in k_vals]
    plt.figure()
    plt.plot(k_vals, k_accs, marker='o')
    plt.title('k-NN: Accuracy vs k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig('figures/dataset1_knn_acc.png')
    
    # RF plot
    ntree_vals = list(rf_res.keys())
    rf_accs = [rf_res[n][0] for n in ntree_vals]
    plt.figure()
    plt.plot(ntree_vals, rf_accs, marker='o')
    plt.title('Random Forest: Accuracy vs Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.savefig('figures/dataset1_rf_acc.png')
    
    # NN plot
    labels = [f"HP{i}" for i in range(len(nn_hyperparams))]
    nn_accs = [nn_res[i][1] for i in range(len(nn_hyperparams))]
    plt.figure()
    plt.bar(labels, nn_accs)
    plt.title('Neural Network: Accuracy vs Hyperparameter Setting')
    plt.xlabel('Hyperparameter Setting')
    plt.ylabel('Accuracy')
    plt.savefig('figures/dataset1_nn_acc.png')

def generate_nn_learning_curve(X, y, best_hp):
    epochs, LN, alpha, lambda_reg, mb_size = best_hp['epochs'], best_hp['LN'], best_hp['alpha'], best_hp['lambda_reg'], best_hp['mb_size']
    num_classes = len(np.unique(y))
    y_oh = np.zeros((y.size, num_classes))
    y_oh[np.arange(y.size), y] = 1
    
    # Train on a single split to generate curve
    X_train, _, y_train_oh, _ = train_test_split(X, y_oh, test_size=0.1, stratify=y, random_state=42)
    
    num_attributes = X_train.shape[1]
    weights = NN.weights_generation(LN, num_attributes)
    train_dataset = np.hstack((X_train, y_train_oh))
    
    instances_seen = []
    costs = []
    total_instances = 0
    
    for epoch in range(epochs):
        minibatches = NN.create_minibatch(train_dataset, mb_size)
        for batch in minibatches:
            minibatch_X = batch[:, :-num_classes]
            labels = batch[:, -num_classes:]
            
            final_prediction, cache = NN.forward_propagation(minibatch_X, weights)
            deltas = NN.find_deltas(final_prediction, cache, weights, labels)
            gradients = NN.compute_gradients(deltas, cache, weights, lambda_reg)
            weights = NN.update_weights(weights, gradients, alpha)
            
            total_instances += len(batch)
            
        full_pred, _ = NN.forward_propagation(X_train, weights)
        cost = NN.compute_regularized_cost(y_train_oh, full_pred, weights, lambda_reg)
        
        instances_seen.append(total_instances)
        costs.append(cost)
        
    plt.figure()
    plt.plot(instances_seen, costs)
    plt.title('Neural Network: Cost J vs Training Instances')
    plt.xlabel('Number of Training Instances Presented')
    plt.ylabel('Cost J')
    plt.savefig('figures/dataset1_nn_cost.png')

if __name__ == '__main__':
    df, X, y = load_digits_data()
    print(f"Loaded Digits dataset: {df.shape[0]} instances, {X.shape[1]} features.")
    
    # Normalize features globally (divide by 16.0 as pixel max is 16)
    feature_cols = df.columns[:-1]
    df[feature_cols] = df[feature_cols] / 16.0
    
    print("Evaluating k-NN...")
    k_values = [1, 3, 5, 7, 9, 15]
    knn_res = evaluate_knn(df, k_values, splits=10)
    
    print("Evaluating Random Forest...")
    ntree_values = [1, 5, 10, 15, 20, 30]
    rf_res = evaluate_rf(df, ntree_values, splits=10)
    
    print("Evaluating Neural Network...")
    X_norm = X / 16.0 
    
    nn_hps = [
        {'epochs': 250, 'LN': [32, 10], 'alpha': 0.1, 'lambda_reg': 0.01, 'mb_size': 32},
        {'epochs': 250, 'LN': [16, 10], 'alpha': 0.1, 'lambda_reg': 0.01, 'mb_size': 32},
        {'epochs': 250, 'LN': [64, 32, 10], 'alpha': 0.1, 'lambda_reg': 0.01, 'mb_size': 32},
        {'epochs': 250, 'LN': [32, 10], 'alpha': 0.01, 'lambda_reg': 0.01, 'mb_size': 32},
        {'epochs': 250, 'LN': [32, 10], 'alpha': 0.5, 'lambda_reg': 0.01, 'mb_size': 32},
        {'epochs': 250, 'LN': [32, 10], 'alpha': 0.1, 'lambda_reg': 0.1, 'mb_size': 32}
    ]
    nn_res = evaluate_nn(X_norm, y, nn_hps, splits=10)
    
    plot_results(knn_res, rf_res, nn_res, nn_hps)
    
    best_nn_idx = max(nn_res, key=lambda k: nn_res[k][1]) # Maximize accuracy
    best_nn_hp = nn_res[best_nn_idx][0]
    
    print(f"Best NN HP: {best_nn_hp}")
    print("Generating NN Cost Learning Curve...")
    generate_nn_learning_curve(X_norm, y, best_nn_hp)
    
    print("Saved plots in cs589-final-project/figures/")
