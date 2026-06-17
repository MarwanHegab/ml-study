import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import NN

def load_digits_data():
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]
    col_names = [f"pixel_{i}_num" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=col_names)
    df['label'] = y
    return df, X, y

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
    plt.plot(instances_seen, costs, color='purple')
    plt.title('Neural Network: Cost J vs Training Instances')
    plt.xlabel('Number of Training Instances Presented')
    plt.ylabel('Cost J')
    plt.savefig('figures/dataset1_nn_cost.png')
    print("Cost curve generated.")

if __name__ == '__main__':
    df, X, y = load_digits_data()
    X_norm = X / 16.0 
    best_nn_hp = {'epochs': 250, 'LN': [32, 10], 'alpha': 0.5, 'lambda_reg': 0.01, 'mb_size': 32}
    generate_nn_learning_curve(X_norm, y, best_nn_hp)
