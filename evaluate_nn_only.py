import pandas as pd
import numpy as np
from sklearn import datasets
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import NN
from dataset1_analysis import evaluate_nn

def load_digits_data():
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]
    return X, y

if __name__ == '__main__':
    X, y = load_digits_data()
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
    
    best_nn_idx = max(nn_res, key=lambda k: nn_res[k][1])
    best_nn_hp = nn_res[best_nn_idx][0]
    print(f"Best NN HP: {best_nn_hp}")
