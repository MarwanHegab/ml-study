import os
import numpy as np
import matplotlib.pyplot as plt

import random_forest as rf

def generate_learning_curves(num_trees_list, metrics_dict, dataset_name):
    os.makedirs("latex_source/figures", exist_ok=True)
    
    metrics_to_plot = {
        'Accuracy': metrics_dict['accuracy'],
        'Precision': metrics_dict['precision'],
        'Recall': metrics_dict['recall'],
        'F1_Score': metrics_dict['f1']
    }
    
    for metric_name, values in metrics_to_plot.items():
        plt.figure(figsize=(8, 5))
        plt.plot(num_trees_list, values, marker='D', color='navy', linestyle='-', linewidth=2)
        plt.xlabel('Number of Trees')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs Number of Trees ({dataset_name})')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        filename = f"latex_source/figures/{dataset_name}_{metric_name}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 
        print(f"Saved graph: {filename}")


def run_evaluation(filepath, target_col, dataset_name):
    print(f"\n{'='*40}")
    print(f"Evaluating Dataset: {dataset_name}")
    print(f"{'='*40}")
    
    k_folds = 10
    trees_to_test = [1, 5, 10, 20, 30, 40, 50]
    
    trees_list, results = rf.evaluate_forest(
        filepath=filepath, 
        target_col=target_col, 
        k=k_folds, 
        num_trees_list=trees_to_test
    )
    
    best_idx = np.argmax(results['accuracy'])
    best_num_trees = trees_list[best_idx]
    best_accuracy = results['accuracy'][best_idx]
    best_f1 = results['f1'][best_idx]
    
    print(f"\n--- Optimal Hyperparameters for {dataset_name} ---")
    print(f"Best Number of Trees: {best_num_trees}")
    print(f"Highest Accuracy:     {best_accuracy:.4f}")
    print(f"Corresponding F1:     {best_f1:.4f}\n")
    
    generate_learning_curves(trees_list, results, dataset_name)
    print(f"Finished evaluating {dataset_name}.\n")





run_evaluation(
    filepath='parkinsons.csv', 
    target_col='Diagnosis', 
    dataset_name='parkinsons'
)

run_evaluation(
    filepath='rice.csv', 
    target_col='label', 
    dataset_name='Rice_Grains'
)
    
