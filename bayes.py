import matplotlib.pyplot as plt
import numpy as np
import math

def count_words(vocab, dataset):
    word_counts = {}
    for word in vocab:
        word_counts[word] = 0
        
    total_words = 0  
    
    for document in dataset:
        for word in document:
            if word in word_counts:
                word_counts[word] += 1
                total_words += 1  
                
    return word_counts, total_words

def find_instances(document, vocab):
    word_instances = {}
    for word in vocab:
        word_instances[word] = 0
        
    for word in document:
        if word in word_instances:
            word_instances[word] = 1
            
    return word_instances

def calculate_denominator(total_words, vocab, alpha):
    return total_words + (alpha * len(vocab))

def predicted_label(document, word_counts, smooth_denom, prob, vocab, alpha):

    summ_temp = 0.0
    
    word_instances = find_instances(document, vocab)
    
    for word in word_instances:
        if word_instances[word] == 1:
            count = word_counts.get(word, 0)
            word_prob = (count + alpha) / smooth_denom
            summ_temp += math.log(word_prob)
            
    final_probability = math.log(prob) + summ_temp
    
    return final_probability

def run_bayes(positive_train, negative_train, positive_test, negative_test, vocab, alpha):
   
    total_train_docs = len(positive_train) + len(negative_train)
    pos_prob = len(positive_train) / total_train_docs
    neg_prob = len(negative_train) / total_train_docs
    
    pos_word_counts, pos_total_words = count_words(vocab, positive_train)
    neg_word_counts, neg_total_words = count_words(vocab, negative_train)
    
    pos_smooth_denom = calculate_denominator(pos_total_words, vocab, alpha)
    neg_smooth_denom = calculate_denominator(neg_total_words, vocab, alpha)
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for document in positive_test:
        positive_prob = predicted_label(document, pos_word_counts, pos_smooth_denom, pos_prob, vocab, alpha)
        negative_prob = predicted_label(document, neg_word_counts, neg_smooth_denom, neg_prob, vocab, alpha)
        
        if positive_prob >= negative_prob:
            true_positive += 1
        else:
            false_negative += 1
            
    for document in negative_test:
        positive_prob = predicted_label(document, pos_word_counts, pos_smooth_denom, pos_prob, vocab, alpha)
        negative_prob = predicted_label(document, neg_word_counts, neg_smooth_denom, neg_prob, vocab, alpha)
        
        if negative_prob >= positive_prob:
            true_negative += 1
        else:
            false_positive += 1
            
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return true_positive, true_negative, false_positive, false_negative, precision, recall, accuracy

def plot_confusion_matrix(tp, tn, fp, fn):
    
    matrix = np.array([[tp, fn], 
                       [fp, tn]])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'])
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), va='center', ha='center', 
                    color='white' if matrix[i, j] > (matrix.max() / 2) else 'black',
                    fontsize=14, fontweight='bold')
            
    plt.title('Naive Bayes Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.show()

import matplotlib.pyplot as plt

def plot_alpha_experiments(pos_train, neg_train, pos_test, neg_test, vocab):
    
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracies = []
    
    for alpha in alphas:
        *_, accuracy = run_bayes(pos_train, neg_train, pos_test, neg_test, vocab, alpha)
        accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, accuracies, marker='o', linestyle='-', color='b')
    
    plt.xscale('log')
    
    plt.xlabel('Alpha (\u03B1) Value (Log Scale)')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Impact of Laplace Smoothing (\u03B1) on Model Accuracy')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.show()
