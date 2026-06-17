
from utils import load_training_set, load_test_set
import numpy as np
import matplotlib.pyplot as plt

def MultinomialNB(pos_training, neg_train, vocab, alpha=0):
    N_pos = len(pos_train) 
    N_neg = len(neg_train)
    N = N_pos + N_neg   # total number of documents available in the training set 
    V_size = len(vocab) # |V|, length of the vocabulary

    # Pr(y_i) = N(y_i) / N
    prior_pr_pos = N_pos / N
    prior_pr_neg = N_neg / N

    # n(w_k, y_i): frequency of word w_k
    word_counts_pos = {}
    word_counts_neg = {}

    for doc in pos_train:
        for word in doc:
            word_counts_pos[word] = word_counts_pos.get(word, 0) + 1

    for doc in neg_train:
        for word in doc:
            word_counts_neg[word] = word_counts_neg.get(word, 0) + 1

    # sum n(w_s, y_i): sum of frequency of word all words in class y_i
    total_words_pos = sum(word_counts_pos.values())
    total_words_neg = sum(word_counts_neg.values())

    # equation of the conditional probability of a word given a class
    cond_pr_pos = {}
    cond_pr_neg = {}
    for word in vocab:
        cond_pr_pos[word] = (word_counts_pos.get(word, 0) + alpha) / (total_words_pos + alpha * V_size)
        cond_pr_neg[word] = (word_counts_neg.get(word, 0) + alpha) / (total_words_neg + alpha * V_size)

    return {
        'prior_pr_pos': prior_pr_pos,
        'prior_pr_neg': prior_pr_neg,
        'cond_pr_pos': cond_pr_pos,
        'cond_pr_neg': cond_pr_neg,
        'vocab': vocab,
    }

# classifing using raw posterior
def classify_raw(doc, model):
    cond_pr_pos = model['cond_pr_pos']
    cond_pr_neg = model['cond_pr_neg']
    vocab = model['vocab']

    # Pr(y_i | Doc) proportional to Pr(y_i) * prod_{k=1}^{len(Doc)} Pr(w_k | y_i)
    posterior_pos = model['prior_pr_pos']
    posterior_neg = model['prior_pr_neg']

    for word in set(doc):
        if word not in vocab:
            continue
        posterior_pos *= cond_pr_pos[word]
        posterior_neg *= cond_pr_neg[word]

    if posterior_pos >= posterior_neg:
        return 'positive'
    else:
        return 'negative'

# classifying using log-posterior
def classify_log(doc, model):
    cond_prob_pos = model['cond_pr_pos']
    cond_prob_neg = model['cond_pr_neg']
    vocab = model['vocab']

    # log Pr(y_i | Doc) = log Pr(y_i) + sum_{k=1}^{len(Doc)} log Pr(w_k | y_i)
    log_posterior_pos = np.log(model['prior_pr_pos'])
    log_posterior_neg = np.log(model['prior_pr_neg'])

    for word in set(doc):
        if word not in vocab:
            continue
        log_posterior_pos += np.log(cond_prob_pos[word])
        log_posterior_neg += np.log(cond_prob_neg[word])

    if log_posterior_pos >= log_posterior_neg:
        return 'positive'
    else:
        return 'negative'

def compute_matrices(pos_test, neg_test, model, classify_fn):
    tp = fp = tn = fn = 0
    for doc in pos_test:
        if classify_fn(doc, model) == 'positive':
            tp += 1
        else:
            fn += 1

    for doc in neg_test:
        if classify_fn(doc, model) == 'negative':
            tn += 1
        else:
            fp += 1

    total = tp + tn + fp + fn

    if total > 0:
        accuracy = (tp + tn) / total 
    else:
        accuracy = 0
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp) 
    else:
        precision = 0

    if (tp + fn) > 0:
        recall = tp / (tp + fn) 
    else:
        recall = 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }

if __name__ == '__main__':
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    # Q1
    print()
    print("Question 1")
    (pos_train, neg_train, vocab) = load_training_set(0.2, 0.2)
    (pos_test, neg_test) = load_test_set(0.2, 0.2)

    print(f"Training: {len(pos_train)} pos, {len(neg_train)} neg")
    print(f"Test: {len(pos_test)} pos, {len(neg_test)} neg")
    print(f"Vocab: {len(vocab)} words")

    model_q1 = MultinomialNB(pos_train, neg_train, vocab, alpha=0)
    results_q1 = compute_matrices(pos_test, neg_test, model_q1, classify_raw)
    print(results_q1)

    # Q2
    print()
    print ("Question 2")
    (pos_train, neg_train, vocab) = load_training_set(0.2, 0.2)
    (pos_test, neg_test) = load_test_set(0.2, 0.2)

    print(f"Training: {len(pos_train)} pos, {len(neg_train)} neg")
    print(f"Test:     {len(pos_test)} pos, {len(neg_test)} neg")
    print(f"Vocab:    {len(vocab)} words")

    print("\n--- alpha = 1 ---")
    model_q2 = MultinomialNB(pos_train, neg_train, vocab, alpha=1)
    results_q2 = compute_matrices(pos_test, neg_test, model_q2, classify_log)
    print(results_q2)

    print("\n--- Alpha sweep ---")
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracies = []

    for a in alphas:
        model = MultinomialNB(pos_train, neg_train, vocab, alpha=a)
        results = compute_matrices(pos_test, neg_test, model, classify_log)
        accuracies.append(results['accuracy'])
        print(f"  alpha={a:<10}  accuracy={results['accuracy']:.4f}")

    best_idx = accuracies.index(max(accuracies))
    best_alpha = alphas[best_idx]
    print(f"\nBest alpha: {best_alpha} (accuracy={max(accuracies):.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Q2: Accuracy vs Alpha (Laplace Smoothing)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q2_accuracy_vs_alpha.png', dpi=150)
    print("Plot saved to q2_accuracy_vs_alpha.png")

    # Q3
    print()
    print ("Question 3")
    (pos_train, neg_train, vocab) = load_training_set(1.0, 1.0)
    (pos_test, neg_test) = load_test_set(1.0, 1.0)

    print(f"Training: {len(pos_train)} pos, {len(neg_train)} neg")
    print(f"Test:     {len(pos_test)} pos, {len(neg_test)} neg")
    print(f"Vocab:    {len(vocab)} words")

    model_q3 = MultinomialNB(pos_train, neg_train, vocab, alpha=best_alpha)
    results_q3 = compute_matrices(pos_test, neg_test, model_q3, classify_log)
    print(results_q3)

    # Q4
    print()
    print ("Question 4")
    print("\n" + "=" * 60)
    print(f"Q4: 30% training, 100% test, alpha={best_alpha}")
    print("=" * 60)

    (pos_train, neg_train, vocab) = load_training_set(0.3, 0.3)
    (pos_test, neg_test) = load_test_set(1.0, 1.0)

    print(f"Training: {len(pos_train)} pos, {len(neg_train)} neg")
    print(f"Test:     {len(pos_test)} pos, {len(neg_test)} neg")
    print(f"Vocab:    {len(vocab)} words")

    model_q4 = MultinomialNB(pos_train, neg_train, vocab, alpha=best_alpha)
    results_q4 = compute_matrices(pos_test, neg_test, model_q4, classify_log)
    print(results_q4)

    # Q6
    print()
    print ("Question 6")
    (pos_train, neg_train, vocab) = load_training_set(0.1, 0.5)
    (pos_test, neg_test) = load_test_set(1.0, 1.0)

    print(f"Training: {len(pos_train)} pos, {len(neg_train)} neg")
    print(f"Test:     {len(pos_test)} pos, {len(neg_test)} neg")
    print(f"Vocab:    {len(vocab)} words")

    model_q6 = MultinomialNB(pos_train, neg_train, vocab, alpha=best_alpha)
    results_q6 = compute_matrices(pos_test, neg_test, model_q6, classify_log)
    print(results_q6)