
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(dataset, unique_threshold=10):
    categorical_indices = []
    numerical_indices = []
    num_columns = dataset.shape[1]
    
    for col_idx in range(num_columns):
        col = dataset[:, col_idx]
        
        try:
            col.astype(float)
            if len(np.unique(col)) <= unique_threshold:
                categorical_indices.append(col_idx)
            else:
                numerical_indices.append(col_idx) 
                
        except ValueError:
            categorical_indices.append(col_idx)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'encoder', 
                OneHotEncoder(drop='if_binary', sparse_output=False), 
                categorical_indices
            )
        ],
        remainder='passthrough' 
    )

    clean_dataset = preprocessor.fit_transform(dataset)
    
    return clean_dataset.astype(float)

#This function will count the number of attributes in a dataset. It will help find the correct amount of neurons in the first layer
def num_attributes(dataset):
    headers = dataset[0]
    attribute_count = 0
   
    for word in headers:
        if word.lower() != "label":
            attribute_count += 1
            
    return attribute_count


#Takes as input a training dataset and a desired minibatch size. Then it will create minibatches of that size from the training dataset
def create_minibatch(dataset, minibatch_size):
    np.random.shuffle(dataset) 
    size = len(dataset)
    minibatches = []
    
    for i in range(0, size, minibatch_size):
        batch = dataset[i : i + minibatch_size]
        minibatches.append(batch)
        
    return minibatches


#Takes as input an array that indicated the number of hidden layers and the number of neurons in each hidden layer. It also takes as input the number of attribute in a the examined dataset. It outputs an array of arrays of the weights. 
def weights_generation(LN, num_attributes):
    prev = num_attributes
    weights = []
    
    for neurons in LN:
        layer_weights = np.random.uniform(-1, 1, size=(neurons, prev + 1))
        weights.append(layer_weights)
        prev = neurons
        
    return weights

#sigmoid function that ensures values are between 0 and 1
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


#Takes as input a minibatch and the weights array and returns a cache that stores all activation and pre-activation states of the nodes after 1 iteration of all minibatches. 
def forward_propagation(minibatch_X, weights):

    cache = {}
    m = minibatch_X.shape[0]
    bias_column = np.ones((m, 1))
    A_curr = np.hstack((bias_column, minibatch_X))
    cache['A0'] = A_curr
    
    for l in range(len(weights)):
        W = weights[l]
        
        Z = np.dot(A_curr, W.T)
        
        cache[f'Z{l+1}'] = Z 
        
        A_next = sigmoid(Z)
        
        if l < len(weights) - 1:
            bias_column = np.ones((m, 1))
            A_next = np.hstack((bias_column, A_next))
            
        cache[f'A{l+1}'] = A_next
        A_curr = A_next
        
    final_prediction = A_curr

    return final_prediction, cache


#Finds deltas for all nodes in the NN. 
def find_deltas(final_prediction, cache, weights, labels):

    deltas = {}
    L = len(weights) 
    
    deltas[f'delta{L}'] = final_prediction - labels
    
    #print(f"delta{L+1}: \n{deltas[f'delta{L}']}\n")
    for l in reversed(range(1, L)):
        delta_next = deltas[f'delta{l+1}']
        W_next = weights[l]
        W_next_no_bias = W_next[:, 1:]
        A_curr = cache[f'A{l}']
        A_curr_no_bias = A_curr[:, 1:]
        sigmoid_derivative = A_curr_no_bias * (1 - A_curr_no_bias)
        delta_curr = np.dot(delta_next, W_next_no_bias) * sigmoid_derivative
        deltas[f'delta{l}'] = delta_curr
        #print(f"delta{l+1}: \n{delta_curr}\n")
        
    return deltas

#computes the average gradients for minibatch
def compute_gradients(deltas, cache, weights, lambda_reg):
    m = cache['A0'].shape[0] 
    
    gradients = []
    L = len(weights) 
    
    for l in range(L):
        
        A_curr = cache[f'A{l}']
        delta_next = deltas[f'delta{l+1}']
        dW = np.dot(delta_next.T, A_curr) / m
        W_curr = weights[l]
        W_penalty = np.copy(W_curr)
        W_penalty[:, 0] = 0 
        dW += (lambda_reg / m) * W_penalty
    
        gradients.append(dW)
        
    return gradients


#updates the weights using the calculated gradients and a hyperparameter alpha
def update_weights(weights, gradients, alpha):
    updated_weights = []
    L = len(weights) 
    
    for l in range(L):
        new_W = weights[l] - (alpha * gradients[l])
        updated_weights.append(new_W)
        
    return updated_weights



def k_fold_training(X, y, LN, alpha, lambda_reg, epochs, minibatch_size, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        num_attributes = X_train.shape[1]
        weights = weights_generation(LN, num_attributes)
        
        train_dataset = np.hstack((X_train, y_train.reshape(-1, 1)))
        
        for epoch in range(epochs):
            
            minibatches = create_minibatch(train_dataset, minibatch_size)
            
            for batch in minibatches:
                minibatch_X = batch[:, :-1] 
                labels = batch[:, -1:]      
                
                final_prediction, cache = forward_propagation(minibatch_X, weights)
                deltas = find_deltas(final_prediction, cache, weights, labels)
                gradients = compute_gradients(deltas, cache, weights, lambda_reg)
                weights = update_weights(weights, gradients, alpha)
                
        test_predictions, _ = forward_propagation(X_test, weights)
        
        predicted_classes = (test_predictions >= 0.5).astype(int)
        
        correct_predictions = np.sum(predicted_classes == y_test.reshape(-1, 1))
        accuracy = correct_predictions / len(y_test)
        fold_accuracies.append(accuracy)
        
        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")
        
    average_accuracy = np.mean(fold_accuracies)
    print(f"\n=== Final Evaluation ===")
    print(f"Average Accuracy across {k} folds: {average_accuracy * 100:.2f}%")
    
    return average_accuracy

def compute_cost(y_true, y_pred):
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    cost = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.sum(cost)

def compute_regularized_cost(y_true, y_pred, weights, lambda_reg):
    m = y_true.shape[0]
    base_cost = compute_cost(y_true, y_pred) / m
    reg_penalty = 0
    for w in weights:
        w_no_bias = w[:, 1:]
        reg_penalty += np.sum(w_no_bias ** 2)
        
    reg_penalty = (lambda_reg / (2 * m)) * reg_penalty
    return base_cost + reg_penalty

def verify_correctness(X, y, initial_weights, lambda_reg):
    """
    Reproduces the step-by-step output required for backpropagation verification.
    """
    print(f"Regularization parameter lambda={lambda_reg:.3f}\n")
 
    for i, w in enumerate(initial_weights):
        print(f"Initial Theta{i+1} (the weights of each neuron, including the bias weight, are stored in the rows):")
        for row in w:
            print("\t" + "  ".join([f"{val:.5f}" for val in row]) + "  ")
        print()

    print("Training set")
    for i in range(len(X)):
        print(f"\tTraining instance {i+1}")
        print(f"\t\tx: {X[i]}")
        print("\t\ty: [" + "   ".join([f"{val:.5f}" for val in y[i]]) + "]")
    print("\n--------------------------------------------")
    
    print("Computing the error/cost, J, of the network")
    
    for i in range(len(X)):
        print(f"\tProcessing training instance {i+1}")
        x_i = X[i:i+1]
        y_i = y[i:i+1]
        print(f"\tForward propagating the input {x_i[0]}")
        
        final_prediction, cache = forward_propagation(x_i, initial_weights)
        
        print(f"\t\ta1: {np.round(cache['A0'][0], 5)}")
        for l in range(1, len(initial_weights) + 1):
            print(f"\n\t\tz{l+1}: {np.round(cache[f'Z{l}'][0], 5)}")
            print(f"\t\ta{l+1}: {np.round(cache[f'A{l}'][0], 5)}")

        print(f"\n\t\tf(x): {np.round(final_prediction[0], 5)}")
        print("\tPredicted output for instance " + str(i+1) + ": [" + "   ".join([f"{val:.5f}" for val in final_prediction[0]]) + "]")
        print("\tExpected output for instance " + str(i+1) + ": [" + "   ".join([f"{val:.5f}" for val in y_i[0]]) + "]")
        
        cost_j = compute_cost(y_i[0], final_prediction[0])
        print(f"\tCost, J, associated with instance {i+1}: {cost_j:.3f}\n")

    full_prediction, _ = forward_propagation(X, initial_weights)
    total_cost = compute_regularized_cost(y, full_prediction, initial_weights, lambda_reg)
    print(f"Final (regularized) cost, J, based on the complete training set: {total_cost:.5f}\n")
    print("--------------------------------------------")
    print("Running backpropagation")

    for i in range(len(X)):
         print(f"\tComputing gradients based on training instance {i+1}")
         x_i = X[i:i+1]
         y_i = y[i:i+1]
         
         final_prediction, cache = forward_propagation(x_i, initial_weights)
         deltas = find_deltas(final_prediction, cache, initial_weights, y_i)
        
         L = len(initial_weights)
         for l in reversed(range(1, L + 1)):
             print(f"\t\tdelta{l+1}: {np.round(deltas[f'delta{l}'][0], 5)}")
             
         print()
         inst_gradients = compute_gradients(deltas, cache, initial_weights, lambda_reg=0)
         
         for l in reversed(range(L)):
             print(f"\t\tGradients of Theta{l+1} based on training instance {i+1}:")
             for row in inst_gradients[l]:
                 print("\t\t\t" + "  ".join([f"{val:.5f}" for val in row]) + "  ")
         print()

    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    full_prediction, full_cache = forward_propagation(X, initial_weights)
    full_deltas = find_deltas(full_prediction, full_cache, initial_weights, y)
    final_gradients = compute_gradients(full_deltas, full_cache, initial_weights, lambda_reg)
    
    for l in range(len(initial_weights)):
        print(f"\t\tFinal regularized gradients of Theta{l+1}:")
        for row in final_gradients[l]:
            print("\t\t\t" + "  ".join([f"{val:.5f}" for val in row]) + "  ")
        print()
