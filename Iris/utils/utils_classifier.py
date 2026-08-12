"""
Functions for machine learning classifier.
"""
# Imports:
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc

# Functions:
def calculate_accuracy(observed, predicted):
    
    """
    Calculates a range of accuracy scores from observed and predicted classes.
    
    Takes two list or NumPy arrays (observed class values, and predicted class 
    values), and returns a dictionary of results.
    
     1) observed positive rate: proportion of observed cases that are +ve
     2) Predicted positive rate: proportion of predicted cases that are +ve
     3) observed negative rate: proportion of observed cases that are -ve
     4) Predicted negative rate: proportion of predicted cases that are -ve  
     5) accuracy: proportion of predicted results that are correct    
     6) precision: proportion of predicted +ve that are correct
     7) recall: proportion of true +ve correctly identified
     8) f1: harmonic mean of precision and recall
     9) sensitivity: Same as recall
    10) specificity: Proportion of true -ve identified:        
    11) positive likelihood: increased probability of true +ve if test +ve
    12) negative likelihood: reduced probability of true +ve if test -ve
    13) false positive rate: proportion of false +ves in true -ve patients
    14) false negative rate: proportion of false -ves in true +ve patients
    15) true positive rate: Same as recall
    16) true negative rate: Same as specificity
    17) positive predictive value: chance of true +ve if test +ve
    18) negative predictive value: chance of true -ve if test -ve
    
    """
    
    # Converts list to NumPy arrays
    if type(observed) == list:
        observed = np.array(observed)
    if type(predicted) == list:
        predicted = np.array(predicted)
    
    # Calculate accuracy scores
    observed_positives = observed == 1
    observed_negatives = observed == 0
    predicted_positives = predicted == 1
    predicted_negatives = predicted == 0
    
    true_positives = (predicted_positives == 1) & (observed_positives == 1)
    
    false_positives = (predicted_positives == 1) & (observed_positives == 0)
    
    true_negatives = (predicted_negatives == 1) & (observed_negatives == 1)
    
    false_negatives = (predicted_negatives == 1) & (observed_negatives == 0)
    
    accuracy = np.mean(predicted == observed)
    
    precision = (np.sum(true_positives) /
                 (np.sum(true_positives) + np.sum(false_positives)))
        
    recall = np.sum(true_positives) / np.sum(observed_positives)
    
    sensitivity = recall
    
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    specificity = np.sum(true_negatives) / np.sum(observed_negatives)

    # Use "if" statement to prevent warning for divide by zero.
    positive_likelihood = (sensitivity / (1 - specificity)
                           if specificity != 1.0 else np.inf)
    
    negative_likelihood = (1 - sensitivity) / specificity
    
    false_positive_rate = 1 - specificity
    
    false_negative_rate = 1 - sensitivity
    
    true_positive_rate = sensitivity
    
    true_negative_rate = specificity
    
    positive_predictive_value = (np.sum(true_positives) / 
                            (np.sum(true_positives) + np.sum(false_positives)))
    
    negative_predictive_value = (np.sum(true_negatives) / 
                            (np.sum(true_negatives) + np.sum(false_negatives)))
    
    # Create dictionary for results, and add results
    results = dict()
    
    results['observed_positive_rate'] = np.mean(observed_positives)
    results['observed_negative_rate'] = np.mean(observed_negatives)
    results['predicted_positive_rate'] = np.mean(predicted_positives)
    results['predicted_negative_rate'] = np.mean(predicted_negatives)
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['sensitivity'] = sensitivity
    results['specificity'] = specificity
    results['positive_likelihood'] = positive_likelihood
    results['negative_likelihood'] = negative_likelihood
    results['false_positive_rate'] = false_positive_rate
    results['false_negative_rate'] = false_negative_rate
    results['true_positive_rate'] = true_positive_rate
    results['true_negative_rate'] = true_negative_rate
    results['positive_predictive_value'] = positive_predictive_value
    results['negative_predictive_value'] = negative_predictive_value
    
    return results

def measure_accuracy_from_kfolds(X, y, feature_names, number_of_splits=10):
    """
    Run k-fold validation and calculate mean accuracy & importance.
    """
    # Set up lists to hold results for each k-fold run
    replicate_accuracy = []
    replicate_precision = []
    replicate_recall = []
    replicate_f1 = []
    replicate_predicted_positive_rate = []
    replicate_observed_positive_rate = []
    
    # Set up DataFrame for feature importances
    importances = pd.DataFrame(index = feature_names)
    
    # Set up splits
    skf = StratifiedKFold(n_splits = number_of_splits)
    skf.get_n_splits(X, y)
    
    # Loop through the k-fold splits
    k_fold_count = 0
    
    for train_index, test_index in skf.split(X, y):   
        # Get X and Y train/test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Set up and fit model (n_jobs=-1 uses all cores on a computer)
        model = XGBClassifier(random_state=42, verbosity=0)
        model.fit(X_train, y_train)
    
        # Predict test set labels and get accuracy scores
        y_pred_test = model.predict(X_test)
        accuracy_scores = calculate_accuracy(y_test, y_pred_test)
        # Store results:
        replicate_accuracy.append(accuracy_scores['accuracy'])
        replicate_precision.append(accuracy_scores['precision'])
        replicate_recall.append(accuracy_scores['recall'])
        replicate_f1.append(accuracy_scores['f1'])
        replicate_predicted_positive_rate.append(
            accuracy_scores['predicted_positive_rate'])
        replicate_observed_positive_rate.append(
            accuracy_scores['observed_positive_rate'])

        # Record feature importances
        col_title = 'split_' + str(k_fold_count)
        importances[col_title] = model.feature_importances_
        k_fold_count +=1

    # Transfer results to list and add to data frame
    results = pd.Series()
    results['accuracy'] = np.mean(replicate_accuracy)
    results['precision'] = np.mean(replicate_precision)
    results['recall'] = np.mean(replicate_recall)
    results['f1'] = np.mean(replicate_f1)
    results['predicted positive rate'] = np.mean(replicate_predicted_positive_rate)
    results['observed positive rate'] = np.mean(replicate_observed_positive_rate)
    # Round values:
    results = results.round(5)
    
    # Get average of feature importances, and sort
    importance_mean = importances.mean(axis=1)
    importance_mean.sort_values(ascending=False, inplace=True)
    # Round values:
    importance_mean = importance_mean.round(5)
    return results, importance_mean

from sklearn.metrics import auc


def run_roc_auc(X_np, y_np, number_of_splits=5):
    # Set up k-fold training/test splits
    
    skf = StratifiedKFold(n_splits = number_of_splits)
    skf.get_n_splits(X_np, y_np)
    
    # Set up thresholds
    thresholds = np.arange(0, 1.01, 0.01)
    
    # Create arrays for overall results (rows=threshold, columns=k fold replicate)
    results_accuracy = np.zeros((len(thresholds),number_of_splits))
    results_precision = np.zeros((len(thresholds),number_of_splits))
    results_recall = np.zeros((len(thresholds),number_of_splits))
    results_f1 = np.zeros((len(thresholds),number_of_splits))
    results_predicted_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_observed_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_true_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_false_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_auc = []
    
    # Loop through the k-fold splits
    loop_index = 0
    for train_index, test_index in skf.split(X_np, y_np):
        
        # Create lists for k-fold results
        threshold_accuracy = []
        threshold_precision = []
        threshold_recall = []
        threshold_f1 = []
        threshold_predicted_positive_rate = []
        threshold_observed_positive_rate = []
        threshold_true_positive_rate = []
        threshold_false_positive_rate = []
    
        # Get X and Y train/test
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
    
        # Set up and fit model (n_jobs=-1 uses all cores on a computer)
        model = XGBClassifier(random_state=42, verbosity=0)
        model.fit(X_train,y_train)
        
        # Get probability of non-survive and survive
        probabilities = model.predict_proba(X_test)
        # Take just the survival probabilities (column 1)
        probability_survival = probabilities[:,1]
        
        # Loop through increments in probability of survival
        for cutoff in thresholds: #  loop 0 --> 1 on steps of 0.1
            # Get whether passengers survive using cutoff
            predicted_survived = probability_survival >= cutoff
            # Call accuracy measures function
            accuracy = calculate_accuracy(y_test, predicted_survived)
            # Add accuracy scores to lists
            threshold_accuracy.append(accuracy['accuracy'])
            threshold_precision.append(accuracy['precision'])
            threshold_recall.append(accuracy['recall'])
            threshold_f1.append(accuracy['f1'])
            threshold_predicted_positive_rate.append(
                    accuracy['predicted_positive_rate'])
            threshold_observed_positive_rate.append(
                    accuracy['observed_positive_rate'])
            threshold_true_positive_rate.append(accuracy['true_positive_rate'])
            threshold_false_positive_rate.append(accuracy['false_positive_rate'])
        
        # Add results to results arrays
        results_accuracy[:,loop_index] = threshold_accuracy
        results_precision[:, loop_index] = threshold_precision
        results_recall[:, loop_index] = threshold_recall
        results_f1[:, loop_index] = threshold_f1
        results_predicted_positive_rate[:, loop_index] = \
            threshold_predicted_positive_rate
        results_observed_positive_rate[:, loop_index] = \
            threshold_observed_positive_rate
        results_true_positive_rate[:, loop_index] = threshold_true_positive_rate
        results_false_positive_rate[:, loop_index] = threshold_false_positive_rate
        
        # Calculate ROC AUC
        roc_auc = auc(threshold_false_positive_rate, threshold_true_positive_rate)
        results_auc.append(roc_auc)
        
        # Increment loop index
        loop_index += 1
        
    
    # Transfer summary results to dataframe
    results = pd.DataFrame(thresholds, columns=['thresholds'])
    results['accuracy'] = results_accuracy.mean(axis=1)
    results['precision'] = results_precision.mean(axis=1)
    results['recall'] = results_recall.mean(axis=1)
    results['f1'] = results_f1.mean(axis=1)
    results['predicted_positive_rate'] = \
        results_predicted_positive_rate.mean(axis=1)
    results['observed_positive_rate'] = \
        results_observed_positive_rate.mean(axis=1)
    results['true_positive_rate'] = results_true_positive_rate.mean(axis=1)
    results['false_positive_rate'] = results_false_positive_rate.mean(axis=1)
    # results['roc_auc'] = results_auc  # NOTE - looks wrong in Titanic book.
    
    mean_auc = np.mean(results_auc)
    mean_auc = np.round(mean_auc, 3)
    return thresholds, results, results_auc, mean_auc, results_false_positive_rate, results_true_positive_rate