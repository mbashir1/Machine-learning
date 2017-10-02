#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

def getTrainingTestSets(labels, features):
    """ Creates training and test sets based on the StratifiedShuffleSplit
    args:
        labels: list of labels from the data
        features: list of features in the data
    """
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
    return features_train, features_test, labels_train, labels_test


def scoreNumFeatures(test_feature_list, test_data_set):
    """ function for determining the best number of features to use
    """
    scaler = MinMaxScaler()
    recall_scores = []
    precision_scores = []
    feature_count = []
    f1_scores = []
    PERF_FORMAT_STRING = "\
    Features: {:>0.{display_precision}f}\t\
    Accuracy: {:>0.{display_precision}f}\t\
    Precision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\t\
    F1: {:>0.{display_precision}f}\t\
    "

    
    clf = RandomForestClassifier()
    for x in range(1, len(test_feature_list)):
        test_data = featureFormat(test_data_set, test_feature_list,
                                  sort_keys=True)
        test_labels, test_features = targetFeatureSplit(test_data)
        test_features = scaler.fit_transform(test_features)
        best_features = getBestFeatures(test_features, test_labels, x, False)
        # Resplit data using best feature list
        test_data = featureFormat(test_data_set, best_features,
                                  sort_keys=True)
        test_labels, test_features = targetFeatureSplit(test_data)
        test_features = scaler.fit_transform(test_features)
        total_predictions, accuracy, precision, recall, true_positives, \
            false_positives, true_negatives, false_negatives, f1, f2 = \
            test_classifier(clf, test_features, test_labels)
        print PERF_FORMAT_STRING.format(x, accuracy, precision, recall, f1,
                                        display_precision=5)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)
        feature_count.append(x)

    plt.plot(feature_count, recall_scores, marker='o', label="Recall")
    plt.plot(feature_count, precision_scores, marker='o', label="Precision")
    plt.plot(feature_count, f1_scores, marker='o', label="F1")
    plt.legend()
    plt.show()


def test_classifier(clf, features, labels):
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = np.sum([true_negatives, false_negatives,
                                    false_positives, true_positives])
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives +
                                   false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return total_predictions, accuracy, precision, recall,\
            true_positives, false_positives, true_negatives, \
            false_negatives, f1, f2
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of \
        true positive predicitons."


def getBestFeatures(features, labels, num_features=19, showResults=False):
    """ Returns the best features based on the Feature Selection Options
    The features are selected based on the highest score / importance
    args:
        labels: list of labels from the data
        features: list of features in the data
        showResults: boolean set to true to print list of features and scores
    """
    features_train, features_test, labels_train, labels_test = \
        getTrainingTestSets(labels, features)
    revised_feature_list = ['poi']
    k_best = SelectKBest(k=num_features)
    k_best.fit(features_train, labels_train)
    importance = k_best.scores_
    
    feature_scores = sorted(zip(features_list[1:], importance),
                            key=lambda l: l[1], reverse=True)
    for feature, importance in feature_scores[:num_features]:
        revised_feature_list.append(feature)
    if showResults:
        print "Top features and scores:"
        print "==================================="
        pprint.pprint(feature_scores[:num_features])
    return revised_feature_list



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

scoreNumFeatures(features_list, my_dataset)

# convert dictionary into features and labels
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# select best features
features_list = getBestFeatures(features, labels, 10, True)
