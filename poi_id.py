#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")  # noqa
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

# Options for feature selction
features_list = ['poi',
                 'salary',
                 'bonus',
                 'total_payments',
                 'loan_advances',
                 'to_messages',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'deferral_payments',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'long_term_incentive',
                 'restricted_stock']

print "Features List", features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
#### Total No. of Data Points
print "No. of Data Points in the dataset is", len(data_dict)

#### Checking the Poi
num_poi = 0
for i in data_dict.values():
    if i['poi']==True:
        num_poi += 1
print "Total Person of Interest in the dataset is :", num_poi
print "Total N0. of non POIs in the dataset are:", len(data_dict)-num_poi

#### Missing Features
print "\n Missing Features of each cateory are:"

nan = [0 for i in range(len(features_list))]
for i in data_dict.values():
    for j,feature in enumerate(features_list):
        if i[feature] == 'NaN':
            nan[j] += 1

for i, feature in enumerate(features_list):
    print feature, nan[i]
    
#### Salary Drawn by POI
print "Names of peron of interest and Salary Drawn by them at ENRON are:"
for i,j in data_dict.items():
    if j['poi']:
        print i,':',j['salary']
        
#### Maximum, Minimum and Mean Salary
import statistics
sal = set()
for i in data_dict.values():
    if i['salary'] != 'NaN':
        sal.add(i['salary'])

print "Maximum Salary drawn by an employee at ENRON is :", max(sal)
print "Minimum Salary drawn by an employee at ENRON is :", min(sal)
print "Mean Salary of the employee at ENRON is :", statistics.mean(sal)

### Task 2: Remove outliers

#### Making a plot between salary and bonus
import matplotlib.pyplot as plt
feature = ['salary', 'bonus']
data = featureFormat(data_dict, feature)
for i in data:
    salary = i[0]
    bonus = i[1]
    plt.scatter(salary, bonus)
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.title('Bonus vs Salary Comaprison')
plt.show()

print "There is an outlier in the plot. So, we shall remove it first."

for i, v in data_dict.items():
    if v['salary'] != 'NaN' and v['salary'] > 10000000:
        print i

data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('TRAVEL AGENCY IN THE PARK', 0)

data = featureFormat(data_dict, feature)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title('Bonus vs Salary Comaprison')
plt.show()

feature1 = ['salary', 'loan_advances']
data1 = featureFormat(data_dict, feature1)
for i in data1:
    salary = i[0]
    loan_advances = i[1]
    plt.scatter(salary, loan_advances)
plt.xlabel('Salary')
plt.ylabel('Loan Advances')
plt.title('Loan vs Salary Comaprison')
plt.show()

print "\n There are outliers in this case too and the contributers are:"

for i, v in data_dict.items():
    if v['loan_advances'] != 'NaN' and v['loan_advances'] > 10000000:
        print i

print "\n Let's remove the outliers"

data_dict.pop('TOTAL',0)
data_dict.pop('LAY KENNETH L',0)

data1= featureFormat(data_dict, feature1)
for point in data1:
    salary = point[0]
    loan_advances = point[1]
    plt.scatter( salary, loan_advances )

plt.xlabel("salary")
plt.ylabel("Loan Advances")
plt.title('Loan vs Salary Comaprison')
plt.show()


feature2 = ['total_payments', 'loan_advances']
data1 = featureFormat(data_dict, feature2)
for i in data1:
    total_payments = i[0]
    loan_advances = i[1]
    plt.scatter(total_payments, loan_advances)
plt.xlabel('Total Payments')
plt.ylabel('Loan Advances')
plt.title('Loan vs Payment Comaprison')
plt.show()

print "\n The Outliers in this case are: "

for i, v in data_dict.items():
    if v['total_payments'] != 'NaN' and v['total_payments'] > 10000000:
        print i

print "\n Removing the Outliers"

data_dict.pop('LAVORATO JOHN J',0)
data_dict.pop('BHATNAGAR SANJAY',0)
data_dict.pop('FREVERT MARK A',0)

data2= featureFormat(data_dict, feature2)
for point in data2:
    total_payments = point[0]
    loan_advances = point[1]
    plt.scatter( total_payments, loan_advances )

plt.xlabel("Total Payments")
plt.ylabel("Loan Advances")
plt.title('Loan vs Payment Comaprison')
plt.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for i, j in my_dataset.iteritems():
    j['from_poi_to_this_person_ratio']=0
    if j['from_poi_to_this_person'] and j['to_messages'] != 'NaN':
        if j['to_messages'] == 0:
            print 'Imposiible Event'
        else:
           j['from_poi_to_this_person_ratio'] = float(j['from_poi_to_this_person'])/float(j['to_messages'])

for i, j in my_dataset.iteritems():
    j['from_this_person_to_poi_ratio']=0
    if j['from_this_person_to_poi'] and j['from_messages'] != 'NaN':
        if j['from_messages'] == 0:
            print 'Imposiible Event'
        else:
           j['from_this_person_to_poi_ratio'] = float(j['from_this_person_to_poi'])/float(j['from_messages'])

features_list.extend(['from_poi_to_this_person_ratio','from_this_person_to_poi_ratio'])

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
    clf = DecisionTreeClassifier()
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


def getBestFeatures(features, labels, num_features=15, showResults=False):
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
   
# convert dictionary into features and labels 
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Checking the scores for the original list of fearures
features_list = getBestFeatures(features, labels, 10, True)

# Seeing the scores for the modified features_list
scoreNumFeatures(features_list, my_dataset)

# convert dictionary into features and labels
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# select 8 best features
features_list = getBestFeatures(features, labels, 8, True)

# Re-split data based on new feature list after getBestFeatures
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

###
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size = 0.3, random_state=42)

####
#clf = GaussianNB()

clf = DecisionTreeClassifier()

#clf = RandomForestClassifier()

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "\n Accuracy is:", acc
print "precision = ", precision_score(labels_test,pred)
print "recall = ", recall_score(labels_test,pred)
print "Confusion matrix"
print confusion_matrix(labels_test, pred)
print "Classification report for is" 
print classification_report(labels_test, pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "\n After Tuning",

### For Decision Tree
clf1 = DecisionTreeClassifier(max_features=6, min_samples_split=4,
                                  criterion='entropy', max_depth=10,
                                  min_samples_leaf=2)
clf1 = clf1.fit(features_train,labels_train)
pred1 = clf1.predict(features_test)

"""
### For Random Forest
tuned_param = {"n_estimators":[2, 3, 5],  "criterion": ('gini', 'entropy')}
clf1 = GridSearchCV(clf, tuned_param)
clf1 = clf1.fit(features_train,labels_train)
pred1 = clf1.predict(features_test)
"""
# Example starting point. Try investigating other evaluation techniques!
acc1 = accuracy_score(pred1, labels_test)
print "\n Accuracy is:", acc1 
print "precision = ", precision_score(labels_test,pred1)
print "recall = ", recall_score(labels_test,pred1)
print "Confusion matrix"
print confusion_matrix(labels_test, pred1)
print "Classification report for is" 
print classification_report(labels_test, pred1)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
