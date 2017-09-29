#!/usr/bin/python

import sys
import pickle
import random
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'total_payments',
                 'loan_advances',
                 'to_messages',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi']

print "\n Features List", features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#### Checking the Poi
num_poi = 0
for i in data_dict.values():
    if i['poi']==True:
        num_poi += 1
print "\n Total Person of Interest in the dataset is :", num_poi

#### Salary Drawn by POI
print "\n Names of peron of interest and Salary Drawn by them at ENRON are:"
for i,j in data_dict.items():
    if j['poi']:
        print i,':',j['salary']
        
#### Maximum, Minimum and Mean Salary
import statistics
sal = set()
for i in data_dict.values():
    if i['salary'] != 'NaN':
        sal.add(i['salary'])

print "\n Maximum Salary drawn by an employee at ENRON is :", max(sal)
print "\n Minimum Salary drawn by an employee at ENRON is :", min(sal)
print "\n Mean Salary of the employee at ENRON is :", statistics.mean(sal)

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

print "\n There is an outlier in the plot. So, we shall remove it first."

for i, v in data_dict.items():
    if v['salary'] != 'NaN' and v['salary'] > 10000000:
        print i

data_dict.pop('TOTAL', 0)

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

print"\n Cleaned Plot is"

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

print "\n Cleaned Plot is:"

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

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#acc = accuracy_score(pred, labels_test)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#acc = accuracy_score(pred, labels_test)

from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

print "precision = ", precision_score(labels_test,pred)
print "recall = ", recall_score(labels_test,pred)
print "\n Accuracy is:", acc

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### For Decision Tree
clf1 = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=2)
clf1 = clf1.fit(features_train,labels_train)
pred1 = clf1.predict(features_test)

### For Random Forest
#clf2 = {"n_estimators":[2, 3, 5],  "criterion": ('gini', 'entropy')}
#clf = GridSearchCV(clf, clf2)

# Example starting point. Try investigating other evaluation techniques!

print "Confusion matrix"
print confusion_matrix(labels_test, pred1)
print "Classification report for %s" % clf1
print classification_report(labels_test, pred1)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
