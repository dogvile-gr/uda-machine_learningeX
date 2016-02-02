#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




import time
from functools import wraps

#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC
clf = SVC(C=10000,kernel="rbf")
start_time = time.time()

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train)
stop_time = time.time()
print("example run in %.2fs" % (stop_time - start_time))


pred = clf.predict(features_test)

stop_time = time.time()
print("example run in %.2fs" % (stop_time - start_time))

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print(acc)




