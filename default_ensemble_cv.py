# http://scikit-learn.org/stable/modules/ensemble.html#adaboost
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#example-ensemble-plot-adaboost-twoclass-py
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import pylab as pl
import pydot 
import os
from os import system
import random
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#download default data and create a matrix of predictors
Default=pd.read_csv("Default.csv")
my_cols=set(Default.columns)
my_cols.remove('Unnamed: 0')
my_cols.remove('default')
my_cols=list(my_cols)
X=Default[my_cols]
X=np.array(X)

#convert matrix of predictors to a dataframe
Xdf=pd.DataFrame(X)
Xdf.columns=['balance','student','income']
#create dummy variables from the 'student' column
dummy_ranks=pd.get_dummies(Xdf['student'],prefix='student')
#removed 'student' column
Xdf.drop('student',1)
#create new 'student' column using one of the resulting dummy variable outputs
Xdf['student']=dummy_ranks['student_Yes']
#create a column for the intercept in the logit model
Xdf['intercept']=1.0
#convert all data to floats
Xdf=Xdf.astype('float')
#training data
n=Xdf.shape
n=n[0]
#drop intercept
Xdf=Xdf.drop('intercept',1)

#scale data
#first determine the parameters of the scaling
scaler_balance = preprocessing.StandardScaler().fit(Xdf['balance'])
#using the scaling parameters, transform the data
scaler_balance.transform(Xdf['balance'])
#perform the inverse transform
scaler_balance.inverse_transform(Xdf['balance'])

#first determine the parameters of the scaling
scaler_income = preprocessing.StandardScaler().fit(Xdf['income'])
#using the scaling parameters, transform the data
scaler_income.transform(Xdf['income'])
#perform the inverse transform
scaler_income.inverse_transform(Xdf['income'])

#default indicator
Ydf=pd.get_dummies(Default['default'],prefix='default').astype('float')

#parameters for cross validation and K-folds
cv=10
n_folds=10
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#use a decision tree with bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#built-in cross validation routine
clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=1.0, max_features=1.0, bootstrap=True)
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=1.0, max_features=1.0, bootstrap=True)
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("bagging")
print(np.mean(score))

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#use the random forest
from sklearn.ensemble import RandomForestClassifier

#built-in cross validation routine
clf = RandomForestClassifier(n_estimators=100)
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = RandomForestClassifier(n_estimators=100)
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("random forest")
print(np.mean(score))



#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#use the extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier

#built-in cross validation routine
clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
    min_samples_split=1, random_state=0).fit(Xdf_train, Ydf_train['default_Yes'])
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
        min_samples_split=1, random_state=0).fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("extra trees")
print(np.mean(score))


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#ADAboost tree classification
from sklearn.ensemble import AdaBoostClassifier

#built-in cross validation routine
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = AdaBoostClassifier(n_estimators=100)
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("ADAboost")
print(np.mean(score))

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#GradientBoostingClassifier classification
from sklearn.ensemble import GradientBoostingClassifier

#built-in cross validation routine
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(Xdf_train, Ydf_train['default_Yes'])
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("Gradient boosting")
print(np.mean(score))


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#support vector classification
#LinearSVC using one-against-all classification
from sklearn import svm

#built-in cross validation routine
clf = svm.LinearSVC()
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = svm.LinearSVC()
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("LinearSVC")
print(np.mean(score))

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#support vector classification
#SVC using one-against-one classification
from sklearn import svm

#built-in cross validation routine
clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = svm.SVC()
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("SVC")
print(np.mean(score))


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#support vector classification
#SVC with radial basis function
from sklearn import svm
from sklearn.grid_search import GridSearchCV

#use a grid search to determine the best parameters for the SVC classifier
param_grid={'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(Xdf, Ydf['default_Yes'])
print("The best classifier is: ", grid.best_estimator_)
C=grid.best_estimator_.C
gamma=grid.best_estimator_.gamma

#built-in cross validation routine
clf = svm.SVC(kernel='rbf',C=C,gamma=gamma)
scores = cross_validation.cross_val_score(clf, Xdf, Ydf['default_Yes'], cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#use KFold to generate validation sets
score=[]
kf = KFold(n, n_folds=n_folds)
for train, test in kf:
    clf = svm.SVC(kernel='rbf',C=C,gamma=gamma)
    Xdf_train=Xdf.ix[train]
    Ydf_train=Ydf.ix[train]
    clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
    Xdf_test=Xdf.ix[test]
    Ydf_test=Ydf.ix[test]
    clf_pred = pd.DataFrame(clf.predict(Xdf_test))
    score.append(accuracy_score(Ydf_test['default_Yes'],clf_pred))

print("SVC")
print(np.mean(score))
