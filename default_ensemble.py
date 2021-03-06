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
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

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
#default indicator
Ydf=pd.get_dummies(Default['default'],prefix='default').astype('float')

#training data
n_train=9000
np.random.seed(0)
rows = random.sample(Xdf.index, n_train)
Xdf_train=Xdf.ix[rows]
#drop intercept
Xdf_train=Xdf_train.drop('intercept',1)
Ydf_train=Ydf.ix[rows]

#test data
Xdf_test=Xdf.drop(rows)
#drop intercept
Xdf_test=Xdf_test.drop('intercept',1)
Ydf_test=Ydf.drop(rows)

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#use a decision tree with bagging
bagging = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=1.0, max_features=1.0, bootstrap=True)
bagging = bagging.fit(Xdf_train, Ydf_train['default_Yes'])
bagging_pred = pd.DataFrame(bagging.predict(Xdf_test))

#confusion matrix
cm = confusion_matrix(Ydf_test['default_Yes'],bagging_pred)
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix (bagging)')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#use the random forest

rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest = rand_forest.fit(Xdf_train, Ydf_train['default_Yes'])
rand_forest_pred = pd.DataFrame(rand_forest.predict(Xdf_test))

print pd.DataFrame(rand_forest.feature_importances_, columns = ["Imp"], index = Xdf_train.columns).sort(['Imp'], ascending = False)

#confusion matrix
cm = confusion_matrix(Ydf_test['default_Yes'],rand_forest_pred)
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix (random forest)')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#ADAboost tree classification
adaboost=AdaBoostClassifier(n_estimators=100)
adaboost=adaboost.fit(Xdf_train, Ydf_train['default_Yes'])
adaboost_pred=pd.DataFrame(adaboost.predict(Xdf_test))

print pd.DataFrame(adaboost.feature_importances_, columns = ["Imp"], index = Xdf_train.columns).sort(['Imp'], ascending = False)

#confusion matrix
cm = confusion_matrix(Ydf_test['default_Yes'],adaboost_pred)
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix (adaboost)')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#########################################################################################
#########################################################################################

#make contour plot of predictions
plot_colors = "br"
plot_step = 100
class_names = "01"

plt.figure(figsize=(10, 5))
# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = Xdf_test.ix[:, 0].min() - 1, Xdf_test.ix[:, 0].max() + 1
y_min, y_max = Xdf_test.ix[:, 2].min() - 1, Xdf_test.ix[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
#generate set of indicators for student=yes
stud_temp=[1.0]*len(xx.ravel())
Z = bagging.predict(np.c_[xx.ravel(),stud_temp,yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(Ydf_train['default_Yes'] == i)
    plt.scatter(Xdf_train[Xdf_train['student']==1.0].ix[idx].ix[:,0], Xdf_train[Xdf_train['student']==1.0].ix[idx].ix[:,2],c=c, cmap=plt.cm.Paired,label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel("Decision Boundary")
plt.show()



#########################################################################################
#########################################################################################

# Plot the decision boundaries
plt.subplot(122)
x_min, x_max = Xdf_test.ix[:, 0].min() - 1, Xdf_test.ix[:, 0].max() + 1
y_min, y_max = Xdf_test.ix[:, 2].min() - 1, Xdf_test.ix[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
#generate set of indicators for student=no
stud_temp=[0.0]*len(xx.ravel())
Z = bagging.predict(np.c_[xx.ravel(),stud_temp,yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(Ydf_train['default_Yes'] == i)
    plt.scatter(Xdf_train[Xdf_train['student']==0.0].ix[idx].ix[:,0], Xdf_train[Xdf_train['student']==0.0].ix[idx].ix[:,2],c=c, cmap=plt.cm.Paired,label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper left')
plt.xlabel("Decision Boundary")
plt.show()





#########################################################################################
#########################################################################################

# Plot the decision boundaries for only the training data with student=no
plt.figure()
x_min, x_max = Xdf_train[Xdf_train['student']==0.0].ix[:, 0].min() - 1, Xdf_train[Xdf_train['student']==0.0].ix[:, 0].max() + 1
y_min, y_max = Xdf_train[Xdf_train['student']==0.0].ix[:, 2].min() - 1, Xdf_train[Xdf_train['student']==0.0].ix[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
                     #generate set of indicators for student=no
stud_temp=[0.0]*len(xx.ravel())
Z = bagging.predict(np.c_[xx.ravel(),stud_temp,yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(Ydf_train['default_Yes'] == i)
    plt.scatter(Xdf_train[Xdf_train['student']==0.0].ix[idx].ix[:,0], Xdf_train[Xdf_train['student']==0.0].ix[idx].ix[:,2],c=c, cmap=plt.cm.Paired,label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper left')
plt.xlabel("Decision Boundary")
plt.show()
