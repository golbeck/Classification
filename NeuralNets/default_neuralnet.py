import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import pydot 
import os
from os import system
import random
from sklearn.metrics import confusion_matrix
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
N=Xdf.shape
N=N[0]
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

#########################################################################################
#########################################################################################
#classification components
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
#graphical devices
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#create a dataset for use in pybrain
from pybrain.datasets import ClassificationDataSet
alldata=ClassificationDataSet(3,nb_classes=2,class_labels=['default_Yes','default_No'])
#classes are encoded into one output unit per class, that takes on a certain value if the class is present
#alldata._convertToOneOfMany(bounds=[0, 1])

#convert back to a single column of class labels
#alldata._convertToClassNb()

#Target dimension is supposed to be 1
#The targets are class labels starting from zero
for i in range(N):
    alldata.appendLinked(Xdf.ix[i,:],Ydf['default_Yes'].ix[i,:])
#generate training and testing data sets
tstdata, trndata = alldata.splitWithProportion(0.10)
#classes are encoded into one output unit per class, that takes on a certain value if the class is present
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
len(tstdata), len(trndata)
#calculate statistics and generate histograms
alldata.calculateStatistics()
print alldata.classHist
print alldata.nClasses
print alldata.getClass(1)

#########################################################################################
#########################################################################################
#use the quick start neural network to train using backward propagation and classify
from pybrain.supervised.trainers import BackpropTrainer
net = buildNetwork(trndata.indim, 30, trndata.outdim, bias=True, hiddenclass=SoftmaxLayer)
trainer = BackpropTrainer(net, trndata,learningrate=0.01, lrdecay=1.0, momentum=0.0)
trainer.train()
#apply trained network to computing error in training set
#train_output=trainer.testOnClassData(dataset=trndata,verbose=False,return_targets=True)
trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
print("  train error: %5.4f%%" % trnresult)

#apply the trained network to classifying the test set
#test_output=trainer.testOnClassData(dataset=tstdata,verbose=False,return_targets=True)
tstresult = percentError( trainer.testOnClassData(dataset=tstdata),tstdata['class'] )
print("  test error: %5.4f%%" % tstresult)
#trainer.trainUntilConvergence()
#########################################################################################
#########################################################################################
#########################################################################################
#build a feed-forward network with _ hidden units. We use the shortcut buildNetwork() for this. The input and output layer size must match the dataset’s input and target dimension. You could add additional hidden layers by inserting more numbers giving the desired layer sizes.
fnn = buildNetwork( trndata.indim, 30, trndata.outdim, outclass=SoftmaxLayer )

#Set up a trainer that basically takes the network and training dataset as input. For a list of trainers, see trainers. We are using a BackpropTrainer for this.
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

#Start the training iterations.
for i in range(30):
    trainer.trainEpochs( 1 )
    #Evaluate the network on the training and test data
    trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.4f%%" % trnresult, \
        "  test error: %5.4f%%" % tstresult

