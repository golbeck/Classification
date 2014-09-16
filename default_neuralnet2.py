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
#########################################################################################
#########################################################################################
#construct the network
from pybrain.structure import FeedForwardNetwork
net=FeedForwardNetwork()

#constructing the input, hidden and output layers
from pybrain.structure import LinearLayer, SigmoidLayer
inLayer = LinearLayer(3,name="input_nodes")
hiddenLayer1 = SigmoidLayer(3,name="hidden_nodes1")
hiddenLayer2 = SigmoidLayer(3,name="hidden_nodes1")
outLayer = LinearLayer(2,name="output_node")

#add layers to the network
net.addInputModule(inLayer)
net.addModule(hiddenLayer1)
net.addModule(hiddenLayer2)
net.addOutputModule(outLayer)

#explicitly determine how the layers should be connected
from pybrain.structure import FullConnection
in_to_hidden = FullConnection(inLayer, hiddenLayer1)
hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
hidden_to_out = FullConnection(hiddenLayer2, outLayer)

#add the connections to the network
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_hidden)
net.addConnection(hidden_to_out)
net.sortModules()

print net

#feed an input to the network
#the weights/parameters of the connections have already been initialized randomly
net.activate([1, 2, 3])
#show the parameters of the connections
in_to_hidden.params
hidden_to_out.params
#this will show all of the parameters in a single array
net.params


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
