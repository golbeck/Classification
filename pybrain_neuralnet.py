from pybrain.structure import FeedForwardNetwork
n=FeedForwardNetwork()

#constructing the input, hidden and output layers
from pybrain.structure import LinearLayer, SigmoidLayer
inLayer = LinearLayer(2,name="input_nodes")
hiddenLayer = SigmoidLayer(3,name="hidden_nodes1")
outLayer = LinearLayer(1,name="output_node")

#add layers to the network
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

#explicitly determine how the layers should be connected
from pybrain.structure import FullConnection
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

#add the connections to the network
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()

print n

#feed an input to the network
#the weights/parameters of the connections have already been initialized randomly
n.activate([1, 2])
#show the parameters of the connections
in_to_hidden.params
hidden_to_out.params
#this will show all of the parameters in a single array
n.params

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
from numpy.random import multivariate_normal

#generate 2d data from MVN with 3 different parametrizations
means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])

tstdata, trndata = alldata.splitWithProportion( 0.25 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

#build a feed-forward network with 5 hidden units. We use the shortcut buildNetwork() for this. The input and output layer size must match the dataset’s input and target dimension. You could add additional hidden layers by inserting more numbers giving the desired layer sizes.
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#Set up a trainer that basically takes the network and training dataset as input. For a list of trainers, see trainers. We are using a BackpropTrainer for this.
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

#Start the training iterations.
for i in range(20):
    trainer.trainEpochs( 1 )
    #Evaluate the network on the training and test data
    trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.2f%%" % trnresult, \
        "  test error: %5.2f%%" % tstresult
    #Run our grid data through the FNN, get the most likely class and shape it into a square array again.
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)
    #Now plot the test data and the underlying grid as a filled contour.
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class']==c)
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot

ioff()
show()
