import sys, pdb
import pandas as pd
import numpy as np
from sklearn import cross_validation, preprocessing

nielsen_path = '/home/kcrum/coding_space/neural-networks-and-deep-learning/'
sys.path.insert(0, nielsen_path + 'code')
# The previous line allows you to import the network class.                 
#import mynetwork as nielsen_net
import network2 as nielsen_net
import mnist_loader

percentCV = 0.3 # Fraction of training data for cross-validation
nhidden = 50    # Number of neurons in hidden layer
nclasses = 7    # Number of output classes
nbatch = 10     # Size of mini-batches (smaller -> more precise)
nepochs = 30    # Number of epochs
eta = 0.05      # Learning rate (at eta = 1000, epochs stuck at 659/4536)
lmbda = 5.0

Debug=False

# Convert input training dataframe (with classes) to Nielsen-style training 
# array. Nielsen uses lists of 2-tuples, where the tuples hold two np.arrays. 
# These two np.arrays hold individual np.arrays themselves. Is this really the 
# best way to do this???
def train_pd_to_nielsen(featuredf, classdf, maxval):
    featarr = np.asarray(featuredf)
    classarr = np.asarray(classdf)
    outlist = [] # Outermost list
    #print 'Incoming length: ', len(classarr)

    for feat, cls in zip(featarr, classarr):
        #### TEST STATEMENT ####
        #if (cls != 5 and cls != 6):
        #    continue
        #### END TEST STATEMET ####

        tempclsarr = np.zeros((maxval+1,1)) # np.array for response vector. To 
        # be filled with individual np.arrays of 0 or 1.
        for i in xrange(maxval+1):
            if i == cls:
                tempclsarr[i] = np.array([1.])
            else:
                tempclsarr[i] = np.array([0.])
        tempfeatarr = np.zeros((len(feat),1)) # np.array for response vector.
        # To be filled with inidvidual np.arrays of feature variables.
        for i in xrange(len(feat)):
            tempfeatarr[i] = np.array([feat[i]])

        outlist.append((tempfeatarr,tempclsarr)) # Create tuple here

    #print 'Outgoing length: ', len(outlist)    
    return outlist

def test_pd_to_nielsen(featuredf, classdf):
    featarr = np.asarray(featuredf)
    classarr = np.asarray(classdf)
    outlist = [] # Outermost list

    for feat, cls in zip(featarr, classarr):
        tempfeatarr = np.zeros((len(feat),1)) # np.array for response vector.
        # To be filled with inidvidual np.arrays of feature variables.
        for i in xrange(len(feat)):
            tempfeatarr[i] = np.array([feat[i]])

        outlist.append((tempfeatarr, cls)) # Create tuple here
    return outlist

# Test function to ensure *_to_nielsen(...) functions work
def convert_test():  
    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target

    newtrain = train_pd_to_nielsen(trainx, trainy, trainy.max())

    print np.asarray(newtrain).shape
    print np.asarray(newtrain)[0][0].shape
    print np.asarray(newtrain)[0][1].shape

    nisttrain, nistvalid, nisttest = mnist_loader.load_data_wrapper(nielsen_path + 'data/mnist.pkl.gz')
    print np.asarray(nisttrain).shape
    print np.asarray(nisttrain)[0][0].shape
    print np.asarray(nisttrain)[0][1].shape

    print '-'*50
    newtest = test_pd_to_nielsen(trainx, trainy)

    print np.asarray(newtest).shape
    print np.asarray(newtest)[0][0].shape
    print np.asarray(newtest)[0][1].shape

    print np.asarray(nistvalid).shape
    print np.asarray(nistvalid)[0][0].shape
    print np.asarray(nistvalid)[0][1].shape


def main():    
    fulltrain = pd.read_csv('dat/train.csv')
    # Read in data
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'].apply(lambda x: x-1) # Target
    
    # Split data for cross-validation 
    xtraincv, xtestcv, ytraincv, ytestcv = cross_validation.train_test_split(\
        trainx, trainy, test_size=percentCV, random_state=0)
    
    # xtraincv[:,:10] # <---- Non-binary columns
    # xtraincv[:,10:] # <---- Binary columns
    trainfloats = xtraincv[:,:10].astype(float)
    trainbinaries = xtraincv[:,10:]
    testfloats = xtestcv[:,:10].astype(float)
    testbinaries = xtestcv[:,10:]

    # Rescale all continuous features to have mean = 0 and stddev = 1.
    trainfloats = preprocessing.scale(trainfloats)
    testfloats = preprocessing.scale(testfloats)
    # Rescale all binaries to be -1 or 1
    trainbinaries = trainbinaries*2 - 1
    testbinaries = testbinaries*2 - 1

    # Reattach binary and float columns
    xtraincv = np.hstack((trainfloats,trainbinaries))
    xtestcv = np.hstack((testfloats,testbinaries))

    trainarr = train_pd_to_nielsen(xtraincv, ytraincv, ytraincv.max())
    testarr = test_pd_to_nielsen(xtestcv, ytestcv)    
    
    neurnet = nielsen_net.Network([(xtraincv.shape)[1], nhidden, nclasses], 
                                  cost=nielsen_net.CrossEntropyCost)
    print neurnet.SGD(trainarr, nepochs, nbatch, eta, lmbda=lmbda,
                      evaluation_data=testarr, monitor_evaluation_cost=True,
                      monitor_evaluation_accuracy=True)

    # ONLY CALL THIS WHEN USING mynetwork.py
    #print neurnet.SGD(trainarr, nepochs, nbatch, eta)
    #print neurnet.evaluate(test_data=testarr)


def nneurons_test(nneurons):

    fulltrain = pd.read_csv('dat/train.csv')
    # Read in data
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'].apply(lambda x: x-1) # Target
    
    # Split data for cross-validation 
    xtraincv, xtestcv, ytraincv, ytestcv = cross_validation.train_test_split(\
        trainx, trainy, test_size=percentCV, random_state=0)
    
    # xtraincv[:,:10] # <---- Non-binary columns
    # xtraincv[:,10:] # <---- Binary columns
    trainfloats = xtraincv[:,:10].astype(float)
    trainbinaries = xtraincv[:,10:]
    testfloats = xtestcv[:,:10].astype(float)
    testbinaries = xtestcv[:,10:]

    # Rescale all continuous features to have mean = 0 and stddev = 1.
    trainfloats = preprocessing.scale(trainfloats)
    testfloats = preprocessing.scale(testfloats)
    # Rescale all binaries to be -1 or 1
    trainbinaries = trainbinaries*2 - 1
    testbinaries = testbinaries*2 - 1

    # Reattach binary and float columns
    xtraincv = np.hstack((trainfloats,trainbinaries))
    xtestcv = np.hstack((testfloats,testbinaries))

    trainarr = train_pd_to_nielsen(xtraincv, ytraincv, ytraincv.max())
    testarr = test_pd_to_nielsen(xtestcv, ytestcv)    

    nsuccess = {}

    for n in nneurons:    
        neurnet = nielsen_net.Network([(xtraincv.shape)[1], n, nclasses])
        neurnet.SGD(trainarr, nepochs, nbatch, eta)

        nsuccess[n] = neurnet.evaluate(test_data=testarr, plotbool=False)

    print nsuccess


if __name__ == '__main__':
    if Debug:
        #test()
        convert_test()
    else:
        main()        
        #nneurons_test([30,40,50,60,70])
    
