import time
import numpy as np
import pandas as pd
from sklearn import cross_validation, svm, preprocessing
import matplotlib.pyplot as plt
from keith_nnclass import scorechart

# What fraction of fulltrain do you want to be your test set?
percentCV = 0.3

def main():
    starttime = time.time()
    # Read in data
    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features only
    trainy = fulltrain['Cover_Type'] # Target

    print 'Data reading complete.'

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
    
    trainstart = time.time()
    # Create and fit SVM
    clf = svm.SVC(gamma=0.7)
    clf.fit(xtraincv, ytraincv)
    
    traintime = time.time() - trainstart
    print 'Fit took %s seconds.' % "{:.1f}".format(traintime)
    
    # Now predict test set
    ypred = clf.predict(xtestcv)

    endtime = time.time() - starttime
    print 'Finished at %s seconds.' % "{:.1f}".format(endtime)
    
    scorechart(ypred, ytestcv)
    

#####
if __name__ == '__main__':
    main()

