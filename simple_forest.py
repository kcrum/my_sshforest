import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing
import matplotlib.pyplot as plt
import sshforest_utilities as util

# What fraction of fulltrain do you want to be your test set?
percentCV = 0.3
binarycombo = True

# This trains on fulltrain and produces a CSV prediction file for the test set.
def submit():
    # Read in data
    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target
    
    fulltest = pd.read_csv('dat/test.csv')
    testid = fulltest['Id'] # You'll need to stitch these onto the predictions
    testx = fulltest.drop(['Id'],axis=1) # Features
    
    print 'Data reading complete.'

    # Combine 'Soil_Type' columns and 'Wilderness_Type' columns so they are
    # just 2 columns, rather than 44.                                          
    if binarycombo:
        trainx = util.combine_binary_columns(trainx)
        testx = util.combine_binary_columns(testx)        

    # Create and fit SVM
    clf = ensemble.RandomForestClassifier()
    clf.fit(trainx, trainy)

    print 'Fit complete.'
    
    # Now predict test set
    ypred = clf.predict(testx)
    # Write prediction to file with Kaggle-approved format
    output = pd.Series(ypred, index=testid, name='Cover_Type')
    #output.to_csv('output/simple_forest_output.csv', header=True)
    output.to_csv('output/simple_forest_cbc_output.csv', header=True)
    
# Main
def main():
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
    testfloats = xtestcv[:,:10].astype(float)

    # Rescale all features to have mean = 0 and stddev = 1. Should you separate
    # out binary columns so they don't get rescaled?
    trainfloats = preprocessing.scale(trainfloats)
    testfloats = preprocessing.scale(testfloats)    

    # Reattach binary columns to scaled float columns
    xtraincv = np.hstack((trainfloats,xtraincv[:,10:]))
    xtestcv = np.hstack((testfloats,xtestcv[:,10:]))

    # Create and fit random forest
    clf = ensemble.RandomForestClassifier()
    clf.fit(xtraincv, ytraincv)
    
    print 'Fit complete.'
    
    # Now predict test set
    ypred = clf.predict(xtestcv)
    util.scorechart(ypred, ytestcv)

#####
if __name__ == '__main__':
    main()
    #submit()
