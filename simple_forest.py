import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble
import matplotlib.pyplot as plt
from keith_nnclass import scorechart

# What fraction of fulltrain do you want to be your test set?
percentCV = 0.3

# This trains on fulltrain and produces a CSV prediction file for the test set.
def submit():
    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target
    
    fulltest = pd.read_csv('dat/test.csv')
    testid = fulltest['Id'] # You'll need to stitch these onto the predictions
    testx = fulltest.drop(['Id'],axis=1) # Features

    # Create and fit SVM
    clf = ensemble.RandomForestClassifier()
    clf.fit(trainx, trainy)
    
    # Now predict test set
    ypred = clf.predict(testx)
    # Write prediction to file with Kaggle-approved format
    output = pd.Series(ypred, index=testid, name='Cover_Type')
    output.to_csv('output/simple_forest_output.csv', header=True)
    
# Main
def main():
    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target

    xtraincv, xtestcv, ytraincv, ytestcv = cross_validation.train_test_split(\
        trainx, trainy, test_size=percentCV, random_state=0)

    # Create and fit SVM
    clf = ensemble.RandomForestClassifier()
    clf.fit(xtraincv, ytraincv)
    
    # Now predict test set
    ypred = clf.predict(xtestcv)

    scorechart(ypred, ytestcv)

#####
if __name__ == '__main__':
    #main()
    submit()
