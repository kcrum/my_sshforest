import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import cross_validation
import matplotlib.pyplot as plt
import sshforest_utilities as util

plotscores = False
plotrates = True 

# Make graph of success rates. Called when plotrates == True.
def rateplot(uniftest, disttest, uniftrain, disttrain, narr):
    plt.plot(narr, uniftest, 'bs', narr, disttest, 'g^', 
             narr, uniftrain, 'bs', narr, disttrain, 'g^')
    plt.show()


def main():
    #Nneighbors = [2]
    Nneighbors = [1,2,3,4,5,7,10,15,25]
    uniftest = np.zeros( len(Nneighbors) )
    disttest = np.zeros( len(Nneighbors) )    
    uniftrain = np.zeros( len(Nneighbors) )
    disttrain = np.zeros( len(Nneighbors) )

    percentCV = 0.3 # Fraction of training data for cross-validation

    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target

    # Combine 'Soil_Type' columns and 'Wilderness_Type' columns so they are 
    # just 2 columns, rather than 44. 
    trainx = util.combine_binary_columns(trainx)

    xtraincv, xtestcv, ytraincv, ytestcv = cross_validation.train_test_split(\
        trainx, trainy, test_size=percentCV, random_state=0)

    cntr = 0
    for neighs in Nneighbors:
        print '-'*50

        for weights in ['uniform', 'distance']:
            print '-'*50
        
            clf = neighbors.KNeighborsClassifier(neighs, weights=weights)
            print 'Training with %s weighting for %s neighbors.' % (weights, 
                                                                    neighs)
            # Fit classifier with training set
            clf.fit(xtraincv, ytraincv)
            # Predict on test set
            ytestpred = clf.predict(xtestcv)
            # Score test set
            testscore = clf.score(xtestcv, ytestcv)
            # Score train set
            trainscore = clf.score(xtraincv, ytraincv)
            print 'Fraction test score is %s/%s = %s' % \
                (testscore*len(ytestcv), len(ytestcv), testscore)

            print 'Fraction train score is %s/%s = %s' % \
                (trainscore*len(ytraincv), len(ytraincv), trainscore)
            # Fill scores array
            if weights == 'uniform': 
                uniftest[cntr] = testscore
                uniftrain[cntr] = trainscore
            else: 
                disttest[cntr] = testscore
                disttrain[cntr] = trainscore

            if plotscores and neighs == 30 and weights == 'distance':  
                util.scorechart(ytestpred, ytestcv)
        cntr += 1

    print '-'*50
    
    if plotrates:
        rateplot(uniftest, disttest, uniftrain, disttrain, Nneighbors)


def submit(neighs = 2, distweight = True, binarycombo = True):
    # Which weighting?
    wt = ''
    if distweight:
        wt = 'distance'
    else:
        wt = 'uniform'

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

    clf = neighbors.KNeighborsClassifier(neighs, weights=wt)
    print 'Training with %s weighting for %s neighbors.' % (wt, neighs)
    # Fit classifier with training set
    clf.fit(trainx, trainy)
    # Predict on test set
    ypred = clf.predict(testx)

    # Write prediction to file with Kaggle-approved format
    output = pd.Series(ypred, index=testid, name='Cover_Type')
    outname = 'output/%sWeight%sNeighborsCBC.csv' % (wt, neighs)
    output.to_csv(outname, header=True)

if __name__ == '__main__':
    main()
    #submit()
