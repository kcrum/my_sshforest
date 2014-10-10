import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import cross_validation
import matplotlib.pyplot as plt
import combine_bin_cols as cbc

plotscores = False
plotrates = True

# Make bar graph showing, for each cover type, number of correct predictions 
# next to number of occurrences. This gets called in __main__ when 
# plotscores == True (and, currently, when neighs == 30 and 
# weights == 'distance'; these conditions are totally arbitrary and are only
#  there to limit us to one plot).
def scorechart(ypred, ytrue):    
    counts = np.zeros(7)
    hits = np.zeros(7)

    # X-axis labels
    labels = []
    for i in xrange(1,8): labels.append('t'+str(i))       

    # Fill arrays of counts per cover type and correct preds per cover type
    for (yp, yt) in zip(ypred, ytrue):
        # Note that yt is {1,2,...,7}, so increment yt - 1 element of array
        counts[yt - 1] += 1        
        if yp == yt: 
            hits[yt - 1] += 1

    wid = 0.4
    x = np.arange(len(labels))
    plt.bar(x - wid/2, counts, align='center', width = wid, hatch="\\", 
            label='Total Predictions')
    plt.bar(x + wid/2, hits, align='center', width = wid, alpha = 0.5, 
            hatch="/", color='g', label='Correct Predictions')
    plt.xticks(x, labels)
    plt.legend()
    tothits, totcounts = hits.sum(), counts.sum()
    print '%s/%s (%s %%) correct.' % (tothits, totcounts, 
                                      float(tothits)/totcounts)
    plt.show()          
 

# Make graph of success rates. Called when plotrates == True.
def rateplot(uniftest, disttest, uniftrain, disttrain, narr):
    plt.plot(narr, uniftest, 'bs', narr, disttest, 'g^', 
             narr, uniftrain, 'bs', narr, disttrain, 'g^')
    plt.show()


if __name__ == '__main__':
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
    trainx = cbc.combine_binary_columns(trainx)

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
                scorechart(ytestpred, ytestcv)
        cntr += 1

    print '-'*50
    
    if plotrates:
        rateplot(uniftest, disttest, uniftrain, disttrain, Nneighbors)
