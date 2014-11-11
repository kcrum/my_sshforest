import re
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

### These will hopefully make your plots prettier.
from mpltools import style
style.use('ggplot')

fulltrainpath='dat/train.csv'


def try_profile(func):
    """
    This allows you to leave a decorator for "kernprof" in your code,
    regardless of whether or not you end up calling it with kernprof. You
    should now add @try_profile instead of @profile before the function you
    would like to profile.
    """
    try:
        wrapped = profile(func)
    except NameError:
        wrapped = func
    return wrapped


def scorechart(ypred, ytrue):
    """
    Make bar graph showing, for each cover type, number of correct predictions
    next to number of occurrences. This gets called in __main__ when
    plotscores == True (and, currently, when neighs == 30 and
    weights == 'distance'; these conditions are totally arbitrary and are only
    there to limit us to one plot).
    """
    truecounts = np.zeros(7)
    predcounts = np.zeros(7)
    hits = np.zeros(7)

    # X-axis labels
    labels = []
    for i in xrange(1,8): labels.append('t'+str(i))

    # Fill arrays of counts per cover type and correct preds per cover type
    for (yp, yt) in zip(ypred, ytrue):
        # Note that yt is {1,2,...,7}, so increment yt - 1 element of array
        truecounts[yt - 1] += 1
        predcounts[yp - 1] += 1
        if yp == yt:
            hits[yt - 1] += 1

    wid = 0.3
    x = np.arange(len(labels))
    plt.bar(x, truecounts, align='center', width = 2*wid, label='True Counts',
            color='lightblue')
    plt.bar(x - wid/2, predcounts, align='center', width = wid, color='r',
            label='Total Predictions', alpha = 0.5, hatch="\\")
    plt.bar(x + wid/2, hits, align='center', width = wid, alpha = 0.5,
            hatch="/", color='g', label='Correct Predictions')
    plt.xlabel('Forest Cover Types')
    plt.xticks(x, labels)
    plt.legend(loc=4)
    tothits, tottruecounts = hits.sum(), truecounts.sum()
    fracright = "{0:.2f}".format(100.*float(tothits)/tottruecounts)
    print '%s/%s (%s %%) correct.' % (tothits, tottruecounts, fracright)

    plt.show()


def decision_boundaries(var1, var2, clf, features, target, npts=200):
    """
    Plot the decision boundaries in the "var1-var2" plane, where the user
    passes two strings specifying these two variables. Note that this expects
    the classifier "clf" to be untrained. clf is trained in this function
    using var1 and var2 of the "feature/target" set.
    """
    print 'Training classifier...'
    clf.fit(features[[var1, var2]].values, target)
    print 'Training done.'
    xmin, xmax = features[var1].min()-1, features[var1].max()+1
    ymin, ymax = features[var2].min()-1, features[var2].max()+1

    # Make mesh grid for decision contour plot
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, num=npts),
                         np.linspace(ymin, ymax, num=npts))
    print 'Building decision contours...'
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print 'Contours done.'
    z = z.reshape(xx.shape)

    # Plot the decision contours
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    ax1.set_xlabel(var1.replace('_',' ')) # Calls to replace get rid of ugly
    # underscores in the axes titles.
    ax1.set_ylabel(var2.replace('_',' '))
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    #ax1.set_xticks(())
    #ax1.set_yticks(())
    ax1.set_title('Classifier Contours')

    # Plot the training points
    ax2.set_axis_bgcolor('white')
    ax2.scatter(features[var1], features[var2], c=target, cmap=plt.cm.Paired)
    ax2.set_xlabel(var1.replace('_',' '))
    ax2.set_ylabel(var2.replace('_',' '))
    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())
    ax2.set_frame_on(True)
    #ax2.set_xticks(())
    #ax2.set_yticks(())
    ax2.set_title('Data Scatter Plot')

    plt.show()


def combine_binary_columns(df):
    """
    This returns your full dataframe with Wilderness_Area columns combined and
    Soil_Type columns combined.
    """
    # Get combined columns
    wildareas = combine_wilderness_areas(df)
    soiltypes = combine_soil_types(df)
    # Drop individual binary columns
    todrop = [x for x in list(df.columns) if
              re.findall('(Soil_Type|Wilderness_).*', x) != []]
    df = df.drop(todrop, axis=1)
    # Add on combined columns
    df['Wilderness_Area'] = wildareas
    df['Soil_Type'] = soiltypes
    return df


def combine_wilderness_areas(df):
    """
    This takes the Wilderness_Area columns and combines them into one column 
    with entries in {1,2,3,4}.
    """
    wa = df.loc[:,'Wilderness_Area1':'Wilderness_Area4']
    waweight = pd.DataFrame(pd.Series([i+1 for i in xrange(4)],
                            index=list(wa.columns), name='Wilderness_Area'))
    return wa.dot(waweight)


def combine_soil_types(df):
    """
    This takes the Soil_Type columns and combines them into one column with
    entries in {1,2,...,40}.
    """
    st = df.loc[:,'Soil_Type1':'Soil_Type40']
    stweight = pd.DataFrame(pd.Series([i+1 for i in xrange(40)], 
                                      index=list(st.columns), 
                                      name='Soil_Type'))
    return st.dot(stweight)


def elevation_vs_wilderness(dfpath = fulltrainpath):
    """
    Example plot of elevation vs. wilderness area
    """
    df = pd.read_csv(dfpath)

    walist = combine_wilderness_areas(df)
    plt.scatter(walist, df.Elevation)
    plt.ylabel('Elevation (meters)')
    plt.xlabel('Wilderness Type')
    #plt.xlim(0,41)
    plt.show()


def elevation_vs_soil(dfpath = fulltrainpath):
    """
    Example plot of elevation vs. soil type    
    """
    df = pd.read_csv(dfpath)
    
    stlist = combine_soil_types(df)
    plt.scatter(stlist, df.Elevation)
    plt.ylabel('Elevation (meters)')
    plt.xlabel('Soil Type')
    plt.xlim(0,41)
    plt.show()

def rescale_trainx(dftrainx):
    """
    Takes in training data (without 'Cover_Type' column) and rescales 
    continuous variables to have mean = 0 and stddev = 1. The range of binary
    variables is shifted from {0,1} to {-1,1}.
    """
    indices = dftrainx.index
    cols = dftrainx.columns
    trainx = dftrainx.values

    # trainx[:,:10] # <---- Non-binary columns
    # trainx[:,10:] # <---- Binary columns
    trainfloats = trainx[:,:10].astype(float)
    trainbinaries = trainx[:,10:]

    # Rescale all continuous features to have mean = 0 and stddev = 1.
    trainfloats = preprocessing.scale(trainfloats)
    # Rescale all binaries to be -1 or 1
    trainbinaries = trainbinaries*2 - 1
    # Reattach binary and float 
    trainx = np.hstack((trainfloats,trainbinaries))

    return pd.DataFrame(trainx, index=indices, columns=cols)


if __name__ == '__main__':
    from sklearn import neighbors
    #from sklearn import ensemble
    #from sklearn import svm

    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target

    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    #clf = ensemble.RandomForestClassifier()

    #clf = svm.SVC(gamma=0.7)
    #trainx = rescale_trainx(trainx)

    var1 = 'Elevation'
    var2 = 'Horizontal_Distance_To_Hydrology'
    #var1 = 'Slope'
    #var2 = 'Aspect'
    #var1 = 'Horizontal_Distance_To_Fire_Points'
    #var2 = 'Vertical_Distance_To_Hydrology'
    #var1 = 'Hillshade_Noon'    
    #var2 = 'Hillshade_9am'
    decision_boundaries(var1, var2, clf, trainx, trainy)
