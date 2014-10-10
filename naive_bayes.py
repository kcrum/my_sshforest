import numpy as np
import pandas as pd
from sklearn import cross_validation
import matplotlib.pyplot as plt
import combine_bin_cols as cbc
from keith_nnclass import scorechart

def var_counts(df, var):
    """
    This returns a pd.Series of counts of "var". Use it as a lookup table:
    >>> ccounts = var_counts(df, 'Cover_Type')
    >>> print ccounts[7]
    2160
    """
    return pd.Series(df[var].values.ravel()).value_counts()

def which_bin(value, edges):
    """
    For a set of bin edges, in which bin would you place the value?
    """
    binnum = 0
    # Note: this implies that if the value falls below lowest edge, it gets put
    # into the lowest bin.
    for edge in edges[1:(len(edges)-1)]:
        if value <= edge:
            return binnum
        binnum += 1
    # This implies that if the value falls above the highest edge, it gets put 
    # into the highest bin.
    return binnum

def dist_lookup(df, disttable):
    """
    Return array of predicted Cover_Types from test df. 
    """
    # Get number of Cover_Types.
    ncovertypes = len(disttable['Soil_Type'][1])
    # Get list of features
    features = df.columns
    # Append ypred with the predicted class of each row in the test set.
    ypred = []
    
    for _, row in df.iterrows():
        # Create array of probabilities: one entry for each cover type
        probs = np.ones(ncovertypes)
        # Now loop through all feature distributions and Cover_Types
        for feat in features:
            # Get value
            value = row[feat]
            # Get edges
            edges = disttable[feat][0] 
            # Get value's bin number
            binnum = which_bin(value, edges)
            if binnum >= 20 and feat != 'Soil_Type':
                print edges
                print value

            # Loop over Cover_Types
            for ct in xrange(ncovertypes):                
                probs[ct] *= disttable[feat][1][ct][binnum]
        
        ypred.append(np.argmax(probs))
    return ypred

def bin_feature(df, feature, edges):
    """
    For a given data frame, feature name, and set of bin edges, return a tuple
    containing the bin contents of each Cover_Type's distribution for the given
    feature in the given binning. Divide each bin's contents by the total 
    number of occurrences of that Cover_Type.
    """
    contents = {}
    # Get number of occurrences of the Cover_Types
    covertotals = pd.Series(df['Cover_Type']).value_counts()
    # Get array of unique Cover_Type values to loop over.
    covertypes = np.unique(df['Cover_Type'])
    for ct in covertypes:
        conts,_ = np.histogram(df[feature][df['Cover_Type'] == ct], bins=edges)
        contents[ct]= conts/float(covertotals[ct])

    return contents


def make_distributions(df):
    """
    This creates the table of distributions for all features, broken up by 
    Cover_Type. The table is a dictionary where each key is a feature name and
    the value is a 2-tuple:
        {feature label: ((bin edges), {cover type: bin contents, ...,
                                       cover type: bin contents}) }
    where each "bin contents" in the tuple is the distribution corresponding to
    a given Cover_Type. "bin edges" are the bin edges shared by each of the 
    "bin contents" in the dictionary sharing the 2-tuple with "bin edges."
    """

    disttable = {}
    # Loop over features. Skip 'Id' and 'Cover_Type' as these are not features.
    for feat in df.columns:
        edges = []
        contentdict = {}
        if feat == 'Cover_Type' or feat == 'Id': 
            continue
        elif feat == 'Wilderness_Area' or feat == 'Soil_Type':
            # Create bin edges
            edges = [0.5]
            for val in xrange(np.unique(df[feat]).max()):
                edges.append(1.5 + val)
        else:
            # Get bin edges for full feature distribution
            _, edges = np.histogram(df[feat], bins=20)

        # Append dict to disttable list
        contentdict = bin_feature(df, feat, edges)
        disttable[feat] = tuple((edges, contentdict))
        
    return disttable

# Main
def main():
    fulltrain = pd.read_csv('dat/train.csv')
    fulltrain = cbc.combine_binary_columns(fulltrain) # Combine binary features
    fulltrain = fulltrain.drop(['Id'], axis=1) # Get rid of useless column 
    # Shift target values down: instead of counting 1 to 7 we now count 0 to 6.
    fulltrain['Cover_Type'] = fulltrain['Cover_Type'].apply(lambda x: x-1) 

    percentCV = 0.3 # Fraction of training data for test set
    train, test = cross_validation.train_test_split(fulltrain, 
                                                    test_size=percentCV, 
                                                    random_state=0)
    cols = fulltrain.columns
    # Put train back into a data frame.
    train = pd.DataFrame({col: train[:,i] for (col,i) in \
                          zip(cols, xrange(len(cols)))})
    # Put test back into a data frame.
    test = pd.DataFrame({col: test[:,i] for (col,i) in zip(cols, 
                                                           xrange(len(cols)))})
    
    testy = test['Cover_Type'] # Target
    testx = test.drop(['Cover_Type'], axis=1) # Features

    disttable = make_distributions(train)

    ncovertypes = len(np.unique(fulltrain.Cover_Type))
    ypred = dist_lookup(testx, disttable)

    scorechart(ypred, testy)

#####
if __name__ == '__main__':
    main()
