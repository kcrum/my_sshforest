import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble
import matplotlib.pyplot as plt
import combine_bin_cols as cbc
from keith_nnclass import scorechart

# What fraction of fulltrain do you want to be your test set?
percentCV = 0.3

# Main
def main():
    fulltrain = pd.read_csv('dat/train.csv')
    trainx = fulltrain.drop(['Id','Cover_Type'], axis=1) # Features
    trainy = fulltrain['Cover_Type'] # Target

    # Combine 'Soil_Type' columns and 'Wilderness_Type' columns so they are 
    # just 2 columns, rather than 44.
    trainx = cbc.combine_binary_columns(trainx)

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
    main()

