import re
import pandas as pd
import matplotlib.pyplot as plt

fulltrain = pd.read_csv('dat/train.csv')

# This returns your full dataframe with Wilderness_Area columns combined and 
# Soil_Type columns combined.
def combine_binary_columns(df = fulltrain):
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

# This takes the Wilderness_Area columns and combines them into one column with
# entries in {1,2,3,4}.
def combine_wilderness_areas(df = fulltrain):
    wa = df.loc[:,'Wilderness_Area1':'Wilderness_Area4']
    waweight = pd.DataFrame(pd.Series([i+1 for i in xrange(4)],
                            index=list(wa.columns), name='Wilderness_Area'))
    return wa.dot(waweight)

# This takes the Soil_Type columns and combines them into one column with
# entries in {1,2,...,40}.
def combine_soil_types(df = fulltrain):
    st = df.loc[:,'Soil_Type1':'Soil_Type40']
    stweight = pd.DataFrame(pd.Series([i+1 for i in xrange(40)], 
                                      index=list(st.columns), 
                                      name='Soil_Type'))
    return st.dot(stweight)

# Example plot of elevation vs. wilderness area
def elevation_vs_wilderness(df = fulltrain):
    walist = combine_wilderness_areas(df)
    plt.scatter(walist, df.Elevation)
    plt.ylabel('Elevation (meters)')
    plt.xlabel('Wilderness Type')
    #plt.xlim(0,41)
    plt.show()

# Example plot of elevation vs. soil type
def elevation_vs_soil(df = fulltrain):
    stlist = combine_soil_types(df)
    plt.scatter(stlist, df.Elevation)
    plt.ylabel('Elevation (meters)')
    plt.xlabel('Soil Type')
    plt.xlim(0,41)
    plt.show()
