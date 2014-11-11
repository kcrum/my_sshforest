# Attempts at a Kaggle Problem

Kaggle is [hosting a competition](https://www.kaggle.com/c/forest-cover-type-prediction) to predict which type tree---one out of seven possibilities---one would expect to encounter on 565,893 plots of Roosevelt National Forest in Colorado, where each plot measures 30 meters x 30 meters. A training set of 15,121 classified plots was provided. Some of the predictive variables, such as the elevation of the plot, are quantitative. Others variables are qualitative, such as which type of soil out of forty possibilities is present at the plot.

This repository contains some algorithms which attempt to solve this classification problem. All algorthims depend on the NumPy/SciPy/Matplotlib framework; most use scikit-learn. Most algorithms have a main() function that is invoked when the user calls
```
python somealgo.py
```
from the command line (here "somealgo.py" corresponds to one of the classifiers in this repository).

## nielsen_net.py
The neural net in "nielsen_net.py" depends on sample code by Michael Nielsen, shared [here](https://github.com/mnielsen/neural-networks-and-deep-learning) on github. Currently you can't simply fork this repository and call "nielsen_net.py" without setting up Nielsen's code on your machine. 

## sshforest_utilities.py
This contains a few useful plotting 
