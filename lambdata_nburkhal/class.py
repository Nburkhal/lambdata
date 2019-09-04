import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ToyData():
    """Load a toy dataset from seaborn by passing the name to the class parameters.

    Different toy datasets can be found here: https://github.com/mwaskom/seaborn-data

    Can load the dataset to a pandas dataframe and do a quick correlation heatmap and 
    kernel density estimation.

    For educational purposes only!
    """
    
    # Load a toy dataset from seaborn
    def __init__(self, name):
        self.name = name
        self.df = sns.load_dataset(name)
        
    # Create correlation heatmap of dataset
    def correlation(self):
        plt.figure(figsize=(10,6))
        sns.heatmap(self.df.corr(), annot=True)
        
    def joint_plot(self, x, y):
        self.x = x
        self.y = y
        sns.jointplot(x=self.x, y=self.y, data=self.df, kind='kde')
        
    