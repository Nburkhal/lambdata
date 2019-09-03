import numpy as np
import pandas as import pd

# Get descriptive statistics for numeric and non-numeric columns of a dataframe

def descriptive(df):

    return df.describe, df.describe(exclude='number')