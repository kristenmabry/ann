import pandas as pd

def readData(fileName: str, seperator: str):
    return pd.read_csv(fileName, sep=seperator, names=['class', 'm00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30'], index_col=False, skiprows=1, engine='python')