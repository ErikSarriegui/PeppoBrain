import numpy as np

def load(path = "dataset/train.csv"):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    
    mask = np.isnan(data).any(axis=1)
    data = data[~mask]
    
    Y = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0
    
    return X, Y