import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_data(v):
    return (v - np.min(v)) / (np.max(v) - np.min(v))

def oversampling(data, targets):
    oversample = SMOTE()
    data, targets = oversample.fit_resample(data, targets)
    return data, targets

def undersampling(data, targets):
    undersample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    X, Y = undersample.fit_resample(data, targets)
    return X, Y

def split_data(data):
    split_point = round(0.70 * len(data))
    return data[0:split_point], data[split_point:]

def apply_pca(X, Xt):
    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    X_test = sc.transform(Xt)
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    print(str(explained_variance))
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X)
    X_test = pca.transform(Xt)
    return X_train, X_test