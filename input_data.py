from builtins import print

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn

name = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
data_set = pd.read_csv("data files/magic04.data", header=None, names=name)

# Encoding categorical Labels
le = LabelEncoder()
data_set["class"] = le.fit_transform(data_set["class"])

# Extract classes and downSample
data_g = data_set.loc[data_set["class"] == 0]
data_h = data_set.loc[data_set["class"] == 1]
data_g_down = sklearn.utils.resample(data_g, replace=False, n_samples=len(data_h), random_state=123)
data_g_down.reset_index(drop=True, inplace=True)
data_h.reset_index(drop=True, inplace=True)

# Extract features and labels
features_g = data_g_down.iloc[:, :-1].values
labels_g = data_g_down.iloc[:, -1].values

features_h = data_h.iloc[:, :-1].values
labels_h = data_h.iloc[:, -1].values

# extract features and labels
# features = data_set.iloc[:, :-1]
# labels = data_set.iloc[:, -1]

# splitting dataset into 70% for training set and 30% for testing set
# TODO : make random_state = NONE to generate random split every time
train_features_g, test_features_g, train_labels_g, test_labels_g = train_test_split(features_g, labels_g,
                                                                                    test_size=0.3,
                                                                                    random_state=42)
train_features_h, test_features_h, train_labels_h, test_labels_h = train_test_split(features_h, labels_h, test_size=0.3,
                                                                                    random_state=42)

train_features = np.concatenate([train_features_g, train_features_h])
train_labels = np.concatenate([train_labels_g, train_labels_h])

test_features = np.concatenate([test_features_g, test_features_h])
test_labels = np.concatenate([test_labels_g, test_labels_h])

# Feature scaling using standardization
# After this step all features will be between -3 and 3
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)
