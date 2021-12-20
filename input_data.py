import pandas as pd
from numpy import ravel
from sklearn.model_selection import train_test_split

data_set = pd.read_csv("data files/magic04.data", header=None)

# extract features from labels
features = data_set.iloc[:, :-1]
labels = data_set.iloc[:, [-1]]

# splitting dataset into 70% for training set and 30% for testing set
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)
train_labels = ravel(train_labels)
