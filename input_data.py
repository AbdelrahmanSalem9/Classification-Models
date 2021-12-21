import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_set = pd.read_csv("data files/magic04.data", header=None)

# extract features and labels
features = data_set.iloc[:, :-1].values
labels = data_set.iloc[:, -1].values

# Encoding categorical Labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# splitting dataset into 70% for training set and 30% for testing set
# TODO : make random_state = NONE to genetrae random split every time
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

# Feature scaling using standardization
# After this step all features will be between -3 and 3
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)
