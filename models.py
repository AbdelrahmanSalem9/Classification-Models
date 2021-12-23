from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from input_data import train_features, train_labels, test_features, test_labels
import time


def results_format(classifier_name, train_time, prediction_time, accuracy, cm):
    print(classifier_name + ": ")
    print("Confusion Matrix : ")
    print(cm)
    print(f"training time= {round(train_time, 6)} s")
    print(f"prediction_time= {round(prediction_time, 6)} s")
    print(f"accuracy = {accuracy}")
    print("-------------------------------------------------------------------------------------")


def decision_tree():
    clf = DecisionTreeClassifier()
    modeling(clf, "Decision Tree")


def knn(k=5):

    clf = KNeighborsClassifier(n_neighbors=k)
    modeling(clf, "K-Nearest Neighbors")


def naive_bayes():
    clf = GaussianNB()
    modeling(clf, "Naive Bayes")


def ada_boost(estimators=50):
    clf = AdaBoostClassifier(n_estimators=estimators)
    modeling(clf, "AdaBoost")


def random_forests(estimators=50):
    clf = RandomForestClassifier(n_estimators=estimators)
    modeling(clf, "Random Forests")


def modeling(clf, classifier_name):
    t1 = time.time()
    clf.fit(train_features, train_labels)
    t2 = time.time()
    train_time = t2 - t1

    t1 = time.time()
    predictions = clf.predict(test_features)
    t2 = time.time()
    prediction_time = t2 - t1

    cm = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(predictions, test_labels)
    results_format(classifier_name, train_time, prediction_time, accuracy, cm)
