from sklearn.model_selection import cross_val_score
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


def get_best_k():
    k_range = range(1, 20)
    k_scores = []

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, train_features, train_labels, cv=7, scoring='accuracy')
        k_scores.append(scores.mean())

    return (k_scores.index(max(k_scores))) + 1


def get_best_n_estimators_ab():
    # default = 50
    n_range = range(40, 70)
    n_scores = []

    for n in n_range:
        clf = AdaBoostClassifier(n_estimators=n)
        scores = cross_val_score(clf, train_features, train_labels, cv=7, scoring='accuracy')
        n_scores.append(scores.mean())

    return (n_scores.index(max(n_scores))) + 30


def get_best_n_estimators_rf():
    # default = 100
    n_range = range(80, 120)
    n_scores = []

    for n in n_range:
        clf = RandomForestClassifier(n_estimators=n)
        scores = cross_val_score(clf, train_features, train_labels, cv=7, scoring='accuracy')
        n_scores.append(scores.mean())

    return (n_scores.index(max(n_scores))) + 80


def knn():
    k = get_best_k()
    clf = KNeighborsClassifier(n_neighbors=k)
    modeling(clf, "K-Nearest Neighbors")


def naive_bayes():
    clf = GaussianNB()
    modeling(clf, "Naive Bayes")


def ada_boost():
    estimators = get_best_n_estimators_ab()
    clf = AdaBoostClassifier(n_estimators=estimators)
    modeling(clf, "AdaBoost")


def random_forests():
    estimators = get_best_n_estimators_rf()
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
