from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from input_data import train_features, train_labels, test_features, test_labels
import matplotlib.pyplot as plt
import time


def results_format(classifier_name, train_time, prediction_time, accuracy, precision, sensitivity, specificity,
                   f1_score, cm, tuning_parameter=None):
    print(classifier_name + ": ")
    print("Confusion Matrix : ")
    print(cm)
    print(f"Training time= {round(train_time, 6)} s")
    print(f"Prediction_time= {round(prediction_time, 6)} s")
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Sensitivity = {sensitivity}")
    print(f"Specificity = {specificity}")
    print(f"F score = {f1_score}")
    if tuning_parameter != None:
        print(f"Tuning parameter = {tuning_parameter}")
    print("-------------------------------------------------------------------------------------")


def decision_tree():
    clf = DecisionTreeClassifier()
    modeling(clf, "Decision Tree")


def get_best_k():
    k_range = range(1, 20)
    k_scores = []
    skf = StratifiedKFold(n_splits=7)
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, train_features, train_labels, cv=skf, scoring='accuracy')
        k_scores.append(scores.mean())
    plot_tuning_parameter(k_range, k_scores, "K for K-Nearest Neighbor", "K-values", "Mean-Accuracy")
    return (k_scores.index(max(k_scores))) + k_range[0]


def get_best_n_estimators_ab():
    # default = 50
    n_range = range(50, 70)
    n_scores = []
    skf = StratifiedKFold(n_splits=7)
    for n in n_range:
        clf = AdaBoostClassifier(n_estimators=n)
        scores = cross_val_score(clf, train_features, train_labels, cv=skf, scoring='accuracy')
        n_scores.append(scores.mean())
    plot_tuning_parameter(n_range, n_scores, "N_Estimators for AdaBoost", "n-values", "Mean-Accuracy")
    return (n_scores.index(max(n_scores))) + n_range[0]


def get_best_n_estimators_rf():
    # default = 100
    n_range = range(70, 100)
    n_scores = []
    skf = StratifiedKFold(n_splits=7)
    for n in n_range:
        clf = RandomForestClassifier(n_estimators=n)
        scores = cross_val_score(clf, train_features, train_labels, cv=skf, scoring='accuracy')
        n_scores.append(scores.mean())
    plot_tuning_parameter(n_range, n_scores, "N_Estimators for Random Forest", "n-values", "Mean-Accuracy")
    return (n_scores.index(max(n_scores))) + n_range[0]


def knn():
    k = get_best_k()
    clf = KNeighborsClassifier(n_neighbors=k)
    modeling(clf, "K-Nearest Neighbors", k)


def naive_bayes():
    clf = GaussianNB()
    modeling(clf, "Naive Bayes")


def ada_boost():
    estimators = get_best_n_estimators_ab()
    clf = AdaBoostClassifier(n_estimators=estimators)
    modeling(clf, "AdaBoost", estimators)


def random_forests():
    estimators = get_best_n_estimators_rf()
    clf = RandomForestClassifier(n_estimators=estimators)
    modeling(clf, "Random Forests", estimators)


def modeling(clf, classifier_name, tuning_parameter=None):
    t1 = time.time()
    clf.fit(train_features, train_labels)
    t2 = time.time()
    train_time = t2 - t1

    t1 = time.time()
    predictions = clf.predict(test_features)
    t2 = time.time()
    prediction_time = t2 - t1

    """
    CM = | TN FP |
         | FN TP |
    """
    cm = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(predictions, test_labels)

    # precision = (TP) / (TP + FP)
    precision = (cm[1][1]) / (cm[1][1] + cm[0][1])

    # sensitivity = (TP) / (TP + FN)
    sensitivity = (cm[1][1]) / (cm[1][1] + cm[1][0])

    # specificity = (TN) / (TN + FP)
    specificity = (cm[0][0]) / (cm[0][0] + cm[0][1])

    # f1_score = 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    results_format(classifier_name, train_time, prediction_time, accuracy, precision, sensitivity, specificity,
                   f1_score,
                   cm, tuning_parameter)


def plot_tuning_parameter(x, y, title, x_label, y_label):
    plt.title(title, fontsize='16')  # title
    plt.plot(x, y)  # plot the points
    plt.xlabel(x_label, fontsize='13')  # adds a label in the x axis
    plt.ylabel(y_label, fontsize='13')  # adds a label in the y axis
    plt.grid()  # shows a grid under the plot
    plt.show()
