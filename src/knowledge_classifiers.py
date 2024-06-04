#!/usr/bin/python
'''
Aldebaro. May, 2024

Run several classical classifiers using sklearn.

From:
# -*- coding: utf-8 -*-
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler and by Aldebaro Klautau
# License: BSD 3 clause
'''

import numpy as np
import csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# routines for feature extraction
from utils_frontend import *


def train_classifier(clf_name, X, y, num_classes):
    """
    Parameters
    ==========
    clf_name: str
        Classifier name to be selected
    X:
    y:
    num_classes: int
        Total number of classes

    """
    names = ["Naive Bayes",
             "Decision Tree", "Random Forest",
             "AdaBoost",
             "Linear SVM", "RBF SVM",
             "Neural Net",
             "QDA", "Nearest Neighbors"]
    # "Gaussian Process",

    classifiers = [
        GaussianNB(),
        DecisionTreeClassifier(max_depth=100),
        RandomForestClassifier(max_depth=80, n_estimators=50, n_jobs=-1),
        AdaBoostClassifier(),
        LinearSVC(),  # linear SVM (maximum margin perceptron)
        SVC(gamma=1, C=1),
        MLPClassifier(alpha=0.1, max_iter=500),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1)]

    # GaussianProcessClassifier(1.0 * RBF(1.0)),

    assert (clf_name in names)

    clf_ind = names.index(clf_name)
    clf = classifiers[clf_ind]
    clf.fit(X, y)

    return clf


def train_model(input_path, max_duration=1000):
    X, y = load_data(input_path)

    # reshape X because it is a 3D array and the classifiers
    # expect a 2D array. In other words, the data is a list
    # of matrices, while these classifiers expect a list of
    # vectors
    X = X.reshape(X.shape[0], -1)

    # Convert labels to categorical
    # y = to_categorical(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # iterate over classifiers
    names = ["Naive Bayes",
             "Decision Tree", "Random Forest",
             "AdaBoost",
             "Linear SVM", "RBF SVM",  # "Gaussian Process",
             "Neural Net",
             "QDA", "Nearest Neighbors"]

    for name in names:
        print("###### Training classifier: ", name)

        clf = train_classifier(name, X_train, y_train, np.max(y))
        clf.fit(X_train, y_train)

        pred_train = clf.predict(X_train)
        print('\nPrediction accuracy for the train dataset:\t {:.2%}'.format(
            metrics.accuracy_score(y_train, pred_train)
        ))

        pred_test = clf.predict(X_test)
        print('\nPrediction accuracy for the test dataset:\t {:.2%}'.format(
            metrics.accuracy_score(y_test, pred_test)
        ))

        print('\n\n')


if __name__ == '__main__':
    max_duration = 100
    # input_path = "C:\github\birds_dnn\knowledge_features"
    input_path = "../knowledge_features"
    train_model(input_path, max_duration=max_duration)
