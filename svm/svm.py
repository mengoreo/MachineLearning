#!/opt/local/bin/python
# _*_ coding: utf-8 _*_

'K Nearest Neighbour'

__author__ = 'Ethan Mengoreo'

import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('../knn/breast-cancer-wisconsin.data')
# Replace not a number
df.replace('?', -99999, inplace=True) # Outlier
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
# Basicly a transpose
example_measures = example_measures.reshape(len(example_measures), -1)
predicted = clf.predict(example_measures)
print(predicted)
