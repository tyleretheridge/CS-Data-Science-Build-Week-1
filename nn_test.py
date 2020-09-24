# nn_test.py

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Test the pipline
iris = datasets.load_iris() 

X, y =  iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

# My KNN Implementation
knn = TE_NearestNeighbor(n_neighbors=3)
knn.fit(X_train, y_train)
predicted_labels = knn.predict(X_test)
my_accuracy = accuracy_score(y_test, predicted_labels)
print(my_accuracy)

#Sci-kit Learn KNN Implementation
skl_knn = KNeighborsClassifier(n_neighbors=3)
skl_knn.fit(X_train, y_train)
skl_predictions = skl_knn.predict(X_test)
skl_accuracy = accuracy_score(y_test, skl_predictions)
print(skl_accuracy)