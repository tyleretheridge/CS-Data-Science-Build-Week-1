# nearestneighbor.py
import numpy as np

# Implement the K-Nearest Neighbors algorithm as a Python Class
# Required Methods: Fit, Predict
# Implement a form of KNN that works based on Euclidean distance

class NearestNeighbor:
    """[summary]

    Returns:
        [type]: [description]
    """
    def __init__(self, n_neighbors=5, X_train=None, y_train=None):
        self.k = n_neighbors

    def calc_distance(self, vec_a, vec_b):
        # TODO: Figure out how to deal with class labels in the matrices
        # Avoid performing distance on the class label if numeric
        """Calculates and returns the distance between two vectors 
        of various dimensions.

        Args:
            vec_a ([type]): [description]
            vec_b ([type]): [description]

        Returns:
            float: the distance between the two given vectors
        """
        return np.linalg.norm(vec_a - vec_b)
        
    def predict_label(self, x):
        """Returns a list of distances between the parameter "x",
        and the entries in the training data.

        Args:
            x (1-D array): Test point to be classified
        """
        distance_list = list()
        # For every entry in X_train
        for x_train in self.X_train:
            # Calculate distance between x (parameter) and each entry
            distance_list.append(self.calc_distance(x,x_train)) 
        
        # A list of length self.k of the indices of the closest points to
        # input x. 
        k_index = np.argsort(distance_list)[:self.k]
        
        # Get the maximum number of class labels
        maximum = len(set(self.y_train))
        # list comprehension to create a buckets array
        buckets = [0 for i in range(maximum + 1)]
        # Count the labels and increment the buckets
        for value in k_index:
            buckets[value] += 1
        # Return the index of the largest bucket
        # AKA return the label with the highest occurrence in the list
        most_common_label = buckets.index(max(buckets))
        return most_common_label

    def fit(self, X_train, y_train):
        """Takes in training data and stores it for comparison.

        Args:
            X_train ([type]): Feature matrix
            y_train ([type]): Target vector
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, y_test):
        """Returns a list of predicted class labels

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
        """
        labels_predict = np.array([self.predict_label(x) for x in X_test])
        return labels_predict
