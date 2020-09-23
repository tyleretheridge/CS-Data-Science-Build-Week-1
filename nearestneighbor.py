# nearestneighbor.py
import numpy as np

class TE_NearestNeighbor:
    def __init__(self, n_neighbors=5, X_train=None, y_train=None):
        self.k = n_neighbors

    def calc_distance(self, vec_a, vec_b):
        return np.linalg.norm(vec_a - vec_b)
        
    def calc_labels(self, x):
        distance_list = list()
        # For every entry in X_train
        for x_train in self.X_train:
            # Calculate distance between x (parameter) and each entry
            distance_list.append(self.calc_distance(x,x_train)) 
        
        # A list of length self.k of the indices of the closest points to
        # input x. 
        k_index = np.argsort(distance_list)[:self.k]
        k_labels = [self.y_train[idx] for idx in k_index]
        # Return labels of k_neighbors
        return k_labels

    def calc_most_common_label(self, arr):
        # Get the maximum number of class labels
        maximum = len(set(self.y_train))
        # list comprehension to create a buckets array
        buckets = [0 for i in range(maximum + 1)]
        # Count the labels and increment the buckets
        for value in arr:
            buckets[value] += 1
        # Return the index of the largest bucket
        # AKA return the label with the highest occurrence in the list
        most_common_label = buckets.index(max(buckets))
        
        return most_common_label

    def point_predict(self, x):
        # Perform prediction routine on a single point
        # Get labels of k neighbors
        k_labels = self.calc_labels(x)
        # Find most common label
        predicted_label = self.calc_most_common_label(k_labels)
        return predicted_label

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Perform self.point_predict on full array
        # Return results in an np.array
        labels = np.array([self.point_predict(x) for x in X_test])
        return labels