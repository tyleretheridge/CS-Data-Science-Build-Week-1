# nearestneighbor.py
import numpy as np

class TE_NearestNeighbor:
    def __init__(self, n_neighbors=5, X_train=None, y_train=None):
        self.k = n_neighbors

    def calc_distance(self, vec_a, vec_b):
        """Calculates Euclidean distance between two vectors or points

        Args:
            vec_a (array): First vector to be used in distance calculation
            vec_b ([type]): Second vector to be used in distance calculation

        Returns:
            float : Euclidean distance between vec_a and vec_b
        """
        return np.linalg.norm(vec_a - vec_b)
        
    def calc_labels(self, x):
        """Finds the input vector's neighbors and returns them in a list with the labels

        Args:
            x (ndarray): Vector or array representation of the test point

        Returns:
            k_labels: A list of length k that contains the labels of the neighbors
        """
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
        """A counting algorithm that returns the most common class label

        Args:
            arr (list): A list of class labels

        Returns:
            most_common_label: The class label in arr with the highest occurence
        """
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
        """Function that performs the prediction pipeline on a single point

        Args:
            x (ndarray): Vector or array representation of the test point

        Returns:
            predicted_label: The predicted class label for the input vector
        """
        # Perform prediction routine on a single point
        # Get labels of k neighbors
        k_labels = self.calc_labels(x)
        # Find most common label
        predicted_label = self.calc_most_common_label(k_labels)
        return predicted_label

    def fit(self, X_train, y_train):
        """Stores the training data as class attributes
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Performs the prediction pipeline on a full array

        Args:
            X_test (array): A feature matrix

        Returns:
            labels:  A list of labels for the predicted values
        """
        # Perform self.point_predict on full array
        # Return results in an np.array
        labels = np.array([self.point_predict(x) for x in X_test])
        return labels