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
    # Init is where user will define parameters for usage
    def __init__(self, n_neighbors=5, X_train=None, y_train=None):
        self.k = n_neighbors

    # Methods that will be used for algorithm calculations
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
        
    def calc_neighbors(self, x):
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


    def calc_most_common_label(self, arr):
        """A counting algorithm that will return the most common 
        class label in the passed array.

        Args:
            arr ([type]): An array of k_indices
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



# Compare passed in parameter to every value in X_train
    # Take in an X
    # Use distance calc comparing it to X_train points
    # Find a cutoff of nearest neighbors
    # Take majority classification labels of the k nearest neighbors
    # use np.argsort to return an array of sorted indices
    # argsort returns a copy of the indices as if the original matrix were sorted
    # useful for maintaining index integrity to label vector

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Class methods to be used by the end user, aka public methods
    def fit(self, X, y):
        """Takes in training data and stores it for comparison.

        Args:
            X ([type]): Feature matrix
            y ([type]): Target vector
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, y):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]
        """

