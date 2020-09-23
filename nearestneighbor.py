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
        self.n_neighbors = n_neighbors

    # Private methods that will be used for algorithm calculations
    def __distance(self, vec_a, vec_b):
        # TODO: Figure out how to deal with class labels in the matrices
        # Avoid performing distance on the class label if numeric
        """Calculates and returns the distance between two vectors of various dimension

        Args:
            vec_a ([type]): [description]
            vec_b ([type]): [description]

        Returns:
            float: the distance between the two given vectors
        """
        return np.linalg.norm(vec_a - vec_b)
        
    def __neighborscalc(self, parameter_list):
        """[summary]

        Args:
            parameter_list ([type]): [description]
        """
        # Make an empty list

        # Use self.__distance() to calculate distances for every relation

        # Append results of distance calc to list


        # Return a list of distances between a record and all of its neighbor
        # relationships
        return



    # Class methods to be used by the end user, aka public methods
    def fit(self, X, y):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, y):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]
        """

