# nearestneighbor.py
import numpy as np

# Implement the K-Nearest Neighbors algorithm as a Python Class
# Required Methods: Fit, Predict
# Implement a form of KNN that works based on Euclidean distance

class NearestNeighbor:
    # Init is where user will define parameters for usage
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        

    # Private methods that will be used for algorithm calculations
    def __distance(self, x, y):
        # A persisting distance var will hold the running
        # sum of the squared differences
        distance = 0.00
        # Iterate through value pairs to sum using loop
        for idx in range((len(x)) - 1):
            distance += (x[idx] - y[idx])**2
        # Return the sqrt of sum of squared differences
        return np.sqrt(distance)

    def __funcname(self, parameter_list):
        pass    


    # Class methods to be used by the end user, aka public methods
    def fit(self, parameter_list):
        pass
    
    def predict(self, parameter_list):
        pass

