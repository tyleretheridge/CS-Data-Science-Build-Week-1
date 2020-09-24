Ideal pipeline for the algorithm's current implementation

0. Stratify data into test and train sets
1. 2 Arrays: Feature matrix, target vector
    1a. target vector needs to be numerically encoded, whole number > 0
2. Use the fit() method to store the data for later comparison
3. Use the predict() method to return the best class label candidate for the given datapoints
    For each x in X_test:
    Take in an x
    Use distance calc comparing it to X_train points
    Find a cutoff of x=k nearest neighbors based on distance
    Count class labels of these nearest neighbors
    Take majority classification labels of the k nearest neighbors and return as prediction

Blog Post:

https://medium.com/@tylerjetheridge98/a-basic-implementation-of-k-nearest-neighbors-in-python-21e02c54a7f6?sk=62928104cfc825b091d496c2d12eeba4
