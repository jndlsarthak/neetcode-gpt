import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        s = [0] * len(X)
        for i in range(len(X)):
            s[i] = np.sum(X[i] * weights)
        return np.round(s,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        acc = 0
        for i in range(len(model_prediction)):
            acc += (model_prediction[i] - ground_truth[i])**2
        
        acc = acc[0]/len(model_prediction)
        return np.round(acc,5)

