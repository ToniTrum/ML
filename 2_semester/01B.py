import numpy as np

def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes linear regression coefficients that minimize MSE on the given `X` and `y`.
    
    Arguments:
        X: 2d array of float, row-major matrix of samples
        y: 1d array of float, true predictions
    
    Returns:
        np.ndarray, computed coefficients
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y
