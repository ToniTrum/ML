import numpy as np

def linear_func(theta: np.ndarray, x: np.ndarray) -> float:
    """
    Computes linear regression value for sample x.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        x: 1d array of float, sample to compute value for
    
    Returns:
        float, linear regression value
    """
    return np.dot(theta, x)

def linear_func_all(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes linear regression value for all samples in matrix X.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        X: 2d array of float, row-major matrix of samples
        
    Returns:
        1d array of float, linear regression values for all samples in matrix X
    """
    return np.dot(X, theta)

def mean_squared_error(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes MSE loss for linear regression with parameters `theta` on samples `X` and true values `y`.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        X: 2d array of float, row-major matrix of samples
        y: 1d array of float, true values
        
    Returns:
        float, mse loss value
    """
    y_pred = linear_func_all(theta, X)
    n = y.shape[0]
    return np.sum((y_pred - y) ** 2) / n

def grad_mean_squared_error(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes gradient of mse loss for logistic regression with parameters `theta` on samples `X` and true values `y`.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        X: 2d array of float, row-major matrix of samples
        y: 1d array of int, true values
        
    Returns:
        1d array of float, mse gradient with respect to `theta`
    """
    y_pred = linear_func_all(theta, X)
    n = y.shape[0]
    return 2 / n * np.dot(X.T, (y_pred - y))
