import numpy as np

def logistic_func(theta: np.ndarray, x: np.ndarray) -> float:
    """
    Computes logistic regression value for sample x.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        x: 1d array of float, sample to compute value for
    
    Returns:
        float, logistic regression value
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return sigmoid(np.dot(theta, x))

def logistic_func_all(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes logistic regression value for all samples in matrix X.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        X: 2d array of float, row-major matrix of samples
        
    Returns:
        1d array of float, logistic regression values for all samples in matrix X
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return sigmoid(np.dot(X, theta))

def cross_entropy_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes binary cross entropy loss for logistic regression with parameters `theta` on samples `X` and true labels `y`.
    
    Arguments:
        theta: 1d array of float, regression coefficients
        X: 2d array of float, row-major matrix of samples
        y: 1d array of int, true class labels from set {-1, 1}
        
    Returns:
        float, cross entropy loss value
    """
    n = y.shape[0]
    return np.sum(np.log(1 + np.exp(-y * (X @ theta)))) / n

def grad_cross_entropy_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes gradient of binary cross entropy loss for logistic regression with parameters `theta` on samples `X` and true values `y`.
    
    Arguments:
        theta: 1d array of float, regresion coefficients
        X: 2d array of float, row-major matrix of samples
        y: 1d array of int, true class labels from set {-1, 1}
        
    Returns:
        1d array of float, cross entorpy gradient with respect to `theta`
    """
    n = y.shape[0]
    return np.dot(-X.T * y, 1 / (1 + np.exp(y * (X @ theta)))) / n
