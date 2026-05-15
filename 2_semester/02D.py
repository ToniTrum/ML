import numpy as np

class AdaGrad:
    '''Represents an AdaGrad optimizer

    Fields:
        eta: learning rate
        epsilon: smoothing term
    '''
    eta: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, epsilon: float = 1e-8) -> None:
        '''Initalizes `eta` and `epsilon` fields'''
        self.eta = eta
        self.epsilon = epsilon 

    def optimize(
        self, oracle, x0: np.ndarray, *,
        max_iter: int = 100, eps: float = 1e-5
    ) -> np.ndarray:
        '''
        Optimizes a function specified as `oracle` starting from point `x0`.
        The optimizations stops when `max_iter` iterations were completed or 
        the L2-norm of the gradient at current point is less than `eps`

        Args:
            oracle: function to optimize
            x0: point to start from
            max_iter: maximal number of iterations
            eps: threshold for L2-norm of gradient

        Returns:
            A point at which the optimization stopped
        '''
        w = x0.astype(float, copy=True)
        G = np.zeros_like(w)

        for _ in range(max_iter):
            grad = oracle.gradient(w)

            if np.linalg.norm(grad) <= eps:
                break

            G += grad ** 2
            w -= (self.eta / np.sqrt(G + self.epsilon)) * grad

        return w
