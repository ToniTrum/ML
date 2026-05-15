import numpy as np

class GDM:
    '''
    Represents a Gradient Descent with Momentum optimizer

    Fields:
        eta: learning rate
        alpha: exponential decay factor
    '''
    eta: float
    alpha: float
    
    def __init__(self, *, alpha: float = 0.9, eta: float = 0.1) -> None:
        '''Initalizes `eta` and `aplha` fields'''
        self.eta = eta
        self.alpha = alpha

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
        delta_w = np.zeros_like(w)

        for _ in range(max_iter):
            grad = oracle.gradient(w)

            if np.linalg.norm(grad) <= eps:
                break

            delta_w = self.alpha * delta_w - self.eta * grad
            w += delta_w

        return w
