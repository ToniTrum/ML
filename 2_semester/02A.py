import numpy as np

class GradientOptimizer:
    def __init__(self, oracle, x0: np.array) -> None:
        self.oracle = oracle
        self.x0 = x0

    def optimize(self, iterations: int, eps: float, alpha: float) -> np.array:
        x = self.x0.copy()

        for _ in range(iterations):
            grad = self.oracle.get_grad(x)
            if np.linalg.norm(grad) <= eps:
                break

            x = x - alpha * grad

        return x
