""" 312581039 吳致廣 Chris Wu 2023/10/22
    ML Homework 03:
    1. Random data generator
        a. Univariate gaussian data generator
            Inputs: Expectation value or mean - m, Variance - s
            Outputs: A data point from N(m, s)
        b. Polynomial basis linear model data generator (y = W*fi + e)
            Inputs: n(basis number), a(variance), w(weight of line)
            Outputs: point (x, y), -1.0 < x < 1.0 and x is uniformly distributed
"""

import numpy as np

class Random_data_generator:
    def __init__(self) -> None:
        pass

    def uni_gaussian_datagen(self, mean, variance) -> float:
        # Marsaglia polar method
        while(True):
            u = np.random.uniform(-1, 1)
            v = np.random.uniform(-1, 1)
            s = u**2 + v**2
            if(0 < s < 1):
                break
        z = u * np.sqrt(-2 * np.log(s)/s)
        return mean + np.sqrt(variance) * z
    
    def poly_linear_datagen(self, n, a, weight) -> (float, float):
        x = np.random.uniform(-1, 1)
        basis = self.basis_func(n, x)
        error = self.uni_gaussian_datagen(0, a)
        y = np.sum(weight * basis) + error
        return x, y
    
    def basis_func(self, n, x) -> float:
        basis = np.zeros(n)
        for exp in range(n):
            basis[exp] = x**exp     # basis = [x^0, x^1, x^2...]
        return basis
    
if __name__ == '__main__':
    pass