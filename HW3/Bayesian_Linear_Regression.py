""" 312581039 吳致廣 Chris Wu 2023/10/22
    ML Homework 03:
    3. Bayesian Linear Regression
        Inputs: ??? , points from Polynomial basis linear data generator
        Outputs: 
"""

import numpy as np
import matplotlib.pyplot as plt
from Rand_datagen import Random_data_generator


class Bayesian_Linear_Regression:
    def __init__(self, b, n, w) -> None:
        self.generator = Random_data_generator() 
        self.gt = w
        self.mean_posterior = np.zeros((n, 1))
        self.variance_posterior = 1/b * np.identity(n)
        self.mean_predictive = 0
        self.variance_predictive = 0
    
    def fit(self, x, y, a) -> None:
        basis = np.expand_dims(self.generator.basis_func(n, x), axis=0)
        S = np.linalg.pinv(self.variance_posterior)
        self.variance_posterior = np.linalg.pinv(a*np.dot(basis.transpose(), basis) + S)      # ^ -1
        self.mean_posterior = np.dot(self.variance_posterior, a*basis.transpose()*y+np.dot(S, self.mean_posterior))
        self.variance_predictive = 1/a + np.dot(np.dot(basis, self.variance_posterior), basis.transpose())
        self.mean_predictive = np.dot(self.mean_posterior.transpose(), basis.transpose())
        
    def poly_datagen(self, n, a, w):
        return self.generator.poly_linear_datagen(n, a, w)
    
    def plot_status(self, x_points, y_points, row, column, title):
        x_line = np.linspace(-2, 2, 500)
        mean_fit, variance_fit = [], []
        mean = self.mean_posterior if len(x_points)>0 else self.gt
        variance = self.variance_posterior if len(x_points)>0 else 0
        for x_point in x_line:
            basis = np.expand_dims(self.generator.basis_func(n, x_point), axis=0)
            mean_fit.append(np.dot(basis, mean).item())
            variance_fit.append((a + np.dot(basis, np.dot(variance, basis.transpose()))).item())

        position = row + column if row > 1 else column
        plt.subplot(2, 2, position), plt.title(title), plt.xlim(-2, 2), plt.ylim(-20, 20)
        plt.scatter(x_points, y_points, c='c')
        plt.plot(x_line, mean_fit, color='b')
        plt.plot(x_line, np.asarray(mean_fit) + np.asarray(variance_fit), color='r')
        plt.plot(x_line, np.asarray(mean_fit) - np.asarray(variance_fit), color='r')
        

if __name__ == '__main__':
    
    b = 1
    n = 3
    a = 3
    w = [1, 2, 3]
    iteration = 1000
    
    model = Bayesian_Linear_Regression(b, n, w)
    x_points, y_points = [], []
    model.plot_status(x_points, y_points, row=1, column=1, title='Ground True')
    
    for idx in range(iteration):
        x, y = model.poly_datagen(n, a, w)
        x_points.append(x)
        y_points.append(y)
        
        model.fit(x, y, a)
        print("Add data point {} :({:.5f}, {:.5f})\n".format(idx+1, x, y))
        print("Posterior mean:\n", model.mean_posterior, "\n")
        print("Posterior variance:\n", model.variance_posterior, "\n")
        print("Predictive distribution ~ N({:.5f}, {:.5f})\n".format(model.mean_predictive.item(), model.variance_predictive.item()))
        
        if idx == 9:
            model.plot_status(x_points, y_points, row=2, column=1, title=f'After {idx+1} incomes')
        elif idx == 49:
            model.plot_status(x_points, y_points, row=2, column=2, title=f'After {idx+1} incomes')
    
    model.plot_status(x_points, y_points, row=1, column=2, title=f'Predict result: {iteration}')
    plt.tight_layout()
    plt.show()