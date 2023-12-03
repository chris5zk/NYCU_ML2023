""" 2023 NYCU ML Homework 05: Gaussian Process - Chris Wu 2023/11/27
    INPUT: 
        - input.data contain 34x2 matrix. Each row corresponds to a 2D data point (X, Y)
        - beta = 5 for fitting the epsilon in noisy observation
    TASK1:
        Apply the Gaussian Process to predict the distribution of f and visualize the result. Please use rational quadratic kernel 
        to compute similarities between different points. Visualize: Show all training data points, draw a line to represent mean 
        of f in range [-60, 60] and mark 95% confidence intervel.
    TASK2:
        Optimize the kernel parameters by minimizing negative marginal log-likelihood, and visualize the result again.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load_data(path):
    X_train, Y_train = [], []
    with open(path, 'r') as f:
        for point in f.readlines():
            x, y = point.strip().split(' ')
            X_train.append(float(x))
            Y_train.append(float(y))
    return X_train, Y_train


def visualization(X_train, Y_train, X_test, means, variances, file):
    # 95% confidence intervel
    standards = np.sqrt(np.diag(variances))
    interval = 1.96 * standards
    # visualization
    plt.figure(), plt.xlim(-60, 60)
    plt.scatter(X_train, Y_train, c='b')
    plt.plot(X_test, means, c='r')
    plt.fill_between(X_test, means + interval, means - interval, color='r', alpha=0.1)
    plt.savefig(f'./results/{file}')
    plt.show()
    

class Gaussian_Process:
    def __init__(self) -> None:
        # noise distribution
        self.beta = 5  
        
        # kernel parameter
        self.alpha = 1
        self.sigma = 1
        self.length_scale = 1
    
    def fit(self, X, Y) -> None:
        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)
        self.C = self.Covariance(self.X)
        
    def Covariance(self, X) -> np.ndarray:
        K = self.RQkernel(X, X)
        return K + np.identity(X.shape[0]) * self.beta**(-1)
    
    def RQkernel(self, X1, X2) -> np.ndarray:
        # Rational quadratic kernel
        return self.sigma**2 * (1 + (X1.reshape(-1, 1) - X2)**2 / (2 * self.alpha * self.length_scale**2))**(-self.alpha)

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        
        # compute kernel between (X_test, X_train)
        k = self.RQkernel(self.X, X)      # (X_train_num, X_test_num)
        C = self.Covariance(X)
  
        # predictive distribution
        means = k.T @ np.linalg.inv(self.C) @ self.Y
        variances = C - k.T @ np.linalg.inv(self.C) @ k
        
        return means, variances

    def optimize(self) -> None:
        # define cost function as negative marginal log-likelihood
        def neg_marginal_logL(theta) -> np.float64:
            theta = theta.ravel()

            # update parameter and compute new covariance
            self.alpha = theta[0]
            self.sigma = theta[1]
            self.length_scale = theta[2]
            C = self.Covariance(self.X)
            
            # cost function
            cost = 0.5 * self.X.shape[0] * np.log(2*np.pi)
            cost += 0.5 * np.log(np.linalg.det(C))
            cost += 0.5 * self.Y.T @ np.linalg.inv(C) @ self.Y
            return cost
        
        # minimize the cost function to get theta
        print('\033[41mOriginal parameter \033[0m\talpha:{:.3f}, sigma:{:.3f}, length:{:.3f}'
                    .format(self.alpha, self.sigma, self.length_scale))
        minimize(neg_marginal_logL, [self.alpha, self.sigma, self.length_scale],
                                    bounds=((1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6)))
        print('\033[46mOptimized parameter\033[0m\talpha:{:.3f}, sigma:{:.3f}, length_scale:{:.3f}'
                    .format(self.alpha, self.sigma, self.length_scale))

        # update Covariance matrix
        self.C = self.Covariance(self.X)
        

if __name__ == '__main__':
    path = './data/input.data'
    
    # load training data and generate testing data
    X_train, Y_train = load_data(path)
    X_test = np.linspace(-60, 60, 500)

    model = Gaussian_Process()
    
    # fit training data and predict
    model.fit(X_train, Y_train)
    means, variances = model.predict(X_test)
    visualization(X_train, Y_train, X_test, means, variances, 'GPresult.png')
    
    # optimize kernel parameter and predict
    model.optimize()
    means, variances = model.predict(X_test)
    visualization(X_train, Y_train, X_test, means, variances, 'GPresult_opt.png')