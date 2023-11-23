""" 312581039 吳致廣 2023/09/21.
    ML Homework-01: Regularized linear model regression and visualization
    Inputs: a bunch of points in testfile.txt, n, lambda
    Outputs: Fitting Line function, Total error, Visualization plot
    
    Required functions:
    a. For closed-form LSE approach:
        1. Use LU decomposition to find the inverse of A^T A + lambda I, Gauss-Jordan elimination will also be accepted.(A is the design matrix).
        2. Print out the equation of the best fitting line and the error.
    b. For steepest descent method:
        1. Use steepest descent with LSE and L1 norm to find the best fitting line.
        2. Print out the equation of the best fitting line and the error.
        3. reference (hint : Consider using a smaller learning rate.)
    c. For Newton's method:
        1. Please use the method mentioned in the lesson.
        2. Print out the equation of the best fitting line and the error, and compare to LSE.
    d. For visualization:
        1. Please visualize the data points which are the input of program, and the best fitting curve.
        2. It's free to use any existing package.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=20) # Set the float precision


class Linear_regression_model:
    def __init__(self, x_points, y_points, n, lambda_):
        self.x = np.array(x_points)
        self.y = np.array(y_points)
        self.n = n
        self.lambda_ = lambda_
    
    def rLSE(self):
        A = self.basis_func()
        ATA = np.dot(A.transpose(), A)
        invertible = ATA + self.lambda_ * np.identity(self.n)
        GJ = np.append(invertible, np.identity(self.n), axis=1)
        inverse = self.Gauss_Jordan_Elimination(GJ)[:, self.n:]
        x = np.dot(np.dot(inverse, A.transpose()), np.array(self.y).transpose())
        loss = np.dot((np.dot(A, x)-self.y), (np.dot(A, x)-self.y))
        return np.flip(x), loss
    
    def SGD(self, epochs=50, lr=0.00001):
        A = self.basis_func()
        x = np.ones((self.n))
        for epoch in range(epochs):
            # 2 AT A x - 2 AT b + derivative of L1 norm
            dL = 2 * np.dot(np.dot(A.transpose(), A), x) - 2 * np.dot(A.transpose(), self.y) + self.lambda_ * x / np.abs(x)
            x -= lr * dL
        loss = np.dot((np.dot(A, x)-self.y), (np.dot(A, x)-self.y))
        return np.flip(x), loss
         
    def Newton_method(self):
        A = self.basis_func()
        ATA = np.dot(A.transpose(), A)
        GJ = np.append(ATA, np.identity(self.n), axis=1)
        inverse = self.Gauss_Jordan_Elimination(GJ)[:, self.n:]
        x = np.dot(np.dot(inverse, A.transpose()), np.array(self.y).transpose())
        loss = np.dot((np.dot(A, x)-self.y), (np.dot(A, x)-self.y))
        return np.flip(x), loss
    
    def basis_func(self):
        # polynomial basis function
        basis = np.zeros((len(self.x), self.n))
        for item in range(self.n):
            basis[:, self.n-1-item] = self.x ** item
        return basis
    
    def Gauss_Jordan_Elimination(self, A):
        k, l = 0, 0
        K, L = A.shape
        
        while True:
            if l > L or k >= K:
                break
            
            # finding the pivot position(column)
            if A[k, l] == 0:
                swap = False
                for behind in range(k, K):
                    if A[behind, l] != 0:
                        A[[k, behind]] = A[[behind, k]]
                        swap = True
                        break
                if swap == False:
                    l += 1
                    print(k, l)
                    continue
            
            # row operation
            A[k] /= A[k, l]
            for non_pivot_row in range(K):
                if non_pivot_row == k:
                    continue
                elif A[non_pivot_row][l] == 0:
                    continue
                else:
                    A[non_pivot_row] -= A[k] * A[non_pivot_row][l]
            k += 1
            l += 1
            
        return A
            
  
def plot_fitting_curve(x_points, y_points, coefficient_lse, coefficient_newton, coefficient_sgd):
    y_lse, y_nwt, y_sgd = 0, 0, 0
    x = np.linspace(-6, 6, 10)
    
    for item in range(n):
        y_lse += (coefficient_lse[item] * (x**item))
        y_nwt += (coefficient_newton[item] * (x**item))
        y_sgd += (coefficient_sgd[item] * (x**item))

    plt.subplot(3, 1, 1), plt.title("rLSE")
    plt.scatter(x_points, y_points, s=20, c='red', edgecolors='black')
    plt.plot(x, y_lse)
    plt.xlim(-6, 6), plt.grid(axis= 'y')
    
    plt.subplot(3, 1, 2), plt.title("Newton's method")
    plt.scatter(x_points, y_points, s=20, c='red', edgecolors='black')
    plt.plot(x, y_nwt)
    plt.xlim(-6, 6), plt.grid(axis= 'y')
    
    plt.subplot(3, 1, 3), plt.title("SGD")
    plt.scatter(x_points, y_points, s=20, c='red', edgecolors='black')
    plt.plot(x, y_sgd)
    plt.xlim(-6, 6), plt.grid(axis= 'y')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    x_points, y_points = [], []
    with open('testfile.txt', 'r') as f:
        for point in f.readlines():
            x, y = point.strip().split(',')
            x_points.append(float(x))
            y_points.append(float(y))
    
    while True:
        n = int(input('Please enter the bases n: '))
        lambda_ = int(input('Please enter the lambda(for LSE and SGD): '))
        print('---------------------------------------------------')
        
        model = Linear_regression_model(x_points, y_points, n, lambda_)
        
        coefficient_lse, error_lse = model.rLSE()
        coefficient_newton, error_newton = model.Newton_method()
        coefficient_sgd, error_sgd = model.SGD()

        function_lse, function_newton, function_sgd = '', '', ''
        for item in reversed(range(n)):
            if item == 0:
                function_lse += f'{coefficient_lse[item]}'
                function_newton += f'{coefficient_newton[item]}'
                function_sgd += f'{coefficient_sgd[item]}'
            else:    
                function_lse += f'{coefficient_lse[item]}X^{item} + '
                function_newton += f'{coefficient_newton[item]}X^{item} + '
                function_sgd += f'{coefficient_sgd[item]}X^{item} + '

        print(f'LSE:\nFitting Line: {function_lse}\nTotal error: {error_lse}\n')
        print(f"Newton's Method:\nFitting Line: {function_newton}\nTotal error: {error_newton}\n")
        print(f"SGD:\nFitting Line: {function_sgd}\nTotal error: {error_sgd}\n")
        plot_fitting_curve(x_points, y_points, coefficient_lse, coefficient_newton, coefficient_sgd)
        
        answer = input('Continue? [Y/N]')
        if answer.lower() != 'y':
            break