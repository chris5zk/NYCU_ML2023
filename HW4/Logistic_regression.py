""" 312581039 吳致廣 Chris Wu 2023/11/05
    ML Homework 04: Logistic Regression
        - Inputs:
            1. N: number of data points
            2. (mx1, vx1, my1, vy1), (mx2, vx2, my2, vy2) (m: means, v: variances)
        - Outputs:
            1. Confusion Matrix, sensitivity, specificity
            2. Visualization: GT, predict results of Gradient descent & Newton's method
        - Function:
            1. Generate n data points: D1 = {(x1, y1), (x2, y2), ... } from N(mx1, vx1) and N(my1, vy1)
            2. Generate n data points: D2 = {(x1, y1), (x2, y2), ... } from N(mx2, vx2) and N(my2, vy2)
            3. Implement both Newton's method and Gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from Rand_datagen import Random_data_generator


class Logistic_regression:
    def __init__(self, D1, D2, N) -> None:
        self.A = self.design_matrix(D1, D2, N)
        self.b = self.label(N)
        self.N = N
        self.eps = 0.01
    
    def Gradient_descent(self, w, lr):
        i = 0
        while True:
            i += 1
            dJ = np.dot(self.A.T, self.sigmoid(np.dot(self.A, w)) - self.b)
            w = w - lr * dJ
            if np.sqrt(np.sum(dJ**2)) < self.eps or i == 15000:
                print('Iteration: {}'.format(i))
                break
        
        return w

    def Newtons_method(self, w, lr):
        i = 0
        while True:
            i += 1
            I = np.identity(self.N*2)
            t = np.dot(self.A, w)
            D = np.multiply(I, np.exp(t) / ((1 + np.exp(-t))**2))
            Hessian = np.dot(self.A.T, np.dot(D, self.A))
            dJ = np.dot(self.A.T, self.sigmoid(np.dot(self.A, w)) - self.b)
            
            if np.linalg.det(Hessian) == 0:
                print("Hessian isn't invertible. Use Gradient descent for this iter.")
                w = w - lr * dJ
            else:
                w = w - np.dot(np.linalg.inv(Hessian), dJ)
            if np.sqrt(np.sum(dJ**2)) < self.eps or i == 15000:
                print('Iteration: {}'.format(i))
                break
        return w
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def design_matrix(self, D1, D2, N):
        A = np.ones((2 * N, 3))
        A[:, 1:] = (D1 + D2)
        return A
    
    def label(self, N):
        b = np.zeros((2 * N, 1))
        b[N:] = 1
        return b

    def predict(self, w):
        x = np.dot(self.A, w)
        y = self.sigmoid(x)
        return np.where(y > 0.5, 1, 0)

    def confusion_matrix(self, pred):
        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(2*self.N):
            TP = TP + 1 if self.b[i]==0 and self.b[i]==pred[i] else TP  # 1
            FN = FN + 1 if self.b[i]==0 and self.b[i]!=pred[i] else FN 
            FP = FP + 1 if self.b[i]==1 and self.b[i]!=pred[i] else FP 
            TN = TN + 1 if self.b[i]==1 and self.b[i]==pred[i] else TN  # 2
        sensitivity = TP / (TP + FP)
        specificity = TN / (TN + FN)
        return [TP, FN, FP, TN], sensitivity, specificity

    def info(self, w, CM, sen, spe):
        print('w:')
        for w_ in w:
            print(w_)
        print('Confusion Matrix:')
        print("\t\t\tPredict cluster 1\tPredict cluster 2")
        print("In cluster 1\t\t{}\t\t\t{}".format(CM[0], CM[1]))
        print("In cluster 2\t\t{}\t\t\t{}\n".format(CM[2], CM[3]))
        print("Sensitivity (Successfully predict cluster 1):", sen)
        print("Specificity (Successfully predict cluster 2):", spe)
        
    def visualization(self, D1, D2, pred_gd, pred_nt):
        x1, y1 = zip(*D1)
        x2, y2 = zip(*D2)
        x = x1 + x2
        y = y1 + y2
        
        plt.subplot(1, 3, 1)
        plt.title('Ground True')
        plt.scatter(x1, y1, c='r')
        plt.scatter(x2, y2, c='b')
        
        plt.subplot(1, 3, 2)
        plt.title('Gradient descent')
        for i in range(self.N*2):
            if pred_gd[i] == 0:
                plt.scatter(x[i], y[i], c='r')
            else:
                plt.scatter(x[i], y[i], c='b')
        
        plt.subplot(1, 3, 3)
        plt.title("Newton's method")
        for i in range(self.N*2):
            if pred_nt[i] == 0:
                plt.scatter(x[i], y[i], c='r')
            else:
                plt.scatter(x[i], y[i], c='b')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    # hyperparameter
    N = 50
    lr = 0.001
    mx1 = my1 = 1
    mx2 = my2 = 3
    vx1 = vy1 = 2
    vx2 = vy2 = 4
    
    # generate data from gaussian distribution
    generator = Random_data_generator()
    D1, D2 = [], []
    for point in range(N):
        x1 = generator.uni_gaussian_datagen(mx1, vx1)
        y1 = generator.uni_gaussian_datagen(my1, vy1)
        D1.append((x1, y1))
        
        x2 = generator.uni_gaussian_datagen(mx2, vx2)
        y2 = generator.uni_gaussian_datagen(my2, vy2)
        D2.append((x2, y2))
    
    model = Logistic_regression(D1, D2, N)
    
    # Steepest Gradient descent
    w = np.zeros((3, 1))
    w = model.Gradient_descent(w, lr)
    pred_gd = model.predict(w)
    CM, sen, spe = model.confusion_matrix(pred_gd)
    
    print('Gradient descent:')
    model.info(w, CM, sen, spe)
    
    print('\n-------------------------------------')
    
    # Newton's method
    w = np.zeros((3, 1))
    w = model.Newtons_method(w, lr)
    pred_nt = model.predict(w)
    CM, sen, spe = model.confusion_matrix(pred_nt)
    
    print("Newton's method:")
    model.info(w, CM, sen, spe)
    
    model.visualization(D1, D2, pred_gd, pred_nt)