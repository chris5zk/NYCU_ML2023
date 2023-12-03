""" 2023 NYCU ML Homework 05: SVM - Chris Wu 2023/11/29
    INPUT: MNIST dataset
        - Training data:
            X_train.csv (5000 x 784 matrix), Y_train.csv (5000 x 1 matrix)
        - Testing data:
            X_test.csv (2500 x 784 matrix), Y_test.csv (2500 x 1 matrix)
    TASK1:
        Use different kernel function (linear, polynomial and RBF) and compare their performance.
    TASK2:
        Use C-SVC (soft-margin SVM) and do the grid search for finding parameters which have the
        best performance on cross-validation.
    TASK3:
        Use linear kernel + RBF kernel together and compare its performance w.r.t. others.
"""

import json
import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist

def load_MNIST_data(root):
    filelist = ['X_train.csv', 'Y_train.csv', 'X_test.csv', 'Y_test.csv']
    data = []
    
    for file in filelist:
        path = root + file
        data.append(np.loadtxt(path, delimiter=',', skiprows=0, usecols=None, dtype=np.float64))
        
    assert(data[0].shape == (5000, 784))
    assert(data[1].shape == (5000,))
    assert(data[2].shape == (2500, 784))
    assert(data[3].shape == (2500,))
    
    return data[0], data[1], data[2], data[3]


class SVM:
    def __init__(self, X, Y) -> None:
        # training data
        self.X = X
        self.Y = Y
        # grid search
        self.kernel = {'Linear': 0, 'Polynomial': 1, 'Radial Basis': 2}
        self.cost = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
        self.coef0 = [1e-1, 0, 1]
        self.gamma = [1e-3, 1e-1, 1e1]
        self.degree = [1, 2, 3]
        
    def train(self, args, preK=None) -> None:
        if preK:
            kernel = self.precomputed_kernel(self.X, self.X)
            format_kernel = np.hstack((np.arange(1, self.X.shape[0]+1).reshape(-1, 1), kernel))
            self.model = svm_train(self.Y, format_kernel, args)
        else:
            self.model = svm_train(self.Y, self.X, args)
        
    def predict(self, X_test, Y_test) -> tuple:
        return svm_predict(Y_test, X_test, self.model)

    def precomputed_kernel(self, X1, X2) -> np.ndarray:
        gamma = 1/784
        kernel_linear = X1 @ X2.T
        kernel_RBF = np.exp(-gamma * cdist(X1, X2, 'euclidean'))
        return kernel_linear + kernel_RBF
    
    def gridSearch(self) -> None:
        # grid search result dictionary
        best = dict()
        best['Best'] = dict()
        best['Best']['accuracy'] = 0
        
        for kernel, args in self.kernel.items():
            
            print(f'\033[31mGrid searching for {kernel} kernel...\033[0m') 
            best_acc, best_opt = 0, ''
            
            for c in self.cost:   
                if kernel == 'Linear':
                    # linear -> -c: cost
                    opt = f'-t {args} -c {c}'
                    best_acc, best_opt = self.bestOpt(opt, best_acc, best_opt)
                    
                elif kernel == 'Polynomial':
                    # polynomial -> -g: gamma, -r: coef0, -d: degree, -c: cost
                    for g in self.gamma:
                        for d in self.degree:
                            for c0 in self.coef0:
                                opt = f'-t {args} -c {c} -g {g} -d {d} -r {c0}'
                                best_acc, best_opt = self.bestOpt(opt, best_acc, best_opt)
                                
                elif kernel == 'Radial Basis':
                    # RBF -> -g: gamma, -c: cost
                    for g in self.gamma:
                        opt = f'-t {args} -c {c} -g {g}'
                        best_acc, best_opt = self.bestOpt(opt, best_acc, best_opt)
                        
            best[kernel] = dict()
            best[kernel]['option'] = best_opt
            best[kernel]['accuracy'] = best_acc
            print(f'\033[4;32m{kernel} kernel\033[0m \033[32mBest option: {best_opt}\t{best_acc}%\033[0m')
            
            if best_acc > best['Best']['accuracy']:    
                best['Best']['kernel'] = kernel
                best['Best']['accuracy'] = best_acc
                best['Best']['option'] = best_opt
        
        print(f"\033[1;31mBest Kernel:\033[0m \033[1;33m{best['Best']['kernel']} Kernel\033[0m \033[0;33m{best['Best']['option']}\033[0m")
        print("\033[1;31mBest Acc:\033[0m \033[1;33m{:.2f}%\033[0m".format(best['Best']['accuracy']))
        self.save_result(best)

    def bestOpt(self, opt, best_acc, best_opt) -> [float, str]:
        print('Options: ' + opt, end='\t')
        acc = svm_train(self.Y, self.X, opt + ' -v 5 -q')
        if acc > best_acc:
            return acc, opt
        else:
            return best_acc, best_opt     
    
    def save_result(self, best) -> None:
        with open('./results/gS_results.json', 'w') as f:
            json.dump(best, f, indent=4)
        

if __name__ == '__main__':
    
    dataset_root = './data/'
    X_train, Y_train, X_test, Y_test = load_MNIST_data(dataset_root)

    model = SVM(X_train, Y_train)

    input_ = input('\033[44mPlease choose the task:\033[0m\nPart 1. compare three types of kernel\nPart 2. grid search\nPart 3. user-defined kernel\n')
    
    if input_ == '1':
        # Linear kernel
        print("\033[46mLinear Kernel SVM \033[0m")
        model.train('-t 0 -q')
        model.predict(X_test, Y_test)
        
        # Polynomial kernel
        print("\033[45mPolynomial Kernel SVM \033[0m")
        model.train('-t 1 -d 1 -q')
        model.predict(X_test, Y_test)
        
        # Radial Basis Function kernel
        print("\033[44mRBF Kernel SVM \033[0m")
        model.train('-t 2 -q')
        model.predict(X_test, Y_test)
    
    elif input_ == '2':
        model.gridSearch()
    
    elif input_ == '3':
        model.train('-t 4 -q', preK=True)
        kernel = model.precomputed_kernel(X_train, X_test)
        format_kernel = np.hstack((np.arange(1, X_test.shape[0]+1).reshape(-1, 1), kernel.T))
        model.predict(format_kernel, Y_test)
        
    else:
        print('\033[41mWrong option!\033[0m')