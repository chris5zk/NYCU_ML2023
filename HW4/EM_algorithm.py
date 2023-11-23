""" 312581039 吳致廣 Chris Wu 2023/11/05
    ML Homework 04: EM algorithm
        - Inputs: MNIST data and label
        - Outputs:
            1. Confusion matrix, sensitivity and specificity
            2. imagination of numbers
        - Function:
            1. Binning the gray level value into two bins. Treating all pixels as random variables following Bernoulli distributions.
            2. Use EM algorithm to cluster each image into ten groups(0 ~ 9).
"""

import gzip
import numpy as np


class EM_algorithm:
    def __init__(self, X_path, Y_path) -> None:
        self.imgs_file = self.read_files(X_path)
        self.lbs_file = self.read_files(Y_path)

    def read_files(self, path):
        with gzip.open(path) as file:
            return file.read()

    def load_data(self):
        self.size, self.flat = self.get_header()
        imgs = np.array(list(self.imgs_file[16:])).reshape(self.size, self.flat)
        labels = np.array(list(self.lbs_file[8:]))
        imgs = imgs // 128
        return imgs, labels

    def get_header(self):
        size = int.from_bytes(self.imgs_file[4:8], byteorder='big')
        h, w = int.from_bytes(self.imgs_file[8:12], byteorder='big'), int.from_bytes(self.imgs_file[12:16], byteorder='big')
        return size, h*w
    
    def fit(self, X):
        # init
        lamb = np.random.rand(10)
        w = np.random.rand(self.size, 10)               # 60000 * 10
        p = np.random.rand(10, self.flat) / 2 + 0.25    # 10 * 784
        
        i = 0
        delta = 100
        p_prev = np.zeros((10, self.flat))
        
        while i <= 10 or delta > 10:
            i += 1
            # E step: calculate w
            for idx in range(self.size):    # 0 ~ 60000
                w[idx] = lamb * np.prod(p ** X[idx], axis=1) * np.prod((1 - p) ** (1 - X[idx]), axis=1)
                w[idx] /= np.sum(w[idx])
            
            # M step: renew p, lamb
            lamb = np.sum(w, axis=0) / self.size
            p = np.dot(w.T, X) / np.sum(w, axis=0).reshape(10, 1)
            p[p == 0] = 1e-5
            
            mapping = np.array([digit for digit in range(10)])
            self.imagination(p, mapping)
            delta = np.sum(abs(p - p_prev))
            print('No. of Iteration: {}\tDifference: {}'.format(i, delta))
            print("--------------------------------------------------------")
            p_prev = p
        
        return p, w, i
          
    def imagination(self, p, mapping, labeled=False):
        for digit in range(10):
            if labeled:
                print('labeled', end=" ")
            print('class {}:'.format(digit))
            real_digit = mapping[digit]
            for i in range(28):
                for j in range(28):
                    print('1', end=" ") if p[real_digit][i*28 + j] > 0.5 else print('0', end=" ")
                print()
            print()
        
    def label_assign(self, Y, w):
        table = np.zeros((10, 10))
        adjust_label = np.zeros(10, dtype='uint32')
        for idx in range(self.size):
            table[Y[idx], np.argmax(w[idx])] += 1
        for _ in range(10):
            index = np.argmax(table) # 0 ~ 99
            cluster = index % 10
            label = index // 10
            adjust_label[label] = cluster
            table[label, :] = 0
            table[:, cluster] = 0
        return adjust_label # label -> cluster link

    def info(self, Y, w, adjust_label, i):
        # something wrong
        CM = np.zeros((10, 2, 2))
        correct = 0
        
        for idx in range(self.size):
            cluster = np.argmax(w[idx])
            pred = np.where(adjust_label == cluster)[0]
            y = Y[idx]
            # print(adjust_label, np.argmax(w[idx]), pred, y)
            for digit in range(10):
                CM[digit, 0, 0] = CM[digit, 0, 0] + 1 if y==digit and pred==digit else CM[digit, 0, 0]  # TP
                CM[digit, 0, 1] = CM[digit, 0, 1] + 1 if y==digit and pred!=digit else CM[digit, 0, 1]  # FN
                CM[digit, 1, 0] = CM[digit, 1, 0] + 1 if y!=digit and pred==digit else CM[digit, 1, 0]  # FP
                CM[digit, 1, 1] = CM[digit, 1, 1] + 1 if y!=digit and pred!=digit else CM[digit, 1, 1]  # TN
                
        for digit in range(10):
            print('--------------------------------------------------------')
            print('Confusion Matrix {}'.format(digit))
            print("\t\t\tPredict number {}\tPredict not number {}".format(digit, digit))
            print("Is number {}\t\t{}\t\t\t{}".format(digit, CM[digit, 0, 0], CM[digit, 0, 1]))
            print("Isn't number {}\t\t{}\t\t\t{}\n".format(digit, CM[digit, 1, 0], CM[digit, 1, 1]))
            sen = CM[digit, 0, 0] / (CM[digit, 0, 0] + CM[digit, 0, 1])
            spe = CM[digit, 1, 1] / (CM[digit, 1, 0] + CM[digit, 1, 1])
            print(f"\nSensitivity (Successfully predict number {digit})\t: {sen}")
            print(f"Specificity (Successfully predict not number {digit}) : {spe}")
            correct += CM[digit, 0, 0]
            
        print('Total iteration to converge: {}'.format(i))
        print('Total error rate: {}'.format(1 - correct/self.size))

if __name__ == '__main__':
    X_path = './input/train-images-idx3-ubyte.gz'
    Y_path = './input/train-labels-idx1-ubyte.gz'
    
    model = EM_algorithm(X_path, Y_path)
    X_train, Y_train = model.load_data()
    
    p, w, i = model.fit(X_train)   # w: predict label, p: imagination
    adjust_label = model.label_assign(Y_train, w)
    model.imagination(p, adjust_label, labeled=True)
    print(adjust_label)
    model.info(Y_train, w, adjust_label, i)