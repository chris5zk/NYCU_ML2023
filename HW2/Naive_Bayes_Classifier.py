""" 312581039 吳致廣 Chris Wu 2023/10/03
    ML-Homework 02: Naive Bayes Classifier
    Inputs: 
        1. Training images from MNIST.
        2. Training label data from MNIST.
        3. Testing images from MNIST.
        4. Testing label data from MNIST.
        5. Toggle Options(discrete or continuous).     
    Outputs:
        1. posterior(in log scale) of ten categories for each image in INPUT 3.
        2. prediction where is the category having highest posterior and comparing with INPUT 4.
        3. imagination of numbers.
            - 0 represents white pixel and 1 represents black pixel.
            - The pixel is 0 when Bayes classifier expect the pixel in this position should less then 128 in original image, otherwise is 1.
        4. Error rate in the end.
    Function: 
        1. In discrete mode:
            - Tally the frequency of the values of each pixel into 32 bins. Use a peudocount (the minimum value in other bins) to avoid empty bin.
        2. In continuous mode:
            - Use MLE to fit a Gaussian distribution for the value of each pixel. Perform Naive Bayes classifier.
"""

import gzip
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


class Naive_Bayes_Classifier:
    def __init__(self, mode):
        self.mode = mode    # 0 for discrete / 1 for continuous
        self.level = 8      # divisible by 256
        self.bins = 256 // self.level
        self.discrete_threshold = 16
        self.continuous_threshold = 128
        self.cls_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
    def read_MNIST_files(self, X_path, Y_path):
        with gzip.open(X_path) as file:
            self.imgs_file = file.read()
        with gzip.open(Y_path) as file:
            self.lbs_file = file.read()
            
    def get_header(self):
        ## header information
        # image
        magic_img = int.from_bytes(self.imgs_file[0:4], byteorder='big')
        size_img = int.from_bytes(self.imgs_file[4:8], byteorder='big')
        h, w = int.from_bytes(self.imgs_file[8:12], byteorder='big'), int.from_bytes(self.imgs_file[12:16], byteorder='big')
        # label
        magic_lbs = int.from_bytes(self.lbs_file[0:4], byteorder='big')
        size_lbs = int.from_bytes(self.lbs_file[4:8], byteorder='big')
        return magic_img, magic_lbs, size_img, size_lbs, h, w

    def load_data(self):
        _, _, self.size, _, h, w = self.get_header()
        imgs = np.array(list(self.imgs_file[16:])).reshape(self.size, h*w)
        labels = np.array(list(self.lbs_file[8:]))
        if self.mode == 0:
            imgs = imgs // self.level
        return imgs, labels
    
    def fit(self, X_train, Y_train):
        # Seems each pixel as an independent variable
        self.prior = self.priors(Y_train)
        if self.mode == 0:
            # A probability in multi-nomial distribution for a certain class
            self.multinomial = self.distribution(X_train, Y_train)
        elif self.mode == 1:
            # A probability in Gaussian distribution for a certain class
            self.means, self.standards = self.distribution(X_train, Y_train)
    
    def priors(self, Y):
        prior = []
        for cls in self.cls_all:
            label = Y == cls
            prior.append(len(Y[label])/len(Y))
        return prior
    
    def distribution(self, X, Y):
        if self.mode == 0:
            multinomial = {}
            # each class
            for cls in self.cls_all:
                # masking(select determine class)
                label = Y == cls
                bin_count = np.zeros((self.bins, X.shape[1]))
                for bin in range(self.bins):
                    bin_count[bin] = np.count_nonzero(X[label]==bin, axis=0)
                bin_count[bin_count == 0] = np.min(bin_count[bin_count!=0])
                multinomial[cls] = bin_count / X[label].shape[0]
            return multinomial
                
        elif self.mode == 1:   
            means, standards = {}, {}
            # each class
            for cls in self.cls_all:
                # masking(select determine class)
                label = Y == cls
                means[cls] = np.mean(X[label], axis=0)
                standards[cls] = np.std(X[label], axis=0)
            return means, standards
    
    def predict(self, X_test, Y_test):
        posterior = np.zeros((len(Y_test), len(self.cls_all)))    
        if self.mode == 0:
            j_index = np.tile(np.arange(X_test.shape[1]), (len(Y_test), 1))
            for cls in self.cls_all:
                table = self.multinomial[cls]
                likelihood = table[X_test, j_index]     # (bin_value, pixel_position)
                posterior[:, cls] = np.sum(np.log(likelihood), axis=1) + np.log(self.prior[cls])
            pred = np.argmax(posterior, axis=1)
        elif self.mode == 1:
            self.Gaussian = {}
            for cls in self.cls_all:
                mean = self.means[cls]
                var = np.square(self.standards[cls]) + 1000     # smoothing
                likelihood = 1 / np.sqrt(2*np.pi*var) * np.exp(-(np.square(X_test - mean)) / (2*var))
                posterior[:, cls] = np.sum(np.log(likelihood), axis=1) + np.log(self.prior[cls])
            pred = np.argmax(posterior, axis=1)
        return pred, posterior
    
    def error_rate(self, pred, Y):
        error_rate = np.count_nonzero(pred-Y)/len(Y)
        print(f'Error rate: {error_rate}')
    
    def imagination_numbers(self):
        if self.mode == 0:
            threshold = 16
            for cls in self.cls_all:
                print(f'{cls}:')
                # biggest probability of bins
                imagination = np.argmax(self.multinomial[cls], axis=0) >= threshold
                for i in range(28):
                    for j in range(28):
                        print('1' if imagination[i*28+j] else '0', end=' ')
                    print()
                print()
            print()
        elif self.mode == 1:
            threshold = 128   
            for cls in self.cls_all:
                print(f'{cls}:')
                # biggest probability = mean
                imagination = self.means[cls] >= threshold
                for i in range(28):
                    for j in range(28):
                        print('1' if imagination[i*28+j] else '0', end=' ')
                    print()
                print()
            print()
            
    def plot_confusion_matrix(self, pred, Y):
        self.error_rate(pred, Y)
        cm = np.zeros((10, 10), dtype='uint32')
        for idx in range(len(Y)):
            cm[Y[idx], pred[idx]] += 1
        df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
        plt.figure(figsize = (10, 7)), sn.heatmap(df_cm, annot=True, fmt=".1f", cmap="crest")
        plt.xlabel('Prediction'), plt.ylabel('Label')
        plt.show()

    def result(self, posterior, pred, Y):
        # for idx in range(len(pred)):
        #     cls_all = np.unique(Y)
        #     print(f'Image\t{idx} - Posterior (in log scale):')
        #     marginal_posterior = posterior[idx-1] / np.sum(posterior[idx])
        #     for cls in cls_all:
        #         print(f'{cls}: {marginal_posterior[cls]}')
        #     print(f'Prediction: {pred[idx]}, Ans: {Y[idx]}\n')
        while True:
            idx = input('Which image do you want to see?(1~10000, others for stop):')
            if idx.isdigit():
                idx = int(idx)
                if 1 <= idx <= 10000:
                    cls_all = np.unique(Y)
                    print(f'Image\t{idx} - Posterior (in log scale):')
                    marginal_posterior = posterior[idx-1] / np.sum(posterior[idx-1])
                    for cls in cls_all:
                        print(f'{cls}: {marginal_posterior[cls]}')
                    print(f'Prediction: {pred[idx-1]}, Ans: {Y[idx-1]}\n')
                else:
                    break
            else:
                break
        
if __name__ == '__main__':
    
    X_train_path, Y_train_path = './input/train-images-idx3-ubyte.gz', './input/train-labels-idx1-ubyte.gz'
    X_test_path, Y_test_path = './input/t10k-images-idx3-ubyte.gz', './input/t10k-labels-idx1-ubyte.gz'
 
    mode = input('Please choose the mode(0 for discrete / 1 for continuous): ')
 
    classifier = Naive_Bayes_Classifier(int(mode)) 
    
    classifier.read_MNIST_files(X_train_path, Y_train_path)
    X_train, Y_train = classifier.load_data()
    classifier.fit(X_train, Y_train)
    
    classifier.read_MNIST_files(X_test_path, Y_test_path)
    X_test, Y_test = classifier.load_data()
    pred, posterior = classifier.predict(X_test, Y_test)
    
    classifier.imagination_numbers()
    classifier.plot_confusion_matrix(pred, Y_test)
    classifier.result(posterior, pred, Y_test)