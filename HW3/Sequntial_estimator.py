""" 312581039 吳致廣 Chris Wu 2023/10/22
    ML Homework 03:
    2. Sequential Estimator
        Inputs: m, s of univariate gaussian data generator
        Outputs: new data point and current estimates of m, s
"""

from Rand_datagen import Random_data_generator


class Sequential_estimator:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0
        self.variance = 0   # unbiased variance
    
    def online_estimate(self, x) -> None:
        # Welford's online algorithm
        self.count += 1
        self.variance = self.variance + ((x - self.mean)**2 / self.count) - (self.variance / (self.count-1)) if self.count > 1 else 0
        self.mean = self.mean + (x - self.mean) / self.count
        

if __name__ == '__main__':
    
    mean, variance = 3.0, 5.0
    epsilon = 0.05
    
    generator = Random_data_generator()
    estimator = Sequential_estimator()
    
    print("Data point source function: N({}, {})".format(mean, variance))
    while(abs(estimator.mean-mean) > epsilon or abs(estimator.variance-variance) > epsilon or estimator.count < 5000):
        point = generator.uni_gaussian_datagen(mean, variance)
        estimator.online_estimate(point)
        print("Add data point {}: {}".format(estimator.count, point))
        print("Mean = {}  Variance = {}".format(estimator.mean, estimator.variance))