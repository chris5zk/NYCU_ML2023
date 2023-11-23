""" 312581039 吳致廣 Chris Wu 2023/10/09
    ML Homework 02: Online-Learning
    Inputs:
        1. A file contains many lines of binary outcomes.
        2. Parameter 'a' for the initial beta prior.
        3. Parameter 'b' for the initial beta prior
    Outputs:
        1. Binomial likelihood.
        2. Beta prior and Beta posterior.
    Function:
        1. Use Beta-Binomial conjugation to perform online learning. 
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import math


def beta_distribution(theta, a, b):
    beta_function_inverse = r(a+b) / (r(a) * r(b))
    return  beta_function_inverse * (theta**(a-1)) * ((1-theta)**(b-1))

def r(x):
    # r(x) = (x-1) * r(x-1) = (x-1) * (x-2) * r(x-2) = ... = (x-1)!
    return math.factorial(x-1)

def binomial(p, N, m):
    # N, m = float(N), float(m)
    print(N, m)
    return math.comb(N, m) * (p**m) * (1-p)**(N-m) 


if __name__ == '__main__':
     
    input_file_path = './input/testfile.txt'
    trails = []
    with open(input_file_path, 'r') as f:
        for trail in f.readlines():
            trails.append(trail.strip())
    
    # initial
    theta = np.linspace(0, 1, 100)
    a, b = 0, 0
    
    # conjugate prior/posterior
    for idx, trail in enumerate(trails):
        print(f'case {idx+1}:', trail)
        
        # likelihood
        N = len(trail)
        m = trail.count('1')
        p = m / N
        print('Likelihood:', binomial(p, N, m))   # binomial
        print('Beta prior:\t', a, b)
        
        # renew a, b
        a += m
        b += N - m
        print('Beta posterior:\t', a, b, '\n')
        
        # plot the curve
        plt.subplot(1, 2, 1)
        plt.title('Likelihood'), plt.xlabel('theta'), plt.ylabel('y'), plt.legend(loc='upper left')
        plt.plot(theta, binomial(theta, N, m), label=f'N={N} m={m}')
        
        plt.subplot(1, 2, 2)
        plt.title('Prior/Posterior'), plt.xlabel('theta'), plt.ylabel('y'), plt.legend(loc='upper left')
        plt.plot(theta, beta_distribution(theta, a, b), label=f'a={a} b={b}')

    plt.show()