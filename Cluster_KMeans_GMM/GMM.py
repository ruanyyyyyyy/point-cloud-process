# Implement GMM

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50, tol=0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.means = None # (k, dim)
        self.covs = None # (k, dim, dim)
        self.weights = np.ones((n_clusters,1))/n_clusters
        self.tol = tol
    
    def fit(self, data):
        dim = data.shape[1]
        self.means = np.random.random((self.n_clusters, dim))
        self.covs = np.array(self.n_clusters * [np.identity(dim)])
        last_nll = float("inf")

        for _ in range(self.max_iter):
            ### E-step, cal gamma
            gamma = np.empty((self.n_clusters, data.shape[0])) # (k, N)
            for i in range(self.n_clusters):
                gamma[i, :] = multivariate_normal.pdf(data, mean=self.means[i], cov=self.covs[i])
            gamma = gamma * self.weights # (k, N) x (k, 1)
            gamma = gamma / np.sum(gamma, axis=0, keepdims=True) # (k, N) / (1, N)

            ### M-step
            N_k = np.sum(gamma, axis=1, keepdims=True) # (k, 1)
            ## update mu
            self.means = np.matmul(gamma, data)/N_k # (k, N) * (N, dim)/(k, 1) = (k, dim)
            ## update Var
            diff_xu = (data[np.newaxis,...]-self.means[:,np.newaxis,:])[..., np.newaxis] #(1, N, dim) - (k, 1, dim) = (k, N, dim)-> (k,N,dim,1) only dim=1 can be broadcast!
            diff_T = np.transpose(diff_xu, (0,1,3,2)) #(k,N,1,dim)
            diff_mul = np.matmul(diff_xu, diff_T)
            self.covs = (np.sum(gamma[..., np.newaxis,  np.newaxis]* diff_mul, axis=1))/N_k[...,np.newaxis]  
            # (k,N,dim,1) * (k,N,1,dim) = (k,N,dim,dim)
            # (k,N,1,1) .* (k,N,dim,dim) = (k,N,dim,dim)
            # sum = (k, dim, dim)
            # N_k->(k,1,1) (k,dim,dim)-(k,1,1)=(k,dim,dim)
            # matmul: Stacks of matrices are broadcast together. So should use matmul for matrix instead of dot

            ## update pi
            self.weights = N_k / data.shape[0]

            # negative log likelihood
            gamma = np.empty((self.n_clusters, data.shape[0])) # (k, N)
            for i in range(self.n_clusters):
                gamma[i, :] = multivariate_normal.pdf(data, mean=self.means[i], cov=self.covs[i])
            gamma = gamma * self.weights # (k,N)
            nll = -np.sum(np.log(np.sum(gamma, axis=0)))
            if last_nll - nll < self.tol:
                break
            last_nll = nll 
    
    def predict(self, data):
        gamma = np.empty((self.n_clusters, data.shape[0])) # (k, N)
        for i in range(self.n_clusters):
            gamma[i, :] = multivariate_normal.pdf(data, mean=self.means[i], cov=self.covs[i])
        gamma = gamma * self.weights # (k,N)
        return np.argmax(gamma, axis=0)
        return

# Generate simulating data
def generate_X(true_Mu, true_Var):
    
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # Merge
    X = np.vstack((X1, X2, X3))
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # Generate data
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # init

    

