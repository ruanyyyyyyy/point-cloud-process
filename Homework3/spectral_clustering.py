import numpy as np
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
import numpy.linalg as LA
from sklearn.cluster import KMeans

class spetral_clustering(object):
    def __init__(self, n_clusters=2, nnk=7, normalized=True):
        self.n_clusters=n_clusters
        self.nnk_ = nnk
        self.labels_ = np.empty(0)
        self.normalized_ = normalized
       
    
    def fit(self, data):
        # get adjacency matrix W
        m = data.shape[0]
        tree = KDTree(data)
        W = np.zeros((m,m))

        for di, datum in enumerate(data):
            ndists, ninds = tree.query([datum], self.nnk_+1)
            
            ninds = ninds[0]
            ndists = ndists[0]
            for ni, ndist in zip(ninds, ndists):
                # the point itself would be among its knn. Skip
                if ni==di:
                    continue
                W[di][ni] = W[ni][di] = 1/ndist

        D = np.diag(W.sum(axis=1))
        L = D - W
        if self.normalized_:
            L = np.matmul(LA.inv(D), L)
        
        eigvals, eigvecs = LA.eig(L)
        sorted_idx = np.argsort(eigvals)
        V = eigvecs[:, sorted_idx[:self.n_clusters]] # (n,k)
        V = V.real #TODO: eigvecs would have complex number which kmeans cannot handle. pick the real part. Not sure if this solution is correct.

        self.labels_ = KMeans(n_clusters=self.n_clusters).fit_predict(V)
    
    def predict(self):
        return self.labels_

if __name__ == '__main__':
    pass