# 文件功能： 实现 K-Means 算法

import numpy as np
from numpy.random import default_rng

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        m = data.shape[0]
        rng = default_rng()
        center_inds = rng.choice(m, self.k_, replace=False) # false: ensure difference
        centers = data[center_inds, :]

        dist_mat = np.empty((m, self.k_))
        for _ in range(self.max_iter_):
            for i, center in enumerate(centers):
                dist_mat[:, i] = np.linalg.norm(data - center, axis=1)

            label_mat = np.argmin(dist_mat, axis = 1)
            # distortion: the sum of square distances of points to their matching centers
            distortion = (np.min(dist_mat, axis=1)**2).sum()
            if distortion < self.tolerance_:
                break
            
            for i in range(self.k_):
                centers[i, :] = np.mean(data[label_mat==i], axis=0)
        
        self.centers_ = centers


    def predict(self, p_datas):
        result = []
        # 作业2
        m = p_datas.shape[0]
        dist_mat = np.empty((m, self.k_))
        for i, center in enumerate(self.centers_):
            dist_mat[:, i] = np.linalg.norm(p_datas - center, axis=1)
        result = np.argmin(dist_mat, axis = 1).tolist()

        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

