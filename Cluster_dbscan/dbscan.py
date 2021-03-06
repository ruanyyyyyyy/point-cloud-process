from scipy.spatial import KDTree
import numpy as np

class DBSCAN(object):
    def __init__(self, radius=0.5, Min_Pts=10):
        self.radius = radius
        self.Min_Pts = Min_Pts
        self.labels_ = None
    
    def fit(self, data):
        N = data.shape[0]
        labels = -1 * np.ones(N)
        unvisited = list(range(N))
        neighbor_unvisited = []
        label = -1
        tree = KDTree(data, leafsize=32)
        while len(unvisited) > 0:
            ind = unvisited.pop()
            neighbors_inds = tree.query_ball_point(data[ind], self.radius)
            if len(neighbors_inds) < self.Min_Pts:
                labels[ind] = -1 # -1: noise point
                continue
            else:
                label += 1
                labels[ind] = label
                neighbor_unvisited.extend(neighbors_inds)
                while len(neighbor_unvisited)>0:
                    cur_ind = neighbor_unvisited.pop()
                    if cur_ind in unvisited:
                        labels[cur_ind] = label
                        cur_nnids = tree.query_ball_point(data[cur_ind], self.radius)
                        if len(cur_nnids) > self.Min_Pts:
                            neighbor_unvisited.extend(cur_nnids)

                        unvisited.remove(cur_ind)
        
        self.labels_ = labels.astype(np.int32)    
    def predict(self):
        return self.labels_


