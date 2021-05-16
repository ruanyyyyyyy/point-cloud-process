import numpy as np
import open3d as o3d 
from pyntcloud import PyntCloud
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm

B = 11

def FPFH(dataWnormal, point_label, tree, radius, B):
    '''
    Parameters:
        dataWnormal, point_label, tree, radius, B
    Returns:
        FPFH histograms
    '''
    data = dataWnormal[:, :3]
    nei_hist, count = 0, 0
    nei_labels = tree.query_radius(data[point_label].reshape(1,-1), radius)[0]
    nei_labels = np.asarray(list(set(nei_labels) - set([point_label]))) # delete key point in neighbors
    histograms = SPFH(dataWnormal, point_label, tree, radius, B).astype(np.double)
    for neighbor_label in tqdm(nei_labels):
        count += 1
        # weighted sum, calculate nei_hist
        wk = 1 / np.linalg.norm(data[point_label] - data[neighbor_label])
        nei_hist += wk * SPFH(dataWnormal, neighbor_label, tree, radius, B)

    histograms += nei_hist / count 

    return histograms

def SPFH(data_Wnormal, point_label, tree, radius, B):
    '''
    Parameters:
        data_Wnormal - w x 6
        point_label - index of a key point
        raidus - search radius
        B - # bins
    Returns:
        histograms - the descriptor histograms, N x 15
    '''
    alpha, phi, theta = [], [], []
    point = data_Wnormal[point_label]

    nei_labels = tree.query_radius(data_Wnormal[point_label][0:3].reshape(1,-1), radius)[0]
    nei_labels = np.asarray(list(set(nei_labels) - set([point_label])))
    local_points = data_Wnormal[nei_labels]

    n1 = data_Wnormal[point_label][3:].reshape(1,-1)
    p1 = data_Wnormal[point_label][0:3].reshape(1,-1)
    for neighbor in local_points:
        # calculate alpha, phi, theta
        n2 = neighbor[3:].reshape(1,-1)
        p2 = neighbor[0:3].reshape(1,-1)
        u = n1
        v = np.cross((p2 - p1), u)
        w = np.cross(u, v)
        
        alpha.append(np.dot(v, n2.T))
        phi.append(np.dot(u, (p2-p1).T/np.linalg.norm(p2-p1)))
        theta.append(np.arctan2(np.dot(w,n2.T), np.dot(u,n2.T)))

    # start voting
    alpha_hist, _ = np.histogram(np.array(alpha), bins=B, range=(-1,1), density=True) # (11,)
    phi_hist, _ = np.histogram(np.array(phi), bins=B, range=(-1,1), density=True)
    theta_hist, _ = np.histogram(np.array(theta), bins=B, range=(-np.pi, np.pi), density=True)
    histograms = np.hstack((alpha_hist, phi_hist, theta_hist)) # (33,)
    return histograms 

def visulization(pc_path):
    point_cloud_pynt = PyntCloud.from_file(pc_path, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    pc_array = np.asarray(point_cloud_o3d.points)
    o3d.visualization.draw_geometries([point_cloud_o3d]) 


if __name__=="__main__":
    pc_path = "./data/chair/chair_0056.txt"
    
    with open(pc_path, 'r') as f:
        point_cloud = f.readlines()
    
    pc_list = [] 
    for line in point_cloud:
        line = line.strip().split(',') 
        pc_list.append([float(entry) for entry in line])
    
    pc_array = np.asarray(pc_list)

    tree = KDTree(pc_array[:,:3], leaf_size = 20)
    histograms1 = FPFH(pc_array, 8628, tree, 0.3, B)
    histograms2 = FPFH(pc_array, 1119, tree, 0.3, B)
    histograms3 = FPFH(pc_array, 2696, tree, 0.3, B)
    
    plt.figure(figsize=(10,5))
    
    # x_coord = np.linspace(0, 32, 1)
    x_coord = list(range(33))
    plt.plot(x_coord, histograms1,color="deeppink", linewidth=2, linestyle=':', label='point1', marker='o')
    plt.plot(x_coord, histograms2,color="blue", linewidth=2, linestyle=':', label='point2', marker='o')
    plt.plot(x_coord, histograms3,color="green", linewidth=2, linestyle=':', label='point3', marker='o')

    plt.legend(loc=2)
    plt.show()
    