from sklearn.neighbors import KDTree
import math
import numpy as np  


def read_pcd_from_file(file):
  np_pts = np.zeros(0)
  with open(file, 'r') as f:
    pts = []
    for line in f:
      one_pt = list(map(float, line[:-1].split(',')))
      pts.append(one_pt[:3])
    np_pts = np.array(pts)
  return np_pts


def PCA(data, correlation=False, sort=True):
  # 屏蔽开始
  # remove mean
  mean = np.mean(data, axis=0)
  mean_removed = data - mean
  # get the cov matrix of sample
  cov_matrix = np.cov(mean_removed, rowvar=0)
  # cal eigenvalues and eigenvectors
  eigenvalues, eigenvectors = np.linalg.eig(np.mat(cov_matrix))
  # 屏蔽结束

  if sort:
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]

  return eigenvalues, eigenvectors, mean_removed


def extract_description_btw_two_points(pt1, pt2, n1, n2):
  d = np.linalg.norm(pt2 - pt1)
  n2 = n2 / np.linalg.norm(n2)
  n1 = n1 / np.linalg.norm(n1)
  u = n1
  v = np.cross(u, (pt2 - pt1) / d)
  w = np.cross(u, v)
  alpha = math.acos(np.dot(v, n2))
  phi = math.acos(np.dot(u, (pt2 - pt1) / d))
  tmp = np.dot(u, n2)
  if tmp == 0.0:
    tmp = 0.000001
  theta = math.atan(np.dot(w, n2) / tmp) + math.pi * 0.5
  return [alpha, phi, theta]


def get_histogram(descriptions):
  resolution = 180.0 / 12.0
  ref_vals = [resolution * i + resolution * 0.5 for i in range(12)]
  histograms = [[0.0 for _ in range(12)] for _ in range(3)]
  for description in descriptions:
    for i in range(len(description)):
      theta = description[i] * 180.0 / math.pi
      assert(theta <= 180 and theta >= 0)
      if theta >= 180.0:
        theta = 179.9
      curr_id = math.floor(theta / resolution)
      left_id = (curr_id - 1 + 12) % 12
      right_id = (curr_id + 1) % 12
      assert(math.fabs(theta - ref_vals[curr_id]) <= 180.0)
      assert(math.fabs(theta - ref_vals[left_id]) <= 180.0)
      assert(math.fabs(theta - ref_vals[right_id]) <= 180.0)
      dist_1 = max(min(math.fabs(theta - ref_vals[curr_id]), 180.0 - math.fabs(theta - ref_vals[curr_id])), 10e-9)
      dist_2 = max(min(math.fabs(theta - ref_vals[left_id]), 180.0 - math.fabs(theta - ref_vals[left_id])), 10e-9)
      dist_3 = max(min(math.fabs(theta - ref_vals[right_id]), 180.0 - math.fabs(theta - ref_vals[right_id])), 10e-9)
      w1 = 1.0 / (dist_1 * dist_1)
      w2 = 1.0 / (dist_2 * dist_2)
      w3 = 1.0 / (dist_3 * dist_3)
      w_sum = w1 + w2 + w3
      histograms[i][curr_id] += w1 / w_sum
      histograms[i][left_id] += w2 / w_sum
      histograms[i][right_id] += w3 / w_sum
  res = []
  for h in histograms:
    res.extend(h)
  res = np.array(res)
  res /= np.linalg.norm(res)
  return list(res)


def fpfh(np_pts, feature_points, feature_radius=0.05):
  tree = KDTree(np_pts, leaf_size=4)
  # find neighbor points for all feature points
  neighbor_ids_array = tree.query_radius(feature_points, feature_radius)
  normal_neighbor_ids_array = tree.query_radius(feature_points, 0.03)
  histograms = []
  for i, neighbor_ids in enumerate(neighbor_ids_array):
    descriptions = []
    ref_pt = feature_points[i]
    np_neighors = np_pts[neighbor_ids]
    # get normal vec of ref_pt
    _, vectors, _ = PCA(np_pts[normal_neighbor_ids_array[i]])
    normal = vectors[:, -1]  # 法向量
    normal_1 = np.array([normal[0, 0], normal[1, 0], normal[2, 0]])
    # get neighbors of each neighbor
    ngbsngb_ids_array = tree.query_radius(np_neighors, feature_radius)
    normal_ngbsngb_ids_array = tree.query_radius(np_neighors, 0.03)
    # for each neighbor
    for j, ngbsngb_ids in enumerate(ngbsngb_ids_array):
      ngb_pt = np_neighors[j]
      np_ngbsngb = np_pts[ngbsngb_ids]
      # get normal vec of neighbor_pt
      _, vectors, _ = PCA(np_pts[normal_ngbsngb_ids_array[j]])
      normal = vectors[:, -1]  # 法向量
      if normal.shape[0] < 3:
        continue
      normal_2 = np.array([normal[0, 0], normal[1, 0], normal[2, 0]])
      # extract descriptor
      if np.linalg.norm(ref_pt - ngb_pt) > 0.00001:
        description = extract_description_btw_two_points(ref_pt, ngb_pt, normal_1, normal_2)
        descriptions.append(description)
      # get neighbors of each neighbor's neighbor
      ngbsngbsngb_ids_array = tree.query_radius(np_ngbsngb, 0.03)
      for k, ngbsngbsngb_ids in enumerate(ngbsngbsngb_ids_array):
        ngbsngb_pt = np_ngbsngb[k]
        np_ngbsngbsngb = np_pts[ngbsngbsngb_ids]
        # get normal vec of neighbor_pt
        _, vectors, _ = PCA(np_ngbsngbsngb)
        normal = vectors[:, -1]  # 法向量
        if normal.shape[0] < 3:
          continue
        normal_3 = np.array([normal[0, 0], normal[1, 0], normal[2, 0]])
        # extract descriptor
        if np.linalg.norm(ngb_pt - ngbsngb_pt) > 0.00001:
          description = extract_description_btw_two_points(ngb_pt, ngbsngb_pt, normal_2, normal_3)
          descriptions.append(description)  
    histogram = get_histogram(descriptions)
    histograms.append(histogram)
  return histograms

if __name__ == "__main__":
  np_pts = read_pcd_from_file("../data/stool_0091.txt")
  similiar_points = [[0.4691, -0.6959, -0.2416]]
  fpfh(np_pts, np.array(similiar_points))
