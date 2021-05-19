import libPCLKeypoint as libpcl
import numpy as np
import math
# import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fpfh


def show_np_pts3d(np_pts):
  pc_view = o3d.geometry.PointCloud()
  pc_view.points = o3d.utility.Vector3dVector(np_pts)
  o3d.visualization.draw_geometries([pc_view])


def point_cloud_show(point_cloud, feature_points):
  fig = plt.figure(dpi=150)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
  ax.scatter(feature_points[:, 0], feature_points[:, 1], feature_points[:, 2], cmap='spectral', s=2, linewidths=5, alpha=1, marker=".", color='red')
  plt.title('Point Cloud')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()


def dist_between_descriptor(descriptor1, descriptor2):
  assert(len(descriptor1) == len(descriptor2))
  dist = 0.0
  for i in range(len(descriptor1)):
    dist += math.fabs(descriptor1[i] - descriptor2[i])
  dist /= len(descriptor1)
  return dist


def read_pcd_from_file(file):
  np_pts = np.zeros(0)
  with open(file, 'r') as f:
    pts = []
    for line in f:
      one_pt = list(map(float, line[:-1].split(',')))
      pts.append(one_pt[:3])
    np_pts = np.array(pts)
  return np_pts


if __name__ == "__main__":
  ref_point = [-0.488, -0.6949, 0.1895]
  similiar_points = [[0.4691, -0.6959, -0.2416],
                     [0.2591, -0.7157, 0.4519],
                     [-0.2534, -0.7121, -0.4699]]
  
  different_points = [[0.0462, 0.1455, -0.02645],
                      [-0.3523, 0.2135, -0.08018],
                      [-0.1752, 0.1472, 0.2368],
                      [-0.0663, 0.1513, -0.3268],
                      [0.04542, -0.5794, 0.02739],
                      [0.3454, 0.2208, 0.02532],
                      [-0.05462, -0.2288, 0.02479],
                      [0.1766, 0.2027, 0.3181],
                      [0.2487, 0.1977, -0.2729]]
  
  
  
  np_pts = read_pcd_from_file("../data/stool_0091.txt")
  # feature_points = libpcl.keypointIss(np_pts, iss_salient_radius=0.3, iss_non_max_radius=0.3, \
  #     iss_gamma_21=0.5, iss_gamma_32=0.5, iss_min_neighbors=30, threads=8)
  print("--------------------------Test My FPFH--------------------------------")
  print("cal descriptors for two similiar points...")
  for i in range(len(similiar_points)):
    feature_points = np.array([ref_point, similiar_points[i]])
    descriptors = fpfh.fpfh(np_pts, feature_points, feature_radius=0.1)
    print("dist between two similiar points: ", dist_between_descriptor(list(descriptors[0]), list(descriptors[1])))
    # point_cloud_show(np_pts, feature_points)

  print("cal descriptors for two different points...")
  for i in range(len(different_points)):
    feature_points = np.array([ref_point, different_points[i]])
    descriptors = fpfh.fpfh(np_pts, feature_points, feature_radius=0.1)
    print("dist between two different points: ", dist_between_descriptor(list(descriptors[0]), list(descriptors[1])))
    # point_cloud_show(np_pts, feature_points)


  print("--------------------------Test PCL FPFH--------------------------------")
  print("cal descriptors for two similiar points...")
  for i in range(len(similiar_points)):
    feature_points = np.array([ref_point, similiar_points[i]])
    descriptors = libpcl.featureFPFH33(np_pts, feature_points, compute_normal_k = 20, feature_radius=0.5)
    print("dist between two similiar points: ", dist_between_descriptor(list(descriptors[0]), list(descriptors[1])))
    # point_cloud_show(np_pts, feature_points)

  print("cal descriptors for two different points...")
  for i in range(len(different_points)):
    feature_points = np.array([ref_point, different_points[i]])
    descriptors = libpcl.featureFPFH33(np_pts, feature_points, compute_normal_k = 20, feature_radius=0.5)
    print("dist between two different points: ", dist_between_descriptor(list(descriptors[0]), list(descriptors[1])))
    # point_cloud_show(np_pts, feature_points)
  

  print("--------------------------Test PCL SHOT--------------------------------")
  print("cal descriptors for two similiar points...")
  for i in range(len(similiar_points)):
    feature_points = np.array([ref_point, similiar_points[i]])
    descriptors = libpcl.featureSHOT352(np_pts, feature_points, feature_radius=0.7)
    print("dist between two similiar points: ", dist_between_descriptor(list(descriptors[0]), list(descriptors[1])))
    # point_cloud_show(np_pts, feature_points)

  print("cal descriptors for two different points...")
  for i in range(len(different_points)):
    feature_points = np.array([ref_point, different_points[i]])
    descriptors = libpcl.featureSHOT352(np_pts, feature_points, feature_radius=0.7)
    print("dist between two different points: ", dist_between_descriptor(list(descriptors[0]), list(descriptors[1])))
    # point_cloud_show(np_pts, feature_points)