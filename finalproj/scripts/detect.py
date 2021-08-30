#!/opt/conda/envs/kitti-detection-pipeline/bin/python

# detect.py
#     1. read Velodyne point cloud measurements
#     2. perform ground and object segmentation on point cloud
#     3. predict object category using classification network

import os
import glob
import argparse
import sys
import progressbar
import datetime

sys.path.insert(0, './')
# disable TF log display:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# prediction:
import numpy as np
import scipy.spatial

from scripts.extract import read_velodyne_bin, read_calib, segment_ground_and_objects

# open3d:
import open3d as o3d
import torch
import torch.nn.functional as F
from pointnet2.data.resampled_dataset import KITTIPCDClsDataset_Wrapper

from scripts.transform_coords_utils import transform_to_cam, transform_to_pixel

def get_orientation_in_camera_frame(X_cam_centered):
    """
    Get object orientation using PCA
    """
    # keep only x-z:
    X_cam_centered = X_cam_centered[:, [0, 2]]

    H = np.cov(X_cam_centered, rowvar=False, bias=True)

    # get eigen pairs:
    eigenvalues, eigenvectors = np.linalg.eig(H)

    idx_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # orientation as arctan2(-z, x):
    return np.arctan2(-eigenvectors[0][1], eigenvectors[0][0])

def to_kitti_eval_format(segmented_objects, object_ids, param, predictions, decoder):
    """
    Write prediction result as KITTI evaluation format
    Parameters
    ----------
    segmented_objects: open3d.geometry.PointCloud
        Point cloud of segmented objects
    object_ids: numpy.ndarray
        Object IDs as numpy.ndarray
    predictions:
        Object Predictions
    Returns
    ----------
    """
    # parse params:
    points = np.asarray(segmented_objects.points)

    # initialize KITTI label:
    label = {
        'type': [],
        'left': [], 'top': [], 'right': [], 'bottom': [],
        'height': [], 'width': [], 'length': [],
        'cx': [], 'cy': [], 'cz': [], 
        'ry': [], 
        # between 0 and 100:
        'score': []
    }
    formatter = lambda x: f'{x:.2f}'
    kitti_type = {
        'vehicle': 'Car',
        'pedestrian': 'Pedestrian',
        'cyclist': 'Cyclist',
        'misc': 'Misc'
    }

    for class_id in predictions:
        # get color
        class_name = decoder[class_id]

        if (class_name == 'misc'):
            continue
        
        # get KITTI type:
        class_name = kitti_type[class_name]

        # show instances:
        for object_id in predictions[class_id]:
            # set object type:
            label['type'].append(class_name)

            # transform to camera frame:
            X_velo = points[object_ids == object_id]
            X_cam = transform_to_cam(X_velo, param)

            # transform to pixel frame:
            X_pixel = transform_to_pixel(X_cam, param)

            # set 2D bounding box:
            top_left = X_pixel.min(axis = 0)
            bottom_right = X_pixel.max(axis = 0)

            label['left'].append(formatter(top_left[0]))
            label['top'].append(formatter(top_left[1]))
            label['right'].append(formatter(bottom_right[0]))
            label['bottom'].append(formatter(bottom_right[1]))

            # set object location:
            c_center = X_cam.mean(axis = 0)

            label['cx'].append(formatter(c_center[0]))
            label['cy'].append(formatter(c_center[1]))
            label['cz'].append(formatter(c_center[2]))

            # set object orientation:
            X_cam_centered = X_cam - c_center
            orientation = get_orientation_in_camera_frame(X_cam_centered)
            label['ry'].append(formatter(orientation))

            # project to object frame:
            cos_ry = np.cos(-orientation)
            sin_ry = np.sin(-orientation)

            R_obj_to_cam = np.asarray(
                [
                    [ cos_ry, 0.0, sin_ry],
                    [    0.0, 1.0,    0.0],
                    [-sin_ry, 0.0, cos_ry]
                ]
            )

            X_obj = np.dot(
                R_obj_to_cam.T, (X_cam_centered).T
            ).T

            # set object dimension:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                X_obj
            )
            bounding_box = object_pcd.get_axis_aligned_bounding_box()
            extent = bounding_box.get_extent()

            # height along y-axis:
            label['height'].append(formatter(extent[1]))
            # width along x-axis:
            label['width'].append(formatter(extent[0]))
            # length along z-axis:
            label['length'].append(formatter(extent[2]))

            # set confidence:
            confidence = 100.0 * predictions[class_id][object_id]
            label['score'].append(formatter(confidence))
    
    # format as pandas dataframe:
    label = pd.DataFrame.from_dict(
        label
    )
    
    # set value for unavailable fields:
    label['truncated'] = -1
    label['occluded'] = -1
    # don't evaluate AOS:
    label['alpha'] = -10

    # set column order:
    label = label[
        [
            'type',
            'truncated',
            'occluded',
            'alpha',
            'left', 'top', 'right', 'bottom',
            'height', 'width', 'length',
            'cx', 'cy', 'cz', 'ry',
            'score'
        ]
    ]

    return label


def get_bounding_boxes(segmented_objects, object_ids, predictions, ind2label):
    """
    Draw bounding boxes for surrounding objects according to classification result
        - red for pedestrian
        - blue for cyclist
        - green for vehicle
    Parameters
    ----------
    segmented_objects: open3d.geometry.PointCloud
        Point cloud of segmented objects
    object_ids: numpy.ndarray
        Object IDs as numpy.ndarray
    predictions:
        Object Predictions
    Returns
    ----------
    """
    # parse params:
    points = np.asarray(segmented_objects.points)

    # color cookbook:
    color = {
        # pedestrian as red:
        'pedestrian': np.asarray([0.5, 0.0, 0.0]),
        # cyclist as blue:
        'cyclist': np.asarray([0.0, 0.0, 0.5]),
        # vehicle as green:
        'vehicle': np.asarray([0.0, 0.5, 0.0]),
    }

    bounding_boxes = []
    for class_id in predictions:
        # get color
        class_name = ind2label[class_id]

        if (class_name == 'misc'):
            continue

        class_color = color[class_name]
        # show instances:
        for object_id in predictions[class_id]:
            # create point cloud:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                points[object_ids == object_id]
            )
            # create bounding box:
            bounding_box = object_pcd.get_axis_aligned_bounding_box()

            # set color according to confidence:
            confidence = predictions[class_id][object_id]
            bounding_box.color = tuple(
                class_color + (1.0 - confidence)*class_color
            )

            # update:
            bounding_boxes.append(bounding_box)
    
    return bounding_boxes

def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        print("==> Done")
    else:
        raise FileNotFoundError

    return epoch

def preprocess(
    segmented_objects, object_ids,
    config
):
    """
    Preprocess for classification network
    """
    # parse config:
    points = np.asarray(segmented_objects.points)
    normals = np.asarray(segmented_objects.normals)
    num_objects = max(object_ids) + 1

    # result:
    X = []
    y = []
    for object_id in range(num_objects):
        # 1. only keep object with enough number of observations:
        if ((object_ids == object_id).sum() <= 4):
            continue
        
        # 2. only keep object within max radius distance:
        object_center = np.mean(points[object_ids == object_id], axis=0)[:2]
        if (np.sqrt((object_center*object_center).sum()) > config['max_radius_distance']):
            continue
        
        # 3. resample:
        points_ = np.copy(points[object_ids == object_id])
        normals_ = np.copy(normals[object_ids == object_id])
        N, _ = points_.shape

        weights = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(points_, 'euclidean')
        ).mean(axis = 0)
        weights /= weights.sum()
        
        idx = np.random.choice(
            np.arange(N), 
            size = (config['num_sample_points'], ), replace=True if config['num_sample_points'] > N else False,
            p = weights
        )

        # 4. translate to zero-mean:
        points_processed, normals_processed = points_[idx], normals_[idx]
        points_processed -= points_.mean(axis = 0)

        # format as numpy.ndarray:
        X.append(
            np.hstack(
                (points_processed, normals_processed)
            )
        )
        y.append(object_id)

    
    # X = np.asarray(X)
    # y = np.asarray(y)

    # pad to batch size:
    N = len(y)
    if (N % config['batch_size'] != 0):
        num_repeat = config['batch_size'] - N % config['batch_size']

        for _ in range(num_repeat):
            X.append(X[0]) 
            y.append(y[0])

        # X = np.vstack(
        #     (
        #         X, 
        #         np.repeat(
        #             X[0], num_repeat, axis=0
        #         ).reshape(
        #             (-1, config['num_sample_points'], 6)
        #         )
        #     )
        # )
        # y = np.hstack(
        #     (y, np.repeat(y[0], num_repeat))
        # )

    curN = len(y)
    dataset = []
    for i in range(curN):
        dataset.append((X[i], y[i]))

    return dataset, N


def predict(segmented_objects, object_ids, model, config):
    """ 
    Load classification network and predict surrounding object category
    Parameters
    ----------
    config: dict 
        Model training configuration
    """
    # prepare data:
    dataset, N = preprocess(segmented_objects, object_ids, config)

    # make predictions:
    predictions = {
        class_id: {} for class_id in range(config['num_classes'])
    }
    num_predicted = 0

    for pts, label in dataset:
        pts, labels = np.asarray(pts), np.asarray(labels)
        pts = torch.from_numpy(pts).cuda(non_blocking=True).float()

        with torch.no_grad():
            # predict:
            prob_preds = model(pts)
            ids = label
        
        prob_score = F.softmax(prob_preds).cpu().numpy()
        # add to prediction:
        for (object_id, class_id, confidence) in zip(
            # object ID: 
            ids, 
            # category:
            np.argmax(prob_score, axis=1),
            # confidence:
            np.max(prob_score, axis=1)
        ):
            predictions[class_id][object_id] = confidence
            num_predicted += 1
            
            # skip padded instances:
            if (num_predicted == N):
                break

    return predictions

def detect(
    dataset_dir, index,
    max_radius_distance, num_sample_points,
    debug_mode
):
    # 0. generate I/O paths:
    input_velodyne = os.path.join(dataset_dir, 'velodyne', f'{index:06d}.bin')
    input_params = os.path.join(dataset_dir, 'calib', f'{index:06d}.txt')
    output_label = os.path.join(dataset_dir, 'shenlan_pipeline_pred_2', 'data', f'{index:06d}.txt')

    # 1. read Velodyne measurements and calib params:
    point_cloud = read_velodyne_bin(input_velodyne)
    param = read_calib(input_params)

    # 2. segment ground and surrounding objects -- here discard intensity channel:
    segmented_ground, segmented_objects, object_ids = segment_ground_and_objects(point_cloud[:, 0:3])

    # 3. predict object category using classification network:
    config = {
        # preprocess:
        'max_radius_distance': max_radius_distance,
        'num_sample_points': num_sample_points,
        # predict:
        'msg' : True,
        'batch_size' : 16,
        'num_classes' : 4,
        'batch_normalization' : False,
        'checkpoint_path' : 'logs/msg_1/model/weights.ckpt',
    }
    model = load_model(config)
    predictions = predict(segmented_objects, object_ids, model, config)

    with open(os.path.join('./data/resampled_KITTI/object_names.txt')) as f:
        file_labels = [l.strip() for l in f.readlines()]
        
        label2ind = {label: ind for ind, label in enumerate(file_labels)}
        ind2label = {ind: label for ind, label in enumerate(file_labels)}

    # debug mode:
    if (debug_mode):
        # print detection results:
        for class_id in predictions:
            # show category:
            print(f'[{ind2label[class_id]}]')
            # show instances:
            for object_id in predictions[class_id]:
                print(f'\t[Object ID]: {object_id}, confidence {predictions[class_id][object_id]:.2f}')

        # visualize:
        bounding_boxes = get_bounding_boxes(
            segmented_objects, object_ids, 
            predictions, ind2label
        )
        o3d.visualization.draw_geometries(
            [segmented_ground, segmented_objects] + bounding_boxes
        )
    
    # 4. format output for KITTI offline evaluation tool:
    label = to_kitti_eval_format(
        segmented_objects, object_ids, param,
        predictions, ind2label
    )
    label.to_csv(output_label, sep=' ', header=False, index=False)

def get_arguments():
    """ 
    Get command-line arguments
    """
    # init parser:
    parser = argparse.ArgumentParser("Perform two-stage object detection on KITTI dataset.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path.",
        required=True, type=str
    )

    optional.add_argument(
        "-d", dest="debug_mode", help="When enabled, visualize the result. Defaults to False. \n",
        required=False, type=bool, default=False
    )
    optional.add_argument(
        "-r", dest="max_radius_distance", help="Maximum radius distance between object and Velodyne lidar. \nUsed for ROI definition. Defaults to 25.0. \nONLY used in 'generate' mode.",
        required=False, type=float, default=25.0
    )
    optional.add_argument(
        "-n", dest="num_sample_points", help="Number of sample points to keep for each object. \nDefaults to 64. \nONLY used in 'generate' mode.",
        required=False, type=int, default=64
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == "__main__":
    # parse command line arguments
    args = get_arguments()

    for label in progressbar.progressbar(
        glob.glob(
            os.path.join(args.input, 'shenlan_pipeline_label_2', '*.txt')
        )
    ):
        # get index:
        index = int(
            os.path.splitext(
                os.path.basename(label)
            )[0]
        )

        # perform object detection:
        detect(
            args.input, index,
            args.max_radius_distance, args.num_sample_points,
            args.debug_mode
        )