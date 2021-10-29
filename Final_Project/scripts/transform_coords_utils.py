import numpy as np


def transform_to_cam(X_velo, param):
    '''
    Transform point from Velodyne frame to camera frame
    Parameters
    ----------
    X_velo: numpy.ndarray
        points in velodyne frame
    param: dict
        Vehicle parameters
    Returns
    ----------
    X_cam: numpy.ndarray
        Points in camera frame
    '''
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # project to unrectified camera frame:
    X_cam = np.dot(
        R_velo_to_cam, X_velo.T
    ).T + t_velo_to_cam

    # rectify:
    X_cam = np.dot(
       R0_rect, X_cam.T
    ).T

    return X_cam

def transform_to_pixel(X_cam, param):
    '''
    Transform point from camera frame to pixel frame
    Parameters
    ----------
    X_cam: numpy.ndarray
        points in camera frame
    param: dict
        Vehicle parameters
    Returns
    ----------
    X_pixel: numpy.ndarray
        Points in pixel frame
    '''
    # get params:
    K, b = param['P2'][:,0:3], param['P2'][:,3]

    # project to pixel frame:
    X_pixel = np.dot(
        K, X_cam.T
    ).T + b

    # rectify:
    X_pixel = (X_pixel[:, :2].T / X_pixel[:, 2]).T

    return X_pixel