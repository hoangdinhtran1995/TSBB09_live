import numpy as np
from functions import *


def calibrate_zhang(model_points, image_points):

    """ Calibrate with Zhang's method
    :param model_points:    np.array of 3D model points, shape (3, N) (3 rows, N columns)
                            In each column, the coordinates are ordered x,y,z
    :param image_points:    np.array of 2D image pixel coordinates, shape (2, N) in the same order
                            as the model points.
    :return: A, d, Rs, ts, rpe   tuple of camera intrinsics, distortion coefficients, camera rotation matrices
                                 camera translation vectors and the average reprojection error [pixels]
    """

    # TODO: Your code here

    # Placeholder output. Remove this:
    A = np.eye(3,3)
    d = np.zeros((1,5))
    Rs = [np.eye(3,3)] * len(model_points)
    ts = [np.zeros(3,1)] * len(model_points)
    rpe = 0.0

    return A, d, Rs, ts, rpe
