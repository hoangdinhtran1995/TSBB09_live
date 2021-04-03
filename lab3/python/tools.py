import numpy as np


def camera_lookat(camera: np.array, target: np.array, up: np.array):
    """ Compute the world-to-camera transform, representing a camera
    looking at a target point.

    :param camera: Camera position in world space (3d vector)
    :param target: Target position in world space (3d vector)
    :param up:     Camera's up direction in world space (3d vector)
    :return: tuple of (R, t)
    """

    camera = camera.squeeze()
    target = target.squeeze()
    up = up.squeeze()

    up = up / np.linalg.norm(up)
    forward = target - camera
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    up = np.cross(right, forward)  # orthogonalize

    # Camera-to-world transform
    R = np.stack((right, up, forward), axis=1)
    t = target

    # World-to-camera transform
    t = -R.T @ t
    R = R.T

    return R, t



