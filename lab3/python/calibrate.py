from pathlib import Path
import json
import numpy as np
import scipy.io
import cv2

# from calibrate_zhang_finished import calibrate_zhang
# from calibrate_zhang import calibrate_zhang

def calibrate_opencv(model_points, image_points, im_size):
    """ Calibrate with OpenCV's method
    :param model_points:    np.array of 3D model points, shape (3, N) (3 rows, N columns)
                            In each column, the coordinates are ordered x,y,z
    :param image_points:    np.array of 2D image pixel coordinates, shape (2, N) in the same order
                            as the model points.
    :param im_size:         tuple (width, height) in pixels
    :return: A, d, Rs, ts, rpe   tuple of camera intrinsics, distortion coefficients, camera rotation matrices
                                 camera translation vectors and the average reprojection error [pixels]
    """

    # OpenCV requires (N,2) or (N,3) shapes
    image_points = [x.T for x in image_points]
    model_points = [x.T for x in model_points]

    flags = 0
    # Enable these to disable all but the distortion coefficients used in Zhang's method:
    # flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3

    rpe, A, d, rs, ts = cv2.calibrateCamera(model_points, image_points, im_size, None, None, flags=flags)
    Rs = [cv2.Rodrigues(r)[0] for r in rs]  # Convert Rodrigues rotation to rotation matrices

    return A, d, Rs, ts, rpe


if __name__ == '__main__':

    # Path to data from find_points.py
    data_path = Path("/home/hoatr725/Documents/TSBB09/Lab 3/data/output").expanduser()

    # Load data

    model_points = []
    image_points = []
    im_size = None

    files = list(sorted(data_path.glob("*.npz")))
    for file in files:
        file = np.load(file)
        if im_size is None:
            im_size = tuple(file['im_size'])
        image_points.append(file['image_points'].T)  # List of np.arrays (one per image), each shaped (N,2)
        model_points.append(file['model_points'].T)  # Like image_points but shaped (N,3)

    print("--- OpenCV's calibration ---\n")

    A, d, Rs, ts, rpe = calibrate_opencv(model_points, image_points, im_size)

    print("Intrinsic matrix:")
    print(np.array_str(A, precision=3, suppress_small=True))
    print("Distortion coefficients:")
    print(np.array_str(d, precision=3, suppress_small=True))
    print("Mean reprojection error: %.2f pixels" % rpe)
    print("\n\n")

    # Save the calibration for later use

    cal_data = dict(A=A.tolist(), d=d.tolist(), im_size=tuple(map(int, im_size)),
                    Rs=np.stack(Rs).tolist(), ts=np.stack(ts).tolist(), rpe=rpe)
    json.dump(cal_data, open(data_path / "calibration_cv.json", "w"))
    scipy.io.savemat(data_path / (data_path / "calibration_cv.mat"), cal_data)

    # TODO: Implement this method
    # print("--- Zhang's calibration ---\n")
    #
    # A, d, Rs, ts, rpe = calibrate_zhang(model_points, image_points)
    #
    # print("Intrinsic matrix:")
    # print(np.array_str(A, precision=3, suppress_small=True))
    # print("Distortion coefficients:")
    # print(np.array_str(d, precision=3, suppress_small=True))
    # print("Mean reprojection error: %.2f pixels" % rpe)

    # Save the calibration for later use

    # cal_data = dict(A=A.tolist(), d=d.tolist(), im_size=tuple(map(int, im_size)),
    #                 Rs=np.stack(Rs).tolist(), ts=np.stack(ts).tolist(), rpe=rpe)
    # json.dump(cal_data, open(data_path / "calibration_zhang.json", "w"))
    # scipy.io.savemat(data_path / (data_path / "calibration_zhang.mat"), cal_data)

