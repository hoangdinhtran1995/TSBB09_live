from pathlib import Path
import numpy as np
import scipy.io
import cv2


def find_points(rgb_image, inner_points):
    """ Detect chessboard corners in an image.
    :param rgb_image:  input image
    :param inner_points: (w, h) - chessboard inner corners, columns and rows
    :return: found, points.
    On success, found is True, and points is an (h*w,2) array of (x,y) image pixel coordinates
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    found, points = cv2.findChessboardCorners(gray, inner_points)
    if found:
        term_crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        points = cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), term_crit)
        return found, points.squeeze(1)
    return False, None


def get_calibration_object_points(inner_points, tile_size):
    """ Generate chessboard points.
    :param inner_points: (w, h) - chessboard inner corners, columns and rows
    :param tile_size: chessboard tile size [meter]
    :return: Array of chessboard points shape (h*w, 3), ordered left-to-right, top-to-bottom.
    This ordering must match the output of find_points()
    """
    w, h = inner_points
    obj_points = np.empty((h*w, 3), np.float32)

    i = 0
    for y in range(h):
        for x in range(w):
            obj_points[i] = (x * tile_size, y * tile_size, 0.0)
            i += 1

    return obj_points


if __name__ == '__main__':

    # Input directory to read images from
    images_path = Path("/courses/TSBB09/CameraCalibration2/data/input").expanduser()
    # Source image file name pattern
    file_pattern = "*.jpg"
    # Output directory for saving visualizations and results
    data_path = Path("/home/hoatr725/Documents/TSBB09/Lab 3/data/output").expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    # Chessboard inner corners, (columns, rows)
    cb_inner_corners = (6, 8)
    # Chessboard tile size [meter]
    cb_tile_size = 0.03475

    im_size = None

    mdl_pts = get_calibration_object_points(cb_inner_corners, cb_tile_size)
    files = list(sorted(images_path.glob(file_pattern)))
    for i, fname in enumerate(files):

        print("Detecting corners in %s (%d of %d)" % (fname.name, i+1, len(files)))

        # Load image

        im = cv2.imread(str(fname))
        if im is None:
            print("Could not read %s." % fname.name)
            continue

        if im_size is None:
            im_size = im.shape[:2]
        else:
            assert im.shape[:2] == im_size

        # Find corners

        found, im_pts = find_points(im, cb_inner_corners)
        if not found:
            print("Skipping %s - could not find all corners." % fname.name)
            continue

        # Visualize and save

        vis = cv2.drawChessboardCorners(im, cb_inner_corners, im_pts, found)
        cv2.imwrite(str(data_path / (fname.stem + ".jpg")), vis)
        data = dict(cb_inner_corners=cb_inner_corners, cb_tile_size=cb_tile_size,
                    im_size=im_size[::-1], image_points=im_pts, model_points=mdl_pts)
        np.savez_compressed(data_path / (fname.stem + ".npz"), **data)
        scipy.io.savemat(data_path / (fname.stem + ".mat"), data)
