import json

import cv2
import numpy as np
from numpy.linalg import svd, inv, cholesky, norm, lstsq
from scipy.optimize import root

def hom(x, axis=0):
    """ Convert the matrix to homogeneous coordinates by adding a
        new row (axis=0) or column (axis=1) of ones. """
    assert len(x.shape) == 2  # Expect a 2D matrix
    assert axis == 0 or axis == 1
    padding = ((0, 1), (0, 0)) if axis == 0 else ((0, 0), (0, 1))
    x = np.pad(x, padding, 'constant', constant_values=(1,))
    return x


def dhom(x, axis=0):
    """ Divide the matrix elements by the homogeneous coordinates in the
    bottom row (axis=0) or rightmost column (axis=1) """
    assert len(x.shape) == 2  # Expect a 2D matrix
    assert axis == 0 or axis == 1
    x = x[:-1, :] / x[None, -1, :] if axis == 0 else x[:, :-1] / x[:, None, -1]
    return x


def rodrigues(x):
    return cv2.Rodrigues(x)[0]


def normalization_matrix(x):
    """ Generate a Hartley normalization matrix.
    :param x: An array of 2D points shape = (2,N)
    :return: The matrix that normalizes the points,
    i.e subtracts the mean and divides by the standard deviation.
    """
    m = np.mean(x, axis=1)
    s = np.sqrt(2) / np.std(x, axis=1)

    N = [[s[0], 0, -s[0] * m[0]],
         [0, s[1], -s[1] * m[1]],
         [0, 0, 1]]
    return np.array(N)


def find_homography(model_points, image_points):
    """ Estimate a homography H such that image_points = H @ model_points
    :param model_points: np.array of 3D model points, shape (3, N) (3 rows, N columns)
                         In each column, the coordinates are ordered x,y,z
    :param image_points: np.array of 2D image pixel coordinates, shape (2, N) in the same order
                         as the model points.
    :return: The estimated homography, np.array shape (3,3)
    """
    A = model_points[:2]
    B = image_points

    Na = normalization_matrix(A)
    Nb = normalization_matrix(B)

    A = dhom(Na @ hom(A))
    B = dhom(Nb @ hom(B))

    M = []

    for k in range(A.shape[1]):
        x, y = A[:, k]
        u, v = B[:, k]
        M.append((x, y, 1, 0, 0, 0, -u*x, -u*y, -u))
        M.append((0, 0, 0, x, y, 1, -v*x, -v*y, -v))
    M = np.stack(M)

    u, s, vt = svd(M)
    h = vt[-1, :]
    H = h.reshape(3, 3)

    H = inv(Nb) @ H @ Na
    H /= H[2, 2]

    # Test:

    # H2, _ = cv2.findHomography(model_points.T, image_points.T)
    # H2 /= H2[2,2]
    # print(H[0,0], H2[0,0])
    # print(np.abs(image_points - dhom(H @ hom(model_points[:2]))).mean(),
    #       np.abs(image_points - dhom(H2 @ hom(model_points[:2]))).mean())
    # print(H)
    # print(H2)
    # print("------------")

    return H


def refine_homography(H, model_points, image_points):
    """ Refine a homography with nonlinear optimization
    :param H:             Homography to optimize, np.array shape (3,3)
    :param model_points:  np.array of 3D model points, shape (3, N) (3 rows, N columns)
                          In each column, the coordinates are ordered x,y,z
    :param image_points:  np.array of 2D image pixel coordinates, shape (2, N)
                          in the same order as the model points.
    :return: The refined homography, np.array shape (3,3)
    """

    model_points = hom(model_points[:2])  # As homogeneous 2D coordinates

    def residual_fn(x):
        H = x.reshape((3,3))
        residuals = image_points - dhom(H @ model_points)
        residuals = residuals.T.flatten()
        return residuals

    result = root(residual_fn, H.flatten(), method='lm')
    H = result.x.reshape((3, 3))
    if not result.success:
        print("Homography refinement failed:", result.message)

    return H / H[2,2]


def get_camera_intrinsics(homographies):
    """ Estimate the camera intrinsic matrix A from a set of homographies.
    :param homographies: list of homographies, each a (3,3) np.array
    :return: The intrinsic matrix A, a (3,3) np.array
    """

    def vectorize(H, i, j):

        v = [H[0, i] * H[0, j],
             H[0, i] * H[1, j] + H[1, i] * H[0, j],
             H[1, i] * H[1, j],
             H[2, i] * H[0, j] + H[0, i] * H[2, j],
             H[2, i] * H[1, j] + H[1, i] * H[2, j],
             H[2, i] * H[2, j]]

        return np.array(v)

    V = []
    for H in homographies:
        V.append(vectorize(H, 0, 1))
        V.append(vectorize(H, 0, 0) - vectorize(H, 1, 1))
    V = np.stack(V)

    u, s, vt = svd(V)
    b = vt[-1, :]

    def decompose_zhang(b):
        b11, b12, b22, b13, b23, b33 = b

        v = (b12*b13 - b11*b23) / (b11*b22 - b12*b12)
        L = b33 - (b13*b13 + v*(b12*b13 - b11*b23)) / b11
        a = np.sqrt(L / b11)
        b = np.sqrt(L * b11 / (b11*b22 - b12*b12))
        g = -b12*a*a*b / L
        u = g*v/b - b13*a*a/L

        A = np.array([[a, g, u],
                      [0, b, v],
                      [0, 0, 1]])
        return A

    def decompose_cholesky(b):
        b11, b12, b22, b13, b23, b33 = b

        B = np.array([[b11, b12, b13],
                      [b12, b22, b23],
                      [b13, b23, b33]])
        if B[0,0] < 0 or B[1,1] < 0 or B[2,2] < 0:
            B = -B
        L = cholesky(B)
        A = inv(L).T * L[2, 2]

        return A

    A1 = decompose_cholesky(b)
    A2 = decompose_zhang(b)
    # print(A1)
    # print(A2)

    return A1


def get_camera_extrinsics(A, Hs):
    """ Decompose a set of homographies into camera rotations and translations.
    :param A:  The intrinsic matrix A, a (3,3) np.array
    :param Hs: A list of homographies H such that image_points = H @ model_points
    :return:   Rs, ts - lists of (3,3) rotation matrices and (3,1) translation vectors
    """

    Rs = []
    ts = []
    Ainv = inv(A)

    for H in Hs:

        h1, h2, h3 = np.hsplit(H, 3)

        r1 = Ainv @ h1
        r2 = Ainv @ h2
        L = 1 / norm(r1)
        r1 = L * r1.squeeze()
        r2 = L * r2.squeeze()
        r3 = np.cross(r1, r2)
        t = L * Ainv @ h3
        R = np.stack((r1, r2, r3), axis=1)

        Rs.append(R)
        ts.append(t)

    return Rs, ts


def estimate_radial_lens_distortion(A, Rs, ts, model_points, image_points):
    """ Estimate radial distortion parameters k1 and k2.
    :param A:  Camera intrinsic matrix
    :param Rs: A list of K camera rotation matrices (one list item per camera)
    :param ts: A list of K camera translation vectors, each shaped (3,1)
    :param model_points: list of K model points matrices
    :param image_points: list of K image points matrices
    :return: The radial distortion coefficients k1 and k2
    """

    npoints = len(model_points[0])

    assert len(ts) == len(Rs)
    assert len(model_points) == len(Rs)
    assert len(image_points) == len(Rs)

    D = []
    d = []
    u0, v0 = A[:2, 2]

    for R, t, X, y in zip(Rs, ts, model_points, image_points):

        xn = R @ X + t
        x = dhom(A @ xn).T
        xn = dhom(xn).T

        for i in range(npoints):

            u, v = x[i]

            r2 = (xn[i] ** 2).sum()
            r4 = r2*r2

            D.append(((u - u0) * r2, (u - u0) * r4))
            D.append(((v - v0) * r2, (v - v0) * r4))

            d.append((y[0, i] - u))
            d.append((y[1, i] - v))

    D = np.stack(D)
    d = np.stack(d)

    k1, k2 = lstsq(D, d, rcond=None)[0]
    return k1, k2


def refine_calibration(A, k1, k2, camera_rotations, camera_translations, model_points, image_points):
    """ Refine a calibration with nonlinear optimization.
    :param A:                   Intrinsic camera matrix, shape (3, 3)
    :param k1:                  Second order radial distortion coefficients
    :param k2:                  Fourth order radial distortion coefficients
    :param camera_rotations:    list of camera rotation matrices, shape (3, 3)
    :param camera_translations: list of camera translation vectors, shape (3, 1)
    :param model_points:        list of np.array objects of 3D model points, shape (3, N) (3 rows, N columns)
                                In each column, the coordinates are ordered x,y,z
    :param image_points:        list of np.array objects of 2D image pixel coordinates, shape (2, N) in the same order
                                as the model points.
    :return:  Refined A, k1, k2, camera_rotations, camera_translations
    """

    def residual_fn(x):

        residuals = []

        fx, ga, u0, fy, v0, k1, k2 = x[:7]

        for i, extrinsics in enumerate(x[7:].reshape((-1, 6))):

            R = rodrigues(extrinsics[:3])
            t = np.atleast_2d(extrinsics[3:]).T

            x = R @ model_points[i] + t              # Transform world to camera coordinates
            y = x[:2] / x[2]                         #
            r2 = (y * y).sum(axis=0)                 # Add distortion
            y = y * (1.0 + k1 * r2 + k2 * r2 * r2)   #
            u = fx * y[0] + ga * y[1] + u0           # Transform to pixel coordinates
            v = fy * y[1] + v0                       #

            ru = (u - image_points[i][0]).flatten()  # Compute residuals
            rv = (v - image_points[i][1]).flatten()  #

            residuals.extend(ru)
            residuals.extend(rv)

        return np.array(residuals)

    # Flatten the parameters for the optimizer

    fx, ga, u0, _, fy, v0 = A[:2, :].flatten()

    x = [fx, ga, u0, fy, v0, k1, k2]
    for R, t in zip(camera_rotations, camera_translations):
        x.extend(rodrigues(R).flatten())
        x.extend(t.flatten())

    # Optimize

    result = root(residual_fn, np.array(x), method='lm')
    if not result.success:
        print("Calibration refinement failed:", result.message)

    # Recover the parameters

    fx, ga, u0, fy, v0, k1, k2 = result.x[:7]
    A = np.array([[fx, ga, u0], [0, fy, v0], [0, 0, 1]])

    cameras = result.x[7:].reshape((-1, 6))
    camera_rotations = [rodrigues(rr) for rr in cameras[:, :3]]
    camera_translations = [np.atleast_2d(t).T for t in cameras[:, 3:]]

    return A, k1, k2, camera_rotations, camera_translations


def project_points(A, d, R, t, points):
    """  Project a set of 3D points into a camera.
    :param A:       Intrinsic matrix (3,3)
    :param d:      Distortion, coefficients
    :param R:       World-to-camera rotation matrix (3,3)
    :param t:       World-to-camera translation vector (3,1).
    :param points:  N 3D points to project. Numpy array, shape (3,N)
    :return: Reprojected (2,N) image 2D points
    """

    if isinstance(d, np.ndarray):
        d = tuple(d.flatten())
    k1, k2, p1, p2, k3 = d[0], d[1], 0, 0, 0
    if len(d) >= 4:
        p1, p2 = d[2], d[3]
    if len(d) >= 5:
        k3 = d[4]

    xn = dhom(R @ points + t)
    r2 = (xn ** 2).sum(axis=0)
    rd = 1.0 + k1 * r2 + k2 * r2*r2 + k3 * r2*r2*r2
    td = np.stack((2*p1*xn[0]*xn[1] + p2*(r2 + 2*xn[0]*xn[0]),
                   p1*(r2 + 2*xn[1]*xn[1]) + 2*p2*xn[0]*xn[1]))

    x = A @ hom(xn * rd + td)
    x = dhom(x)

    return x

def reprojection_errors(A, d, Rs, ts, model_points, image_points):
    """
    :param A:             Intrinsic camera matrix, shape (3, 3)
    :param d:             Distortion coefficients
    :param Rs:            List of camera rotation matrices
    :param ts:            List of camera translation vectors, shape (3, 1)
    :param model_points:  List of matrices of 3D model points, each shaped (3, N)
    :                     In each column, the coordinates are ordered x,y,z
    :param image_points:  List of np.array objects of 2D image pixel coordinates,
                          each shaped (2, N), in the same order as the model points.
    :returns: List of (1,N) per-camera reprojection errors.
    """
    rpes = []
    for i, (R, t, mdl_pts, im_pts) in enumerate(zip(Rs, ts, model_points, image_points)):
        uv = project_points(A, d, R, t, mdl_pts.T).T
        rpe = np.linalg.norm((uv - im_pts), axis=0)
        rpes.append(rpe)
    return rpes

def plot_reprojection_errors(im_size, A, d, Rs, ts, model_points, image_points):
    """ Plot reprojection errors in an image
    :param im_size:       Image (height, width)
    :param A:             Intrinsic camera matrix, shape (3, 3)
    :param d:             Distortion coefficients
    :param Rs:            List of camera rotation matrices
    :param ts:            List of camera translation vectors, shape (3, 1)
    :param model_points:  List of matrices of 3D model points, each shaped (3, N)
    :                     In each column, the coordinates are ordered x,y,z
    :param image_points:  List of np.array objects of 2D image pixel coordinates,
                          each shaped (2, N), in the same order as the model points.
    :returns: Image with the reprojection errors plotted.
    """

    w, h = im_size
    im = np.ones((h, w, 3), dtype=np.uint8) * 30  # Dark grey

    for i, (R, t, mdl_pts, im_pts) in enumerate(zip(Rs, ts, model_points, image_points)):

        y1 = project_points(A, d, R, t, mdl_pts.T)

        y = y1.round().astype(int)
        im_pts = im_pts.round().astype(int)

        color = tuple(map(int, tuple(np.random.randint(0, 255, (3,)))))

        for pt1, pt2 in zip(y.T, im_pts):
            # t = np.abs(pt1.astype(float))
            # if t[0] > 10000 or t[1] > 10000:
            #     continue
            cv2.line(im, tuple(pt1), tuple(pt2), color, thickness=2, lineType=cv2.LINE_AA)

    return im


def load_calibration_input(data_path):
    """ Load chessboard model points, detected image points and image dimensions
    % from all .npz files in the given data_path"""

    model_points = []
    image_points = []
    im_size = None
    cb_inner_corners = None
    cb_tile_size = None

    files = list(sorted(data_path.glob("*.npz")))
    for file in files:
        file = np.load(file)
        if im_size is None:
            im_size = tuple(file['im_size'])
            cb_inner_corners = tuple(file['cb_inner_corners'])
            cb_tile_size = file['cb_tile_size']
        image_points.append(file['image_points'])
        model_points.append(file['model_points'])

    return cb_inner_corners, cb_tile_size, im_size, model_points, image_points


def load_calibration(filename):
    """ Load a set of camera calibration parameters.
    :param filename:  .json file to load
    :returns:  A, d, Rs, ts, rpe
               A is the camera intrinsic matrix
               d is a list of distortion model coefficients
               Rs a cell array of camera rotation matrices,
               ts a cell array of camera translation vectors
               rpe is the average reprojection error."""

    cal = json.load(open(filename))
    A = np.array(cal['A'])
    d = np.array(cal['d'])
    Rs = list(np.array(cal['Rs']))
    ts = list(np.array(cal['ts']))
    rpe = cal['rpe']

    return A, d, Rs, ts, rpe
