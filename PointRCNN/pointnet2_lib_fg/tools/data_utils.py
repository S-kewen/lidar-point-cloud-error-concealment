import sys
import glob
import os
import numpy as np
import math

def in_hull_count(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    results = hull.find_simplex(p) >= 0
    return np.sum(results!=0)

def get_hull_points(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    results = hull.find_simplex(p[:, :3]) >= 0
    return p[np.where(results!=0)]

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p[:, :3])

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(calibs, h, w, l, x, y, z, ry):
    ry = roty(ry)
    x_corners = [l / 2, l / 2, -l / 2, -
                    l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2,
                    w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(ry, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    ro = calibs["R0_rect"]
    ro = np.reshape(ro, [3, 3])
    c2v = inverse_rigid_trans(np.reshape(calibs["Tr_velo_to_cam"], [3, 4]))
    return project_rect_to_velo(np.transpose(corners_3d), ro, c2v)

def project_rect_to_velo(pts_3d_rect, ro, c2v):
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, ro)
    return project_ref_to_velo(pts_3d_ref, c2v)

def project_rect_to_ref(pts_3d_rect, ro):
    """ Input and Output are nx3 points """
    return np.transpose(np.dot(np.linalg.inv(ro), np.transpose(pts_3d_rect)))

def project_ref_to_velo(pts_3d_ref, c2v):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(c2v))

def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def read_calib_file(filepath):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data