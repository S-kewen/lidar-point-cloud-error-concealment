import time
import open3d as o3d
import copy
import numpy as np
from matplotlib import cm
import math
import time


class O3dUtil(object):
    def __init__(self):
        pass

    def format_ply_by_json_list(file_name, json_list):
        points_i = []
        points_xyz = []
        with open(file_name) as pc:
            lines = pc.readlines()
            for i in range(len(lines)):
                coor = lines[i].split()
                if len(coor) < 4:
                    continue
                coor = [float(item) for item in coor]
                if coor[0] == 0.0:
                    if coor[1] > 0:
                        azimuth = 90.0
                    else:
                        azimuth = -90.0
                else:
                    azimuth = round(math.atan(coor[1] / coor[0]) / math.pi * 180, 4)
                if coor[0] < 0:
                    azimuth = azimuth + 180
                if azimuth < 0:
                    azimuth = azimuth + 360
                for json in json_list:
                    if int(json['segment']) * 4.8 <= azimuth and (int(json['segment']) + 1) * 4.8 > azimuth:
                        xyz = [0.0, 0.0, 0.0]
                        points_i.append(coor[-1])
                        for j in range(3):
                            xyz[j] = coor[j]
                        xyz[0] = -xyz[0]
                        points_xyz.append(xyz)
        return points_xyz, points_i

    def get_point_color(points_i):
        VIRIDIS = np.array(cm.get_cmap('plasma').colors)
        VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
        intensity_col = 1.0 - np.log(points_i) / np.log(np.exp(-0.4))
        result = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
        return np.clip(result, 0.0 , 1.0)

    def get_point_count(file_name):
        point_list = o3d.io.read_point_cloud(file_name)
        return np.asarray(point_list.points).shape[0]

    def get_euclidean_distance(p1, p2):
        # p1, p2 shape = [1, 3]
        return np.linalg.norm(np.asarray(p1) - np.asarray(p2))



    def cal_snn_rmse_by_open3d(source_file_name, target_file_name, source_avg_distance=None, target_avg_distance=None):
        if source_avg_distance is None or target_avg_distance is None:
            source_file_name = str(source_file_name)
            if source_file_name.split(".")[-1] == "bin":
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(np.fromfile(source_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3])
            else:
                source = o3d.io.read_point_cloud(source_file_name)

            target_file_name = str(target_file_name)
            if target_file_name.split(".")[-1] == "bin":
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(np.fromfile(
                    target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3])
            else:
                target = o3d.io.read_point_cloud(target_file_name)

            source_distances = np.asarray(source.compute_point_cloud_distance(target))
            source_avg_distance = np.sum(np.square(source_distances))/len(source.points)  # MSE(P, Q)

            target_distances = np.asarray(target.compute_point_cloud_distance(source))
            target_avg_distance = np.sum(np.square(target_distances))/len(target.points)  # MSE(P, Q)

        return math.sqrt((source_avg_distance + target_avg_distance)/2)
    
    def cal_snn_rmse_by_file_name(source_file_name, target_file_name, source_avg_distance=None, target_avg_distance=None):
        if source_avg_distance is None or target_avg_distance is None:
            source_file_name = str(source_file_name)
            if source_file_name.split(".")[-1] == "bin":
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(np.fromfile(
                    source_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3])
            else:
                source = o3d.io.read_point_cloud(source_file_name)

            target_file_name = str(target_file_name)
            if target_file_name.split(".")[-1] == "bin":
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(np.fromfile(
                    target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3])
            else:
                target = o3d.io.read_point_cloud(target_file_name)

            source_distances = np.asarray(source.compute_point_cloud_distance(target))
            source_avg_distance = np.sum(np.square(source_distances))/len(source.points)  # MSE(P, Q)

            target_distances = np.asarray(target.compute_point_cloud_distance(source))
            target_avg_distance = np.sum(np.square(target_distances))/len(target.points)  # MSE(P, Q)

        return math.sqrt((source_avg_distance + target_avg_distance)/2)

    def cal_acd_by_file_name(source_file_name, target_file_name, source_distances=None):
        source_file_name = str(source_file_name)
        target_file_name = str(target_file_name)

        if source_file_name.split(".")[-1] == "bin":
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(np.fromfile(
                source_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3])
        else:
            source = o3d.io.read_point_cloud(source_file_name)

        if source_distances is None:
            if target_file_name.split(".")[-1] == "bin":
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(np.fromfile(
                    target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3])
            else:
                target = o3d.io.read_point_cloud(target_file_name)
            source_distances = np.asarray(source.compute_point_cloud_distance(target))

        return np.sum(np.square(source_distances))/len(source.points)

    def cal_cd_by_file_name(source_file_name, target_file_name, acd_source=None, target_distances=None):
        if acd_source is None:
            acd_source = O3dUtil.cal_acd_by_file_name(source_file_name, target_file_name)

        return (acd_source + O3dUtil.cal_acd_by_file_name(target_file_name, source_file_name, target_distances))/2

    def cal_cd_psnr_by_file_name(source_file_name, target_file_name, cd=None):
        target_file_name = str(target_file_name)

        if target_file_name.split(".")[-1] == "bin":
            target = np.fromfile(target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            target = np.asarray(o3d.io.read_point_cloud(target_file_name).points)

        point1 = [target[:, 0].min(), target[:, 1].min(), target[:, 2].min()]
        point2 = [target[:, 0].max(), target[:, 1].max(), target[:, 2].max()]

        max_diameter = O3dUtil.get_euclidean_distance(point1, point2)

        if cd is None:
            cd = O3dUtil.cal_cd_by_file_name(source_file_name, target_file_name)

        result = np.square(max_diameter) / cd
        return 10 * math.log(result, 10)

    def cal_hd_by_file_name(source_file_name, target_file_name):
        from hausdorff import hausdorff_distance
        source_file_name = str(source_file_name)
        target_file_name = str(target_file_name)

        if source_file_name.split(".")[-1] == "bin":
            source = np.fromfile(source_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            source = np.asarray(o3d.io.read_point_cloud(source_file_name).points)

        if target_file_name.split(".")[-1] == "bin":
            target = np.fromfile(target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            target = np.asarray(o3d.io.read_point_cloud(target_file_name).points)

        return hausdorff_distance(source, target, distance='euclidean')

    def cal_emd_by_file_name(source_file_name, target_file_name):
        import torch
        from ShapeMeasure.distance import EMDLoss, ChamferLoss
        source_file_name = str(source_file_name)
        target_file_name = str(target_file_name)

        if source_file_name.split(".")[-1] == "bin":
            source = np.fromfile(source_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            source = np.asarray(o3d.io.read_point_cloud(source_file_name).points)

        if target_file_name.split(".")[-1] == "bin":
            target = np.fromfile(target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            target = np.asarray(o3d.io.read_point_cloud(target_file_name).points)

        emd_util = EMDLoss()
        p1 = torch.from_numpy(source).cuda()  # .double()
        p2 = torch.from_numpy(target).cuda()  # .double()

        p1.requires_grad = True
        p2.requires_grad = True

        emd_list = emd_util(p1, p2)
        return torch.mean(emd_list)

    def get_max_distance(source_file_name, target_file_name):
        source_file_name = str(source_file_name)
        target_file_name = str(target_file_name)

        if source_file_name.split(".")[-1] == "bin":
            source = np.fromfile(source_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            source = np.asarray(o3d.io.read_point_cloud(source_file_name).points)

        if target_file_name.split(".")[-1] == "bin":
            target = np.fromfile(target_file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            target = np.asarray(o3d.io.read_point_cloud(target_file_name).points)

        distance_list = []
        for p1 in source:
            p1_distance_list = []
            for p2 in target:
                if p1.all() != p2.all():
                    p1_distance_list.append(O3dUtil.get_euclidean_distance(p1, p2))
            distance_list.append(np.asarray(p1_distance_list).max())
        return np.asarray(distance_list).max()

    def get_max_distance_one_frame(file_name):
        file_name = str(file_name)

        if file_name.split(".")[-1] == "bin":
            points_xyz = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        else:
            points_xyz = np.asarray(o3d.io.read_point_cloud(file_name).points)

        points_number = len(points_xyz)
        max_distance = 0.0
        for i in range(points_number):
            # Last i elements are already in place
            for j in range(0, points_number-i-1):
                print(j, j+1)
                distance = O3dUtil.get_euclidean_distance(points_xyz[j], points_xyz[j+1])
                if distance > max_distance:
                    max_distance = distance

        return max_distance


    def rotate_by_matrix(points_xyzi, rotation_matrix):
        points_xyzi[:, :3] = np.dot(points_xyzi[:, :3], rotation_matrix)
        return points_xyzi
    
    def translate_by_matrix(points_xyzi, translate):
        points_xyzi[:, :3] = points_xyzi[:, :3] + translate
        return points_xyzi
    
    def get_rotation_matrix_from_angles(roll, pitch, yaw):
        # [skewen]: roll,pitch,yaw is in degree, need to convert to radian
        roll,pitch,yaw = roll*math.pi/180, pitch*math.pi/180, yaw*math.pi/180
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    
    def save_ply_by_xyzi(points_xyzi, file_name):
        point_list = o3d.geometry.PointCloud()
        point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
        point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
        o3d.io.write_point_cloud(str(file_name), point_list)
        