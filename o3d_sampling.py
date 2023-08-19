import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
from socket import *
import json
import time
import zmq
from multiprocessing import Process
import threading
from multiprocessing import Pool
import yaml
import math
from pathlib import Path
from o3d_icp import GlobalRegistration
from o3d_util import O3dUtil


class O3dSampling(object):
    def __init__(self):
        pass
    def get_lidar_by_bin_multi_type(points_xyzi, npoints, sampling_type, remove_outlier=False):
        import torch
        import torch.nn as nn
        if npoints < 0:
            pc = points_xyzi
            pc = torch.from_numpy(pc).t()
            color = np.zeros([points_xyzi.shape[0], 3]).astype('float32')
            color = torch.from_numpy(color).t()
            return pc, color
        if sampling_type == 0:
            # random
            return O3dSampling.get_lidar_by_bin_random(points_xyzi, npoints, remove_outlier)
        elif sampling_type == 1:
            # grid
            return O3dSampling.get_lidar_by_bin_grid(points_xyzi, npoints, remove_outlier)
        elif sampling_type == 2:
            # voxel
            return O3dSampling.get_lidar_by_bin_voxel(points_xyzi, npoints, remove_outlier)
        elif sampling_type == 3:
            # geometric
            return O3dSampling.get_lidar_by_bin_geometric(points_xyzi, npoints, remove_outlier)
        elif sampling_type == 4:
            # farthest
            return O3dSampling.get_lidar_by_bin_farthest_pointnet2(points_xyzi, npoints, remove_outlier)
        else:
            # random
            return O3dSampling.get_lidar_by_bin_random(points_xyzi, npoints, remove_outlier)


    def get_lidar_by_bin_random(points_xyzi, npoints, remove_outlier):
        import torch
        import torch.nn as nn
        if remove_outlier:
            point_list = o3d.geometry.PointCloud()
            point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
            point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
            point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                        (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
            point_list = O3dSampling.o3d_remove_outlier(point_list)
            points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
            points_xyzi = points_xyzi.astype(np.float32)
            
        raw_num = points_xyzi.shape[0]
        if raw_num >= npoints:
            sample_idx = np.random.choice(raw_num, npoints, replace=False)
        else:
            sample_idx = np.concatenate((np.arange(raw_num), np.random.choice(
                raw_num, npoints - raw_num, replace=True)), axis=-1)

        pc = points_xyzi[sample_idx, :]
        pc = torch.from_numpy(pc).t()
        color = np.zeros([npoints, 3]).astype('float32')
        color = torch.from_numpy(color).t()
        
        pc = pc.unsqueeze(0).cuda()
        color = color.unsqueeze(0).cuda()

        return pc, color

    def get_lidar_by_bin_grid(points_xyzi, npoints, remove_outlier):
        import torch
        import torch.nn as nn
        sampling_value = int(points_xyzi.shape[0] / npoints)
        point_list = o3d.geometry.PointCloud()
        point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
        point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
        point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                    (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
        
        if remove_outlier:
            point_list = O3dSampling.o3d_remove_outlier(point_list)

        if len(point_list.points) > npoints:
            point_list = point_list.uniform_down_sample(sampling_value)
        
        points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
        
        points_xyzi = points_xyzi.astype(np.float32)

        points_xyzi = O3dSampling.adaptive_sampling_by_xyzi(points_xyzi, npoints)
        
        pc = torch.from_numpy(points_xyzi).t()
        color = np.zeros([npoints, 3]).astype('float32')
        color = torch.from_numpy(color).t()

        pc = pc.unsqueeze(0).cuda()
        color = color.unsqueeze(0).cuda()
        return pc, color


    def get_lidar_by_bin_voxel(points_xyzi, npoints, remove_outlier):
        import torch
        import torch.nn as nn
        point_list = o3d.geometry.PointCloud()
        point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
        point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
        point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                    (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
        if remove_outlier:
            point_list = O3dSampling.o3d_remove_outlier(point_list)

        point_list = point_list.voxel_down_sample(0.45)
        
        points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
        
        points_xyzi = points_xyzi.astype(np.float32)

        points_xyzi = O3dSampling.adaptive_sampling_by_xyzi(points_xyzi, npoints)
        
        pc = torch.from_numpy(points_xyzi).t()
        color = np.zeros([npoints, 3]).astype('float32')
        color = torch.from_numpy(color).t()

        pc = pc.unsqueeze(0).cuda()
        color = color.unsqueeze(0).cuda()

        return pc, color

    def get_lidar_by_bin_geometric(points_xyzi, npoints, remove_outlier):
        import torch
        import torch.nn as nn
        sampling_value = int(points_xyzi.shape[0] / npoints)
        point_list = o3d.geometry.PointCloud()
        point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
        point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
        point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                    (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))

        if remove_outlier:
            point_list = O3dSampling.o3d_remove_outlier(point_list)

        knn_num = 10
        angle_thre = 30

        point = np.asarray(point_list.points)
        point_size = point.shape[0]
        tree = o3d.geometry.KDTreeFlann(point_list)
        o3d.geometry.PointCloud.estimate_normals(
            point_list, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
        normal = np.asarray(point_list.normals)
        normal_angle = np.zeros((point_size))
        for i in range(point_size):
            [_, idx, dis] = tree.search_knn_vector_3d(point[i], knn_num + 1)
            current_normal = normal[i]
            knn_normal = normal[idx[1:]]
            normal_angle[i] = np.mean(O3dSampling.vector_angle(current_normal, knn_normal))
        
        pcd_high = o3d.geometry.PointCloud()
        pcd_high.points = o3d.utility.Vector3dVector(point[np.where(normal_angle >= angle_thre)])
        pcd_high.normals = o3d.utility.Vector3dVector(normal[np.where(normal_angle >= angle_thre)])
        
        pcd_low = o3d.geometry.PointCloud()
        pcd_low.points = o3d.utility.Vector3dVector(point[np.where(normal_angle < angle_thre)])
        pcd_low.normals = o3d.utility.Vector3dVector(normal[np.where(normal_angle < angle_thre)])
        
        
        sampling_value = (len(pcd_high.points)+len(pcd_low.points)) / npoints
        N = round(sampling_value*3)
        C = round(sampling_value/3*2)
        
        pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, N)
        pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, C)
        
        pcd_finl = o3d.geometry.PointCloud()
        pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                    np.asarray(pcd_low_down.points))))
        pcd_finl.normals = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.normals),
                                    np.asarray(pcd_low_down.normals))))
        
        points_xyzi = np.concatenate((np.asarray(pcd_finl.points), np.asarray(pcd_finl.normals)[:, 0].reshape(-1, 1)), axis=1)

        points_xyzi = points_xyzi.astype(np.float32)

        points_xyzi = O3dSampling.adaptive_sampling_by_xyzi(points_xyzi, npoints)
        
        pc = torch.from_numpy(points_xyzi).t()
        color = np.zeros([npoints, 3]).astype('float32')
        color = torch.from_numpy(color).t()

        pc = pc.unsqueeze(0).cuda()
        color = color.unsqueeze(0).cuda()

        return pc, color

    def get_lidar_by_bin_farthest_open3d(points_xyzi, npoints, remove_outlier):
        import torch
        import torch.nn as nn
        point_list = o3d.geometry.PointCloud()
        point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
        point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
        point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                    (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))

        if remove_outlier:
            point_list = O3dSampling.o3d_remove_outlier(point_list)
        point_list = point_list.farthest_point_down_sample(npoints)

       
        points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
        
        points_xyzi = points_xyzi.astype(np.float32)

        pc = torch.from_numpy(points_xyzi).t()
        color = np.zeros([npoints, 3]).astype('float32')
        color = torch.from_numpy(color).t()

        pc = pc.unsqueeze(0).cuda()
        color = color.unsqueeze(0).cuda()
        return pc, color

    def get_lidar_by_bin_farthest_pointnet2(points_xyzi, npoints, remove_outlier):
        import torch
        import torch.nn as nn
        sys.path.append(os.path.join("/home/skewen/lidar-base-point-cloud-error-concealment/Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops"))
        from pointnet2_utils import furthest_point_sample

        torch_points_xyz = torch.from_numpy(points_xyzi[:, :3]).unsqueeze(0).transpose(2,1).contiguous()
        torch_points_xyz = torch_points_xyz.float().cuda()
        result = furthest_point_sample(torch_points_xyz, npoints)
        np_result = result.squeeze(0).cpu().numpy()
        points_xyzi = points_xyzi[np_result]
        
        points_xyzi = points_xyzi.astype(np.float32)

        pc = torch.from_numpy(points_xyzi).t()
        color = np.zeros([npoints, 3]).astype('float32')
        color = torch.from_numpy(color).t()

        pc = pc.unsqueeze(0).cuda()
        color = color.unsqueeze(0).cuda()
        return pc, color

    def adaptive_sampling_by_xyzi(points, npoints):
        raw_num = points.shape[0]
        if raw_num >= npoints:
            sample_idx = np.random.choice(raw_num, npoints, replace=False)
        else:
            sample_idx = np.concatenate((np.arange(raw_num), np.random.choice(
                raw_num, npoints - raw_num, replace=True)), axis=-1)

        return points[sample_idx, :]

    def vector_angle(x, y):
        Lx = np.sqrt(x.dot(x))
        Ly = (np.sum(y ** 2, axis=1)) ** (0.5)
        cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        angle2 = angle * 360 / 2 / np.pi
        return angle2
    
    def o3d_remove_outlier(point_cloud):
        point_cloud, index = point_cloud.remove_radius_outlier(nb_points=5, radius=2)
        return point_cloud

    def bin2xyzi(file_name):
        return np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])

def main():
    file_name = "/mnt/data/skewen/kittiGenerator/output/20221110_2_50_0_454/object/training/velodyne/000000.bin"
    point_xyzis = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    for i in range(5):
        pc, color = O3dSampling.get_lidar_by_bin_multi_type(point_xyzis, 16384, i)
        print("{} -> {} {}".format(i, pc.shape, color.shape))
        
if __name__ == "__main__":
    main()