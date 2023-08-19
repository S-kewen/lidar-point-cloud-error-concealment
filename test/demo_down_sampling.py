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
import torch
import torch.nn as nn

with open('config.yaml', 'r') as yamlfile:
    m_config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def format_ply(file_name):
    points_i = []  # [i1,i2...]
    points_xyz = []  # [[x,y,z],...]
    with open(file_name) as pc:  # format ply file to xyzi array
        lines = pc.readlines()
        for i in range(len(lines)):
            coor = lines[i].split()
            if len(coor) < 4:
                continue
            coor = [float(item) for item in coor]

            pointsXYZ = [0.0, 0.0, 0.0]
            points_i.append(coor[-1])
            for j in range(3):
                pointsXYZ[j] = coor[j]
            pointsXYZ[0] = -pointsXYZ[0]
            points_xyz.append(pointsXYZ)  # [[x,y,z],...]

    return points_xyz, points_i

def fun_sampling_random_depth(file_name, save_dir, npoints=16384):
    points_xyz, points_i = format_ply(file_name)
    points_xyzi = np.concatenate((points_xyz, np.asarray(points_i).reshape((-1, 1))), axis=1)

    if npoints < len(points_xyzi):
        # downsampling
        pts_depth = points_xyzi[:, 2]
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=False)
        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
            if len(far_idxs_choice) > 0 else near_idxs_choice
    else:
        # upsampling
        choice = np.arange(0, len(points_xyzi), dtype=np.int32)
        if npoints > len(points_xyzi):
            # here is adaptive choice points
            extra_choice = np.random.choice(choice, npoints - len(points_xyzi),
                                            replace=(npoints - len(points_xyzi)) > len(choice))
            choice = np.concatenate((choice, extra_choice), axis=0)

    # np.random.shuffle(choice)
    result = points_xyzi[choice, :]
    result.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))
    return result


def create_dir(dir):
    if not dir.exists():
        print("mkdir: {}".format(dir))
        dir.mkdir(parents=True)


def adaptive_sampling_by_xyzi(points, npoints):
    # random sampling
    raw_num = points.shape[0]
    if raw_num >= npoints:
        #downsampling
        sample_idx = np.random.choice(raw_num, npoints, replace=False)
    else:
        #upsampling
        sample_idx = np.concatenate((np.arange(raw_num), np.random.choice(
            raw_num, npoints - raw_num, replace=True)), axis=-1)

    return points[sample_idx, :]

def xyzi2pc(points_xyzi):
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
    return point_list


def fun_sampling_random(file_name, save_dir, nPoints=16384):

    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    points_xyzi = np.unique(points_xyzi, axis=0)

    points_xyzi = adaptive_sampling_by_xyzi(points_xyzi, nPoints)
    
    point_list = xyzi2pc(points_xyzi)
    
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(file_name.stem)), point_list)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))


def fun_sampling_grid(file_name, save_dir, nPoints=16384):
    
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    points_xyzi = np.unique(points_xyzi, axis=0)
    
    sampling_value = int(points_xyzi.shape[0] / nPoints)
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    if len(point_list.points) > nPoints:
        point_list = point_list.uniform_down_sample(sampling_value)
    
    points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
    
    points_xyzi = adaptive_sampling_by_xyzi(points_xyzi, nPoints)
    
    point_list = xyzi2pc(points_xyzi)
    
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(file_name.stem)), point_list)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))

def fun_sampling_poisson(file_name, save_dir, nPoints=16384):

    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    points_xyzi = np.unique(points_xyzi, axis=0)
    
    sampling_value = int(points_xyzi.shape[0] / nPoints)
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    point_list = o3d.geometry.sample_points_poisson_disk(point_list, sampling_value)
    
    points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
    
    points_xyzi = adaptive_sampling_by_xyzi(points_xyzi, nPoints)
    
    point_list = xyzi2pc(points_xyzi)
    
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(file_name.stem)), point_list)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))

def vector_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = (np.sum(y ** 2, axis=1)) ** (0.5)
    cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2



def fun_sampling_voxel(file_name, save_dir, nPoints=16384):

    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    points_xyzi = np.unique(points_xyzi, axis=0)
    
    sampling_value = int(points_xyzi.shape[0] / nPoints)
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    point_list = point_list.voxel_down_sample(0.45)
    
    points_xyzi = np.concatenate((np.asarray(point_list.points), np.asarray(point_list.normals)[:, 0].reshape(-1, 1)), axis=1)
    
    points_xyzi = adaptive_sampling_by_xyzi(points_xyzi, nPoints)
    
    point_list = xyzi2pc(points_xyzi)
    
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(file_name.stem)), point_list)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))


def fun_sampling_geometric(file_name, save_dir, nPoints=16384):
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    points_xyzi = np.unique(points_xyzi, axis=0)
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    knn_num = 10  # 自定义参数值(邻域点数)
    angle_thre = 30  # 自定义参数值(角度值)

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
        normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))
    
    pcd_high = o3d.geometry.PointCloud()
    pcd_high.points = o3d.utility.Vector3dVector(point[np.where(normal_angle >= angle_thre)])
    pcd_high.normals = o3d.utility.Vector3dVector(normal[np.where(normal_angle >= angle_thre)])
    
    pcd_low = o3d.geometry.PointCloud()
    pcd_low.points = o3d.utility.Vector3dVector(point[np.where(normal_angle < angle_thre)])
    pcd_low.normals = o3d.utility.Vector3dVector(normal[np.where(normal_angle < angle_thre)])
    
    
    sampling_value = (len(pcd_high.points)+len(pcd_low.points)) / nPoints
    N = round(sampling_value*3)  # 自定义参数值(每N个点采样一次)
    C = round(sampling_value/3*2)  # 自定义参数值(采样均匀性>N)
    
    pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, N)
    pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, C)
    
    pcd_finl = o3d.geometry.PointCloud()
    pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                np.asarray(pcd_low_down.points))))
    pcd_finl.normals = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.normals),
                                np.asarray(pcd_low_down.normals))))
    
    points_xyzi = np.concatenate((np.asarray(pcd_finl.points), np.asarray(pcd_finl.normals)[:, 0].reshape(-1, 1)), axis=1)
    
    points_xyzi = adaptive_sampling_by_xyzi(points_xyzi, nPoints)
    
    point_list = xyzi2pc(points_xyzi)
    
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(file_name.stem)), point_list)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))
    
    
def fun_sampling_farthest(file_name, save_dir, nPoints=16384):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    points_xyzi = np.unique(points_xyzi, axis=0)
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzi[:, 3]))
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    N, D = points_xyzi.shape
    xyz = points_xyzi[:,:3]
    centroids = np.zeros((nPoints,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(nPoints):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points_xyzi = points_xyzi[centroids.astype(np.int32)]
    
    point_list = xyzi2pc(points_xyzi)
    
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(file_name.stem)), point_list)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))


def main():
    
    exp_path = m_config["exp"]["path"]
    
    sector_size = m_config["exp"]["sectorSize"]
    
    packet_loss_rate_list = m_config["exp"]["packetLossRateList"]
    
    source_mac = str(m_config["exp"]["sourceMac"])
    receive_mac = str(m_config["exp"]["receiveMac"])
    
    nPoints = m_config["exp"]["nPoints"]

    start_time = time.time()
    for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
        sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_random"
        create_dir(sampling_dir)
        fun_sampling_random(file_name, sampling_dir, nPoints)
    
    print("random total time: {}".format(time.time() - start_time))

    start_time = time.time()
    for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
        sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_grid"
        create_dir(sampling_dir)
        fun_sampling_grid(file_name, sampling_dir, nPoints)

    print("grid total time: {}".format(time.time() - start_time))

    start_time = time.time()
    for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
        sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_voxel"
        create_dir(sampling_dir)
        fun_sampling_voxel(file_name, sampling_dir, nPoints)
        
    print("voxel total time: {}".format(time.time() - start_time))
    # for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
    #     sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_poisson"
    #     create_dir(sampling_dir)
    #     fun_sampling_poisson(file_name, sampling_dir, nPoints)
        
    start_time = time.time()
    for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
        sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_geometric"
        create_dir(sampling_dir)
        fun_sampling_geometric(file_name, sampling_dir, nPoints)
    
    print("geometric total time: {}".format(time.time() - start_time))
    
    start_time = time.time()
    for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
        sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_farthest"
        create_dir(sampling_dir)
        fun_sampling_farthest(file_name, sampling_dir, nPoints)
    
    print("farthest total time: {}".format(time.time() - start_time))
    # with Pool(processes=None) as pool:
    #     print("generating adaptive_sampling_random...")
    #     sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_random"
    #     create_dir(sampling_dir)
    #     for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
    #             pool.apply_async(fun_sampling_random, (file_name, sampling_dir, nPoints, ))
    #     pool.close()
    #     pool.join()
        
    # with Pool(processes=None) as pool:
    #     print("generating adaptive_sampling_grid...")
    #     sampling_dir = Path() / exp_path / "cache_frame" / "adaptive_sampling_grid"
    #     create_dir(sampling_dir)
    #     for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
    #             pool.apply_async(fun_sampling_grid, (file_name, sampling_dir, nPoints, ))
    #     pool.close()
    #     pool.join()

    print("MAIN THREAD STOP")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time()-start_time))