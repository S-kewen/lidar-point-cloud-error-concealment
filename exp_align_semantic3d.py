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
import open3d as o3d
from sklearn.neighbors import KDTree

'''
Using to re-match semantic3D-format dataset from concealed frames
'''

with open('config.yaml', 'r') as yamlfile:
    m_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

def fun_align_semantic3d(exp_path, file_name):
    start_time = time.time()
    semantic3d_xyzirgb = np.loadtxt(Path () / exp_path / "semantic3d_xyzirgb" / "{}.txt".format(file_name.stem), dtype=np.float32)
    semantic3d_label = np.loadtxt(Path () / exp_path / "semantic3d_label" / "{}.labels".format(file_name.stem), dtype=np.float32)
    assert semantic3d_xyzirgb.shape[0] > 0
    assert semantic3d_xyzirgb.shape[0] == semantic3d_label.shape[0]

    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])

    kd_tree = KDTree(semantic3d_xyzirgb[:, :3])

    dist, indexs = kd_tree.query(points_xyzi[:, :3], k=1)
    indexs = indexs.reshape(-1)
    points_xyzirgb = np.concatenate((points_xyzi, semantic3d_xyzirgb[indexs][:, 4:7]), axis=1)
    
    np.savetxt(file_name.parent / "{}_semantic3d_xyzirgb.txt".format(file_name.stem), points_xyzirgb,fmt='%f',delimiter=' ')
    np.savetxt(file_name.parent / "{}_semantic3d_label.labels".format(file_name.stem), semantic3d_label[indexs],fmt='%d',delimiter=' ')

def fun_align_semantic3d_knn(exp_path, file_name, k):
    semantic3d_xyzirgb = np.loadtxt(Path () / exp_path / "semantic3d_xyzirgb" / "{}.txt".format(file_name.stem), dtype=np.float32)
    semantic3d_label = np.loadtxt(Path () / exp_path / "semantic3d_label" / "{}.labels".format(file_name.stem), dtype=np.int64)
    
    assert semantic3d_xyzirgb.shape[0] > 0
    assert semantic3d_xyzirgb.shape[0] == semantic3d_label.shape[0]

    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])

    kd_tree = KDTree(semantic3d_xyzirgb[:, :3])

    assert k > 0
    dist, indexs = kd_tree.query(points_xyzi[:, :3], k=k)
    
    if k == 1:
        most_frequent_indexs = indexs.reshape(-1)
        points_xyzirgb = np.concatenate((points_xyzi, semantic3d_xyzirgb[most_frequent_indexs][:, 4:7]), axis=1)
        results_semantic3d_label = semantic3d_label[most_frequent_indexs]
    else:
        rgbs = np.asarray([[0, 0, 0], [128, 64, 128], [145, 170, 100], [107, 142, 35], [40, 60, 150], [70, 70, 70], [102, 102, 156], [220, 220, 0], [0, 0, 142]])
        
        results_semantic3d_label = np.apply_along_axis(get_most_frequent, 1, semantic3d_label[indexs])
        
        semantic3d_rgb = rgbs[results_semantic3d_label]
        
        points_xyzirgb = np.concatenate((points_xyzi, semantic3d_rgb), axis=1)
    
    np.savetxt(file_name.parent / "{}_semantic3d_xyzirgb.txt".format(file_name.stem), points_xyzirgb, fmt='%f', delimiter=' ')
    np.savetxt(file_name.parent / "{}_semantic3d_label.labels".format(file_name.stem), results_semantic3d_label, fmt='%d', delimiter=' ')
    

def get_most_frequent(x):
    return np.argmax(np.bincount(x))

def get_rgb_by_class_id(class_id):
    rgb_maps = {0: [0, 0, 0], 1: [128, 64, 128], 2: [145, 170, 100], 3: [107, 142, 35], 4: [40, 60, 150],
                5: [70, 70, 70], 6: [102, 102, 156], 7: [220, 220, 0], 8: [0, 0, 142]}
    if class_id in rgb_maps.keys():
        return rgb_maps[class_id]
    else:
        return [0, 0, 0]
    
def get_cache_list():
    cache_list = []
    if m_config["exp"]["none"]["enable"]:
        cache_list.append("none")
    if m_config["exp"]["location-stiching"]["enable"]:
        cache_list.append("location_stiching")
    if m_config["exp"]["mc-stiching"]["enable"]:
        cache_list.append("mc_stiching")
    if m_config["exp"]["icp"]["enable"]:
        cache_list.append("icp")
    return cache_list

def main():
    exp_path = m_config["exp"]["path"]

    packet_loss_rate_list = m_config["exp"]["packetLossRateList"]

    cache_list = get_cache_list()
        
    interpolation_list = m_config["exp"]["interpolationList"]

    dirs = []
    for packet_loss_rate in packet_loss_rate_list:
        for cache in cache_list:
            for interpolation in interpolation_list:
                dirs.append(Path() / exp_path / "cache_frame" /
                            "receiver_{}_{}".format(cache, packet_loss_rate) / str(interpolation))

    with Pool(processes=None) as pool:
        for dir in dirs:
            print(dir)
            for file_name in sorted(dir.glob("*.bin")):
                pool.apply_async(fun_align_semantic3d_knn, (exp_path, file_name, 3))
                
        pool.close()
        pool.join()
    print("MAIN THREAD STOP")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time()-start_time))
