import time
import threading
import open3d as o3d
import copy
import numpy as np
from pathlib import Path
from o3d_icp import GlobalRegistration
import math
from matplotlib import cm
import random

from o3d_util import O3dUtil

from hausdorff import hausdorff_distance
import shutil
import yaml
import json
from my_util import MyUtil as my_util

m_config = my_util.get_config("config.yaml")


def get_point_color(point_i):
    VIRIDIS = np.array(cm.get_cmap('plasma').colors)

    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
    intensity_col = 1.0 - np.log(point_i) / np.log(np.exp(-0.004 * 100))
    return np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]


def show_point_cloud_by_bin(file_name, title=""):
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    print(points_xyzi.shape)
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    o3d.visualization.draw_geometries([point_list], title)


def save_ply(points_xyzis, save_name):
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzis[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_xyzis[:, 3]))
    print("point_list.points： {}".format(np.asarray(point_list.points).shape))
    o3d.io.write_point_cloud(str(save_name), point_list)


# show all frame
# base_path = Path() / m_config["exp"]["path"] / "cache_frame"
# for i, file_name in enumerate(sorted((base_path).glob("**/*.bin"))):
#     print("[{}] {}".format(i, file_name))
#     point_count = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
#     show_point_cloud_by_bin(str(file_name), "{} {}".format(point_count.shape, file_name))


# print(np.load('/mnt/sda/kittiGenerator/output/test_100/object/training/velodyne/evaluate/pointnet++_cls.npy', allow_pickle=True))


# xxxx = json.loads(json.dumps({"identity":"111", "cust_name":"xxx", "listtype":"2222", "start_date":"2019-12-12", "end_date":"2019-12-29"}))
# for k in xxxx:
#     print(k, xxxx[k])


# base_path = Path() / m_config["exp"]["path"] / "ns3"
# latency_list = []
# for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
#     print("[{}] {}".format(i, file_name))
#     with open(file_name, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             json_data = json.loads(line)
#             if json_data["type"] == "rx":
#                 latency = (json_data["recvTime"] - json_data["sendTime"])*1000
#             else:
#                 latency = -1
#             latency_list.append(latency)

# print(latency_list)



# base_path = Path() / m_config["exp"]["path"] / "ns3"
# for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
#     if int(file_name.stem)>=700:
#         file_name.unlink()



# point_cloud = o3d.io.read_point_cloud("/mnt/sda/kittiGenerator/output/NS3_OfdmRate27MbpsBW10MHz/object/training/cache_frame/receiver_none_ns3/000007.ply")

# print(np.asarray(point_cloud.points).shape)

# base_path = Path() / m_config["exp"]["path"] / "cache_frame" / "receiver_none_ns3" / "pointinet_0.5"
# with open("{}.txt".format("test"), "a+") as f:
#     for i, file_name in enumerate(sorted((base_path).glob("**/*.bin"))):
#         f.write(file_name.stem+"\n")



# base_path = Path() / "/mnt/data/skewen/dataset/kitti/object/training/label_2"
# for i, file_name in enumerate(sorted((base_path).glob("*.txt"))):
#     with open(file_name, 'r') as f:
#         result_list = []
#         lines = f.readlines()
#         for line in lines:
#             arrs = line.split(" ")
#             # temp = arrs[10]
#             # arrs[10] = arrs[9]
#             # arrs[9] = temp
#             # arrs[-1] = arrs[-1].replace("\n","")
#             # result_list.append(arrs)
#             if arrs[0] == "Car":
#                 print(arrs[8:11])




# base_path = Path() / "/home/skewen/lidar-base-point-cloud-error-concealment"
# save_path = Path() / "/home/skewen/lidar-base-point-cloud-error-concealment"
# for i, file_name in enumerate(sorted((base_path).glob("*.bin"))):
#     point_xyzs = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
#     save_ply(point_xyzs, save_path / "{}.ply".format(file_name.stem))



# base_path = Path() / "/mnt/data/skewen/kittiGenerator/output/val_20221116_50_0_200/object/training/cache_frame"
# dirs = []
# packet_loss_rate_list = m_config["exp"]["packetLossRateList"]

# for packet_loss_rate in packet_loss_rate_list:
#     dirs.append("receiver_none_{}".format(packet_loss_rate))
    
# for dir in dirs:
#     sum_packet_loss_rate = 0.0
#     sum_file_count = 0
#     for i, file_name in enumerate(sorted((Path() / base_path / dir).glob("*_log.npy"))):
#         sector_list = np.load(file_name, allow_pickle=True).item()["sectorList"]
#         sector_size = np.load(file_name, allow_pickle=True).item()["sectorSize"]
#         packet_loss_rate = 1 - len(sector_list)/sector_size
#         #print("[{}] {} = {}".format(dir, sector_list, packet_loss_rate))
#         sum_packet_loss_rate += packet_loss_rate
#         sum_file_count += 1
#     print("Avg Packet Loss Rate ({}): {}".format(dir, sum_packet_loss_rate/sum_file_count))
        
# def get_object_num(easy, mid, hard):
#     easy_count = 464*easy*0.01
#     mid_count = 71*mid*0.01
#     hard_count = 16*hard*0.01
#     return easy_count+mid_count+hard_count

# print(get_object_num(32.1848, 28.3152, 28.3340))

# base_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230208_20_1_50_0_200/object/training/location"
# for i, file_name in enumerate(sorted((base_path).glob("*.npy"))):
#     location = np.load(file_name, allow_pickle=True).item()
#     print("{}".format(location))


# start_time = time.process_time()
# base_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230211_False_1_1_50_0_200/object/training/cache_frame/receiver_closest_spatial_0.2"
# for i, file_name in enumerate(sorted((base_path).glob("*.npy"))):
#     reference_id_list = np.load(file_name, allow_pickle=True).item()["reference_id_list"]
#     print("[{}] {}".format(i, reference_id_list))

# print("time.process_time(): {}".format(time.process_time() - start_time))

# 20230308_False_1_1_140_0_600
# 20230308_False_1_3_140_0_600
# 20230309_False_1_5_140_0_600
# 20230309_False_1_7_140_0_600.

# dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/lec_training_set"

# file_names = ["lec_training_set_01.csv", "lec_training_set_02.csv", "lec_training_set_04.csv", "lec_training_set_05.csv", "lec_training_set_06.csv", "lec_training_set_08.csv", "lec_training_set_09.csv", "lec_training_set_10.csv"]
# result = np.zeros((0, 7))
# # for i, file_name in enumerate(sorted((dir).glob("*.csv"))):
# #      result = np.concatenate((result, np.genfromtxt(file_name, delimiter=',')), axis=0)

# for i, file_name in enumerate(file_names):
#     result = np.concatenate((result, np.genfromtxt(dir / file_name, delimiter=',')), axis=0)

# print("result: {}".format(result.shape))
# np.save(dir / "output.npy", result)


# csv_file_name = "/mnt/data2/skewen/kittiGenerator/output/20230420_False_1_1_110_0_10000/object/training/lec_training_set.csv"
# npy_file_name = "/mnt/data2/skewen/kittiGenerator/output/20230420_False_1_1_110_0_10000/object/training/lec_training_set.npy"
# data = np.genfromtxt(csv_file_name, delimiter=',')
# np.save(npy_file_name, data)


# array = np.load(npy_file_name, allow_pickle=True)
# print(array.shape)

file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230413_False_1_6_110_0_600_1/object/training/velodyne_compression/000000.bin"

point_xyzs = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
print(point_xyzs)
point_xyzs[:, 2] = -point_xyzs[:, 2]
print(point_xyzs)
save_ply(point_xyzs,  "/mnt/data2/skewen/kittiGenerator/output/{}.ply".format(file_name.stem))