import time
import numpy as np
from pathlib import Path
import math
import timeit

def get_ycd_by_xyzis(point_xyzis, channel_list, lidar_coordinate = [0,0,0]):
    points_a = np.arctan(point_xyzis[:, 1] / point_xyzis[:, 0]) / math.pi * 180
    points_xyzias = np.concatenate((point_xyzis, points_a.reshape(-1, 1)), axis = 1)
    points_xyzias[np.where((points_xyzias[:, 0] == 0.0) & (points_xyzias[:, 1] > 0)), 4] = 90.0
    points_xyzias[np.where((points_xyzias[:, 0] == 0.0) & (points_xyzias[:, 1] <= 0)), 4] = -90.0
    points_xyzias[np.where(points_xyzias[:, 0] < 0), 4] += 180
    points_xyzias[np.where(points_xyzias[:, 4] < 0), 4] += 360

    points_c = np.arctan(point_xyzis[:, 2] / (np.sqrt(np.square(point_xyzis[:, 0]) + np.square(point_xyzis[:, 1])))) / np.pi * 180
    point_xyzics = np.concatenate((point_xyzis, -np.ones((point_xyzis.shape[0], 1))), axis=1)
    point_xyzics = np.concatenate((point_xyzics, points_c.reshape(-1, 1)), axis=1)
    for i, c in enumerate(channel_list):
        point_xyzics[np.where(((abs(point_xyzics[:, 5] - c) <= 0.1) & (point_xyzics[:, 4] == -1))), 4] = i
    return np.concatenate((points_xyzias[:, -1].reshape(-1, 1), point_xyzics[:, -2].reshape(-1, 1), np.sqrt(np.sum((point_xyzis[:, :3] - lidar_coordinate) ** 2, axis=1)).reshape(-1, 1)), axis = 1)


def get_pitch_by_channel_size(start, end, channel_size):
    unit = abs(start-end)/(channel_size-1)
    return np.concatenate((np.arange(start, end, unit, dtype=float), [end]))


def distance_between_point(point1, point2):
    return math.sqrt(pow(point1[0]-point2[0], 2)+pow(point1[1]-point2[1], 2)+pow(point1[2]-point2[2], 2))


def get_xyz_by_ypd(yaw, pitch, dist):
    x = dist * np.cos(yaw) * np.cos(pitch)
    y = dist * np.sin(yaw) * np.cos(pitch)
    z = dist * np.sin(pitch)
    return x, y, z

def get_channel_by_point_cloud(point):
    return np.arctan(point[2]/(np.sqrt(np.square(point[0]) + np.square(point[1])))) / np.pi * 180


def get_channel_by_pointxyzis(point_xyzis, channels):
    points_c = np.arctan(point_xyzis[:, 2]/(np.sqrt(np.square(point_xyzis[:, 0]) + np.square(point_xyzis[:, 1])))) / np.pi * 180
    point_xyzics = np.concatenate((point_xyzis, -np.ones((point_xyzis.shape[0], 1))), axis = 1)
    point_xyzics = np.concatenate((point_xyzics, points_c.reshape(-1, 1)), axis = 1)
    for i, c in enumerate(channels):
        point_xyzics[np.where(((abs(point_xyzics[:, 5] - c) <= 0.1) & (point_xyzics[:, 4] == -1))), 4] = i
    return point_xyzics[:, :5]

def get_channel_by_pointxyzis_sm(point_xyzis, channels):
    points_c = np.arctan(point_xyzis[:, 2]/(np.sqrt(np.square(point_xyzis[:, 0]) + np.square(point_xyzis[:, 1])))) / np.pi * 180
    _channels = channels.reshape((-1, 64))
    point_xyzics = np.concatenate((point_xyzis, points_c.reshape(-1, 1)), axis = 1)
    i_s = np.arange(64).astype(int)
    mask = (np.abs(point_xyzics[:, 4].reshape((-1, 1)) - _channels) <= 0.1)
    point_xyzics[np.sum(mask, axis=1) == 1, 4] = i_s[np.argmax(mask, axis=1)][np.sum(mask, axis=1) == 1]
    return point_xyzics

def get_angle_by_point(point):
    result = math.atan(point[1]/point[0])/math.pi*180
    
    if point[0] == 0.0 and point[1] > 0:
        result = 90.0
    if point[0] == 0.0 and point[1] <= 0:
        result = -90.0
    if point[0] < 0:
        result += 180
    if result < 0:
        result += 360
    return result

def get_angle_by_point_xyzis(point_xyzis):
    points_a = np.arctan(point_xyzis[:, 1] / point_xyzis[:, 0]) / math.pi * 180
    points_xyzia = np.concatenate((point_xyzis, points_a.reshape(-1, 1)), axis = 1)

    points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] > 0)), 4] = 90.0
    points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] <= 0)), 4] = -90.0
    points_xyzia[np.where(points_xyzia[:, 0] < 0), 4] += 180
    points_xyzia[np.where(points_xyzia[:, 4] < 0), 4] += 360
    
    return points_xyzia[:, 4]



def check_points():
    pitch_list = get_pitch_by_channel_size(-16.8, 10, 64)

    base_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230221_False_1_1_150_0_10/object/training/velodyne"
    running_time_list = []
    last_angle = 0.0
    channels_list = []
    for i, file_name in enumerate(sorted((base_dir).glob("*.bin"))):
        point_xyzis = np.fromfile(str(file_name), dtype=np.float32, count=-1).reshape([-1, 4])
        # point_xyzics = get_channel_by_pointxyzis(point_xyzis, pitch_list)

        # print(np.sum(point_xyzics[:, 5] == -1))
        for p in point_xyzis:
            yaw = get_angle_by_point(p) / 180 * np.pi
            pitch = get_channel_by_point_cloud(p) / 180 * np.pi
            dist = distance_between_point(p, [0,0,0])
            
            p2 = get_xyz_by_ypd(yaw, pitch, dist)
            print("p1: {}".format(p[:3]))
            print("p2: {}".format(np.asarray(p2)))
            break
        # point_as = get_angle_by_point_xyzis(point_xyzis)
        # for i, p in enumerate(point_as):
        #     print(i, p)
        # print(np.unique(channels_list).shape)
        # np.savetxt("angles.txt", point_as,fmt='%f',delimiter=',')
        # break

def main():
    check_points()
    
    
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time() - start_time))