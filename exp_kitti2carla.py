import numpy as np
import math
from pathlib import Path
from my_util import MyUtil as my_util
from o3d_util import O3dUtil as o3d_util

from json.tool import main
import json
import time
from tqdm import tqdm
import open3d as o3d
from multiprocessing import Pool
from multiprocessing import Process


def get_point_count_by_sector_size(points_xyzi, sector_size):
    sector_list = range(sector_size)
    sector_range = 360 / sector_size
    points_a = np.arctan(points_xyzi[:, 1] / points_xyzi[:, 0]) / math.pi * 180
    points_xyzia = np.concatenate((points_xyzi, points_a.reshape(-1, 1)), axis = 1)

    points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] > 0)), 4] = 90.0
    points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] <= 0)), 4] = -90.0
    points_xyzia[np.where(points_xyzia[:, 0] < 0), 4] += 180
    points_xyzia[np.where(points_xyzia[:, 4] < 0), 4] += 360

    result = []
    for sector in sector_list:
        result.append(points_xyzi[np.where((points_xyzia[:, 4] >= sector * sector_range) & (points_xyzia[:, 4] < (sector + 1) * sector_range))].shape[0])
    return result

def save_packet_data(file_name, timestamp, save_dir, packet_mode = 2, packet_size = 1206, sector_size = 180):
    packet_size_list = []
    if packet_mode == 1:
        packet_size_list = np.full((sector_size,), packet_size).astype(int)
    else:
        npy_log = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()
        compressed_size = npy_log["compressed_size"]
        point_count = npy_log["point_count"]
        bytes_per_point = compressed_size / point_count
        
        points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        packet_size_list = get_point_count_by_sector_size(points_xyzi, sector_size)
        packet_size_list = np.round(np.asarray(packet_size_list) * bytes_per_point).astype(int)
    
    save_txt(save_dir / "{}.txt".format(file_name.stem), {"file_name": str(file_name), "packet_size_list": packet_size_list.tolist(), "timestamp": timestamp})
    np.save(save_dir / "{}.npy".format(file_name.stem), {"file_name": str(file_name), "packet_size_list": packet_size_list, "timestamp": timestamp})

def save_compression_lidar_data(file_name, points_xyzi):
    import DracoPy
    start_time = time.process_time()
    
    colors = np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1)
    colors = (colors * 255).astype(np.uint8)
    binary = DracoPy.encode(points_xyzi[:, :3], colors = colors, preserve_order = True)
    
    buffer_bin = np.frombuffer(binary, dtype=np.uint8)
    buffer_bin.tofile(file_name.parent / "{}.drc".format(file_name.stem))
    
    compressed_drc = DracoPy.decode(np.fromfile(file_name.parent / "{}.drc".format(file_name.stem), dtype=np.uint8).tobytes())
    compressed_points_xyzi = np.concatenate((compressed_drc.points, compressed_drc.colors[:, 0].reshape(-1, 1) / 255), axis=1)
    
    compressed_points_xyzi.astype(np.float32).tofile(file_name.parent / "{}.bin".format(file_name.stem))
    
    o3d_util.save_ply_by_xyzi(compressed_points_xyzi, file_name.parent / "{}.ply".format(file_name.stem))
    
    compressed_size = Path(file_name.parent / "{}.drc".format(file_name.stem)).stat().st_size
    
    point_count = compressed_points_xyzi.shape[0]

    running_time = time.process_time() - start_time
    np.save(file_name.parent / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "compressed_size": compressed_size, "point_count": point_count, "time": time.time(), "running_time": running_time})


def save_txt(file_name, text):
    with open(file_name, "w") as f:
        f.write(json.dumps(np.asarray(text).tolist()))


def save_ground_removal_lidar_data(file_name, points_xyzi, voxel_size = 0.1, ransac_n = 3, distance_threshold = 0.1, num_iterations = 1000):
    start_time = time.process_time()
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    pcd_filtered = point_list.voxel_down_sample(voxel_size=voxel_size)

    all_indexs = np.arange(len(pcd_filtered.points))

    [planes, ground_indexs] = pcd_filtered.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    non_ground_indexs = list(set(all_indexs) - set(ground_indexs))
    
    pcd_non_ground = pcd_filtered.select_by_index(non_ground_indexs)
    
    result = np.concatenate((np.asarray(pcd_non_ground.points), np.asarray(pcd_non_ground.normals)[:, 0].reshape(-1, 1)), axis=1)

    running_time = time.process_time() - start_time
    result.astype(np.float32).tofile(str(file_name))
    o3d_util.save_ply_by_xyzi(result, str(file_name.parent / "{}.ply".format(file_name.stem)))
    np.save(file_name.parent / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "voxel_size": voxel_size, "ransac_n": ransac_n, "distance_threshold": distance_threshold, "num_iterations": num_iterations, "time": time.time(), "running_time": running_time})
    
    return result


def homogeneous_to_absolute(matrix):
    R = matrix[0:3, 0:3]
    
    t = matrix[0:3, 3]
    
    pitch = math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2))
    if pitch == math.pi/2:
        yaw = 0
        roll = math.atan2(R[0,1], R[1,1])
    elif pitch == -math.pi/2:
        yaw = 0
        roll = math.atan2(-R[0,1], R[1,1])
    else:
        yaw = math.atan2(R[1,0]/math.cos(pitch), R[0,0]/math.cos(pitch))
        roll = math.atan2(R[2,1]/math.cos(pitch), R[2,2]/math.cos(pitch))
    
    pitch = pitch * 180 / math.pi
    yaw = yaw * 180 / math.pi
    roll = roll * 180 / math.pi
    
    return t[0], t[1], t[2], roll, pitch, yaw

# def kitti2carla(exp_path, sequence):
#     save_dir = Path() / exp_path / "sequences" / sequence
#     pose_file_name = Path() / exp_path / "poses" / "{}.txt".format(sequence)
    
#     my_util.create_dir(save_dir / "ply")
#     my_util.create_dir(save_dir / "location")
#     my_util.create_dir(save_dir / "packet")
#     my_util.create_dir(save_dir / "velodyne_compression")
#     my_util.create_dir(save_dir / "velodyne_ground_removal")
    
#     file_names = sorted((save_dir / "velodyne").glob("*.bin"))
    
#     for i, file_name in tqdm(enumerate(file_names), desc="velodyne {}".format(sequence), total=len(file_names)):
#         print(file_name)
#         points_xyzi = np.fromfile(str(file_name), dtype=np.float32, count=-1).reshape([-1, 4])
    
#         # loading ply
#         o3d_util.save_ply_by_xyzi(points_xyzi, str(save_dir / "ply" / "{}.ply".format(file_name.stem)))
        
#         # loading velodyne_ground_removal
#         gr_points_xyzi = save_ground_removal_lidar_data(save_dir / "velodyne_ground_removal" / "{}.bin".format(file_name.stem), points_xyzi)
        
#         # loading velodyne_compression
#         save_compression_lidar_data(save_dir / "velodyne_compression" / "{}.bin".format(file_name.stem), gr_points_xyzi)
        
#         # loading packet
#         save_packet_data(save_dir / "velodyne_compression" / "{}.bin".format(file_name.stem), i*0.1, save_dir / "packet")
        
    
#     # loading location
#     data = np.loadtxt(pose_file_name)
#     nrows = data.shape[0]
#     ncols = 12
#     data_reshaped = data.reshape((nrows, ncols))
#     for i, data in tqdm(enumerate(data_reshaped), desc="location {}".format(sequence), total=data_reshaped.shape[0]):
#         x, y, z, roll, pitch, yaw = homogeneous_to_absolute(np.reshape(data, (3, 4)))
#         location = {"x": x, "y": y, "z": z, "rx": roll, "ry": pitch, "rz": yaw, "timestamp": i*0.1}
#         save_txt(save_dir / "location" / "{0:06}.txt".format(i), location)
#         np.save(save_dir / "location" / "{0:06}.npy".format(i), location)

def main():
    sequence_list = ["00"] # , "01", "02", "03", "04", "05", "06", "07", "08", "09", "10" , "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"
    for sequence in sequence_list:
        save_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences" / sequence
        pose_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/poses" / "{}.txt".format(sequence)
        
        my_util.create_dir(save_dir / "ply")
        my_util.create_dir(save_dir / "location")
        my_util.create_dir(save_dir / "packet")
        my_util.create_dir(save_dir / "velodyne_compression")
        my_util.create_dir(save_dir / "velodyne_ground_removal")
        
        file_names = sorted((save_dir / "velodyne").glob("*.bin"))
        
        # for i, file_name in tqdm(enumerate(file_names), desc="velodyne {}".format(sequence), total=len(file_names)):
        #     points_xyzi = np.fromfile(str(file_name), dtype=np.float32, count=-1).reshape([-1, 4])
        
        #     # loading ply
        #     o3d_util.save_ply_by_xyzi(points_xyzi, str(save_dir / "ply" / "{}.ply".format(file_name.stem)))
            
        #     # loading velodyne_ground_removal
        #     gr_points_xyzi = save_ground_removal_lidar_data(save_dir / "velodyne_ground_removal" / "{}.bin".format(file_name.stem), points_xyzi)
            
        #     # loading velodyne_compression
        #     save_compression_lidar_data(save_dir / "velodyne_compression" / "{}.bin".format(file_name.stem), gr_points_xyzi)
            
        #     # loading packet
        #     save_packet_data(save_dir / "velodyne_compression" / "{}.bin".format(file_name.stem), i*0.1, save_dir / "packet")
            
        
        # loading location
        data = np.loadtxt(pose_file_name)
        data_reshaped = data.reshape((data.shape[0], 12))
        for i, data in tqdm(enumerate(data_reshaped), desc="location {}".format(sequence), total=data_reshaped.shape[0]):
            # x, y, z, roll, pitch, yaw = homogeneous_to_absolute(np.reshape(data, (3, 4)))
            
            _x = data[3]
            _y = data[7]
            _z = data[11]

            _pitch = math.atan2(-data[9], math.sqrt(data[10] ** 2 + data[11] ** 2))
            _yaw = math.atan2(data[10], data[11])
            _roll = math.atan2(data[4], data[0])
            x, y, z, roll, pitch, yaw = _z, -_x, -_y, _yaw, -_roll, -_pitch
                        
            
            location = {"x": x, "y": y, "z": z, "rx": roll, "ry": pitch, "rz": yaw, "timestamp": i*0.1}
            
            # x, y, z = z, -x, -y # KITTI camera coordinate to lidar coordinate
            # roll, pitch, yaw = yaw, -roll, -pitch # KITTI camera coordinate to lidar coordinate
            # location = {"x": z, "y": -x, "z": -y, "rx": yaw, "ry": -roll, "rz": -pitch, "timestamp": i*0.1} # KITTI CAMERA TO LIDAR
            # location = {"x": -x, "y": z, "z": y, "rx": -roll, "ry": yaw, "rz": pitch, "timestamp": i*0.1} # KITTI CAMERA TO CARLA WORLD
            # location = {"x": -y, "y": x, "z": -z, "rx": -pitch, "ry": roll, "rz": -yaw, "timestamp": i*0.1} # KITTI WORLD TO CARLA WORLD
            save_txt(save_dir / "location" / "{0:06}.txt".format(i), location)
            np.save(save_dir / "location" / "{0:06}.npy".format(i), location)

# def multi_main():
#     exp_path = "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset"
#     sequence_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
#     with Pool(processes=None) as pool:
#         for sequence in sequence_list:
#             kitti2carla(exp_path, sequence)
#             # pool.apply_async(kitti2carla, (exp_path, sequence))
#         pool.close()
#         pool.join()


if __name__ == "__main__":
    main()

