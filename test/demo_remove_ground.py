import open3d as o3d
import numpy as np
import time
from pathlib import Path
from multiprocessing import Pool
import threading
from multiprocessing import Process

def remove_ground_by_ply():
    pcd = o3d.io.read_point_cloud("/mnt/data2/skewen/datasets/20221223_50_0_TRUE_200/object/training/ply/000000.ply")
    pcd_filtered = pcd.voxel_down_sample(voxel_size=0.1)
    o3d.io.write_point_cloud("pcd_filtered.ply", pcd_filtered)
    all_indexs = np.arange(len(pcd_filtered.points))
    [planes, ground_indexs] = pcd_filtered.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=1000)


    non_ground_indexs = list(set(all_indexs) - set(ground_indexs))

    pcd_ground = pcd_filtered.select_by_index(ground_indexs)

    print("all: {}, ground: {}, non-ground: {}".format(len(all_indexs), len(ground_indexs), len(non_ground_indexs)))

    o3d.io.write_point_cloud("point_cloud_ground.ply", pcd_ground)

    pcd_ground = pcd_filtered.select_by_index(non_ground_indexs)
    o3d.io.write_point_cloud("point_cloud_non_ground.ply", pcd_ground)

def remove_ground_by_bin(file_name, save_dir):
    print(file_name)
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
                (points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    pcd_filtered = point_list.voxel_down_sample(voxel_size=0.1)

    all_indexs = np.arange(len(pcd_filtered.points))
    [planes, ground_indexs] = pcd_filtered.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=1000)

    non_ground_indexs = list(set(all_indexs) - set(ground_indexs))

    pcd_non_ground = pcd_filtered.select_by_index(non_ground_indexs)
    
    result = np.concatenate((np.asarray(pcd_non_ground.points), np.asarray(pcd_non_ground.normals)[:, 0].reshape(-1, 1)), axis=1)

    result.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))

def main():
    dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230203_20_1_50_0_50/object/training/velodyne"
    save_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230203_20_1_50_0_50/object/training/velodyne_remove_ground"
    if save_dir.exists() == False:
        save_dir.mkdir(parents=True)
        
    file_names = sorted(dir.glob("*.bin"))
    
    for file_name in file_names:
        remove_ground_by_bin(file_name, save_dir)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("time: ", time.time() - start_time)