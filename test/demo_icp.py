
import time
from o3d_util import O3dUtil as o3d_util
from o3d_sampling import O3dSampling as o3d_sampling

from pathlib import Path
from multiprocessing import Pool
import threading
import open3d as o3d
from tqdm import tqdm
import copy
import argparse


import numpy as np
from scipy.spatial.transform import Rotation as R
from my_util import MyUtil as my_util

def get_half22(transformation_matrix):
    # Extract the rotation and translation components from the transformation matrix
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]

    # Divide the rotation angles by 2 to reduce the rotation by half
    rotation_angles = np.degrees(np.arccos((np.trace(rotation_matrix) - 1) / 2))
    
    print("rotation_angles: {}".format(rotation_angles.shape))
    rotation_matrix_half = np.dot(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1 / 2]]), rotation_matrix)

    # Divide the translation vector by 2 to reduce the translation by half
    translation_vector_half = translation_vector / 2

    # Combine the rotation and translation components back into a 4x4 transformation matrix
    transformation_matrix_half = np.zeros((4, 4))
    transformation_matrix_half[:3, :3] = rotation_matrix_half
    transformation_matrix_half[:3, 3] = translation_vector_half
    
    transformation_matrix_half[3, 3] = 1

    return transformation_matrix_half

def get_half33(transformation_matrix):
    # Extract the rotation and translation components from the transformation matrix
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]

    # Divide the rotation angles by 2 to reduce the rotation by half
    rotation_angles = np.degrees(np.arccos((np.trace(rotation_matrix) - 1) / 2))
    
    print("rotation_angles: {}".format(rotation_angles.shape))
    rotation_matrix_half = np.dot(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1 / 2]]), rotation_matrix)

    # Divide the translation vector by 2 to reduce the translation by half
    translation_vector_half = translation_vector / 2

    # Combine the rotation and translation components back into a 4x4 transformation matrix
    transformation_matrix_half = np.zeros((4, 4))
    transformation_matrix_half[:3, :3] = rotation_matrix_half
    transformation_matrix_half[:3, 3] = translation_vector_half
    
    transformation_matrix_half[3, 3] = 1

    return transformation_matrix_half


def rotation_matrix_to_euler_angles(R):
    # 计算旋转矩阵的欧拉角
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    # 判断旋转矩阵是否为奇异矩阵，若为奇异矩阵，则返回异常
    if sy < 1e-6:
        x = np.arctan2(R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])

    # 将弧度转换为角度
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)

    return [x, y, z]


def get_half(transformation):
    result = np.zeros((4, 4))
    
    r_transformation = transformation[:3, :3]
    
    
    theta = np.arccos((np.trace(r_transformation) - 1) / 2)

    theta_new = theta / 2

    v = np.array([r_transformation[2, 1] - r_transformation[1, 2], r_transformation[0, 2] - r_transformation[2, 0], r_transformation[1, 0] - r_transformation[0, 1]])

    v_cross = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    R_new = np.cos(theta_new) * np.identity(3) + (1 - np.cos(theta_new)) * np.outer(v, v) + np.sin(theta_new) * v_cross
    
    result[:3, :3] = R_new
    
    result[:3, 3] = transformation[:3, 3] / 2
    
    result[3, 3] = 1
    
    return R_new



def get_half(transformation):
    print(transformation/2)
    
    result = np.zeros((4, 4))
    
    r = R.from_matrix(transformation[:3, :3]) # 4x4
    euler_degree = r.as_euler('xyz', degrees=True) / 2
    
    result[:3, :3] = R.from_euler('xyz', euler_degree, degrees=True).as_matrix()
    
    result[:3, 3] = transformation[:3, 3] / 2
    
    result[3, 3] = 1
    
    print(result)
    
    return result

def get_half_last(mat):
    result = np.zeros((4, 4))
    
    r = R.from_matrix(mat[:3, :3]) # 4x4
    
    euler_angles = r.as_euler('xyz', degrees=True)
    
    print("euler_angles: {}".format(euler_angles))
    
    print("euler_angles DIV 2: {}".format(euler_angles/2))
    
    result[:3, :3] = R.from_euler('xyz', euler_angles/2, degrees=True).as_matrix()
    result[:3, 3] = mat[:3, 3] / 2
    result[3, 3] = 1
    
    return result

       
 
def get_half_direly(transformation):
    result = np.zeros((4, 4))
    result[:3, :] = transformation[:3, :] / 2
    
    result[3, 3] = 1
    
    return result

  
        
def main():
    from o3d_icp import GlobalRegistration
    
    save_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/test_False_1_1_140_0_100/object/training/test"
    
    point_count = -1
    sector_size = 180
    icp_voxel_size = 0.75

    queue = []
    
    file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/test_False_1_1_140_0_100/object/training/cache_frame/receiver_none_ns3_nr_c_v2x_1/000001.bin"
    fn1_bin = Path() / "/mnt/data2/skewen/kittiGenerator/output/test_False_1_1_140_0_100/object/training/cache_frame/receiver_none_ns3_nr_c_v2x_1/000000.bin"
    fn2_bin = Path() / "/mnt/data2/skewen/kittiGenerator/output/test_False_1_1_140_0_100/object/training/cache_frame/receiver_none_ns3_nr_c_v2x_1/000002.bin"

    mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)
    
    current_frame_pc =  np.fromfile(fn2_bin, dtype=np.float32, count=-1).reshape([-1, 4])
    queue.append(current_frame_pc)
    sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
    all_sector_list = range(sector_size)
    drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))

    first_pc = np.fromfile(fn1_bin, dtype=np.float32, count=-1).reshape([-1, 4])  # xyzi
    last_pc = np.fromfile(fn2_bin, dtype=np.float32, count=-1).reshape([-1, 4])  # xyzi

    target_point_list = o3d.geometry.PointCloud()
    target_point_list.points = o3d.utility.Vector3dVector(last_pc[:, :3])
    target_point_list.colors = o3d.utility.Vector3dVector(o3d_util.get_point_color(last_pc[:, 3].reshape(-1, 1)))
    target_point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
        (last_pc[:, 3].reshape(-1, 1), np.zeros((last_pc.shape[0], 2))), axis=1))
    # if len(target_point_list.points) > 16384:
    #     target_point_list = o3d.geometry.PointCloud.uniform_down_sample(target_point_list, int(len(target_point_list.points) / 16384))

    source_point_list = o3d.geometry.PointCloud()
    source_point_list.points = o3d.utility.Vector3dVector(first_pc[:, :3])
    source_point_list.colors = o3d.utility.Vector3dVector(
        o3d_util.get_point_color(first_pc[:, 3].reshape(-1, 1)))
    source_point_list.normals = o3d.utility.Vector3dVector(np.concatenate(
        (first_pc[:, 3].reshape(-1, 1), np.zeros((first_pc.shape[0], 2))), axis=1))
    # if len(source_point_list.points) > 16384:
    #     source_point_list = o3d.geometry.PointCloud.uniform_down_sample(source_point_list, int(len(source_point_list.points) / 16384))

    source_point_list, target_point_list, source_down, target_down, source_fpfh, target_fpfh = GlobalRegistration.prepare_dataset(source_point_list, target_point_list, icp_voxel_size)  # loading two point cloud frame and random init position

    result_ransac = GlobalRegistration.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, icp_voxel_size)

    result_icp = GlobalRegistration.refine_registration(source_point_list, target_point_list, source_fpfh, target_fpfh, icp_voxel_size, result_ransac.transformation)
    
    print(result_icp.transformation)
    
    
    
    # old div2

    # source_point_list = source_point_list.transform(result_icp.transformation/2)
    
    # print(result_icp.transformation/2)
    
    
    # new div2 by tang
    
    source_point_list = source_point_list.transform(get_half22(result_icp.transformation))
    
    # new div2
    
    # source_point_list = source_point_list.transform(get_half_direly(result_icp.transformation))

    pred_mid_pc = np.concatenate((np.asarray(source_point_list.points), np.asarray(source_point_list.normals)[:, 0].reshape((-1, 1))), axis=1)
    
    queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
    
    all_points_xyzi = np.concatenate(queue, axis=0)
    
    if point_count != -1:
        all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
    
    all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}_last.bin".format(mid_frame_number)))
    
    o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}_last.ply".format(mid_frame_number)))

  
        

if __name__ == "__main__":
    start_time = time.process_time()
    main()
    print("Total Running Time: {}".format(time.process_time() - start_time))