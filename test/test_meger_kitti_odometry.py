import time
import numpy as np
import time
from multiprocessing import Pool
import threading
from pathlib import Path
from tqdm import tqdm
from o3d_util import O3dUtil as o3d_util
from my_util import MyUtil as my_util
from o3d_sampling import O3dSampling as o3d_sampling
import copy



sector_size = 180

# previous_bin_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230413_False_1_6_110_0_600_1/object/training/velodyne_compression/000000.bin"
# previous_location_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230413_False_1_6_110_0_600_1/object/training/location/000000.npy"
# file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230413_False_1_6_110_0_600_1/object/training/velodyne_compression/000010.bin"
# npy_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230413_False_1_6_110_0_600_1/object/training/location/000010.npy"

previous_bin_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00/velodyne_compression/000000.bin"
previous_location_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00/location/000000.npy"
file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00/velodyne_compression/000010.bin"
npy_file_name = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00/location/000010.npy"

save_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00"

assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)
assert previous_location_file_name.exists(), "npy file not exists: {}".format(previous_location_file_name)



start_time = time.process_time()
queue= []
points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
queue.append(points_xyzi)



npy_location = np.load(npy_file_name, allow_pickle=True).item()




previous_points_xyzi = np.fromfile(previous_bin_file_name, dtype=np.float32, count=-1).reshape([-1, 4])
previous_location = np.load(previous_location_file_name, allow_pickle=True).item()


new_offset_xyzrs = np.asarray(np.array([npy_location["x"], npy_location["y"], npy_location["z"], npy_location["rx"], npy_location["ry"], npy_location["rz"]])-np.array([previous_location["x"], previous_location["y"], previous_location["z"], previous_location["rx"], previous_location["ry"], previous_location["rz"]]))
        
# queue.append(o3d_util.translate_by_matrix(o3d_util.rotate_by_matrix(previous_points_xyzi, o3d_util.get_rotation_matrix_from_angles(new_offset_xyzrs[3], new_offset_xyzrs[4], new_offset_xyzrs[5])), new_offset_xyzrs[0:3]))
print(new_offset_xyzrs)
queue.append(o3d_util.translate_by_matrix(o3d_util.rotate_by_matrix(previous_points_xyzi, o3d_util.get_rotation_matrix_from_angles(new_offset_xyzrs[3], new_offset_xyzrs[4], new_offset_xyzrs[5])), -new_offset_xyzrs[0:3]))
# queue.append(o3d_util.translate_by_matrix(previous_points_xyzi, new_offset_xyzrs[0:3]))

all_points_xyzi = np.concatenate(queue, axis=0)

all_points_xyzi.astype(np.float32).tofile(save_dir / "{}_output.bin".format(file_name.stem))

o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}_output.ply".format(file_name.stem)))

running_time = time.process_time() - start_time
print("TOTAL TIME: {}".format(running_time))
