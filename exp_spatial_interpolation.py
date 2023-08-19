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
import argparse
'''
****** README ******
xyzi: x, y, z, intensity
y: yaw
p: pitch
d: distance
a: azimuth
c: channel id
'''

# ------------------------- obsoleted algorithmes start -------------------------
# def power_law(x, C, alpha):
#     return C * x**(-alpha)

# def fun_likelihood_power_law(params, data):
#     C, alpha = params
#     loglik = np.sum(np.log(power_law(data[:, 2], C, alpha)))
#     return -loglik

# def fun_likelihood_uniform(params, data):
#     z_min, z_max = params
#     # z_min = data[:, -1].min()
#     # z_max = data[:, -1].max()
    
#     z_range = z_max - z_min
#     log_likelihood = -data.shape[0] * np.log(z_range)
#     for x, y, z in data:
#         if z_min <= z <= z_max:
#             log_likelihood -= np.log(z_range)
#         else:
#             log_likelihood = -np.inf
#             # assert False, "z_min: {}, z_max: {}, z: {}, params: {}".format(z_min, z_max, z, params)
#             break
#     return -log_likelihood


# def fun_maximum_likelihood(exp_path, packet_loss_rate, pro_name, args):
#     from scipy.optimize import minimize
    
#     lidar_range = args["lidar_range"]
#     lidar_coordinate = args["lidar_coordinate"]
#     azimuth_size = args["azimuth_size"]
#     channel_size = args["channel_size"]
#     v_fov = args["v_fov"]
#     v_fov_start = args["v_fov_start"]
#     v_fov_end = args["v_fov_end"]
    
#     h_fov_start = args["h_fov_start"]
#     h_fov_end = args["h_fov_end"]
#     angular_resolution = args["angular_resolution"]
    
#     save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
#     my_util.create_dir(save_dir)

#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         queue, reference_id_list = [], []
#         points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        
#         start_time = time.process_time()
        
#         added_points_ypdac = add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate)
        
#         # queue.append(points_xyzi) # [skewen]: check~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!
        
#         sector_size = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_size"]
#         sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
#         queue.append(my_util.split_sector_xyzi_by_sector_list(points_xyzi, sector_size, sector_list))
        
#         all_sector_list = range(sector_size)
#         drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
#         received_points_ypd = my_util.split_sector_ypd_by_sector_list(added_points_ypdac[:, :3], sector_size, sector_list)
        
#         received_points_ypd[np.where(received_points_ypd[:, 1] < 0), 1] += 360
        
#         print("received_points_ypd.shape: {}".format(received_points_ypd.shape))
        
#         dropped_points_yp = get_yps_by_sector_list(sector_size, drop_sector_list, azimuth_size, v_fov, v_fov_start, v_fov_end, channel_size)
        
#         dropped_points_yp[np.where(dropped_points_yp[:, 1] < 0), 1] += 360
        
#         print("dropped_points_yp.shape: {}, dropped sector shape: {}".format(dropped_points_yp.shape, len(drop_sector_list)))


#         # uniform distribution
#         # result = minimize(fun_likelihood, [0.0, lidar_range], args=(received_points_ypd,), bounds=((None, None), (None, None)))
#         # z_min, z_max = result.x
#         # predicted_d = []
#         # for x, y in dropped_points_yp:
#         #     z = (x + y) / 2 + (z_max - z_min) / 2
#         #     print(x, y, z)
#         #     predicted_d.append(z)
            
#         # power law distribution
#         result = minimize(fun_likelihood_power_law, [0.5, 5], args=(received_points_ypd,), method='Nelder-Mead')
#         C, alpha = result.x
#         print("C: {}, alpha: {}".format(C, alpha))
#         predicted_d = []
#         for x, y in dropped_points_yp:
#             z = C * (x**(-alpha)) * (y**(-alpha))
#             predicted_d.append(z)

#         predicted_points_ypd = np.concatenate((dropped_points_yp, np.array(predicted_d).reshape([-1, 1])), axis=1)
        
#         # remove out of range points
#         predicted_points_ypd = predicted_points_ypd[np.where((predicted_points_ypd[:, 2] >= 0) & (predicted_points_ypd[:, 2] < lidar_range))]
        
#         # for x in predicted_points_ypd:
#         #     print(x)
        
#         print("predicted_points_ypd.shape: {}".format(predicted_points_ypd.shape))
        
#         queue.append(np.concatenate((get_xyzs_by_ypds(predicted_points_ypd), -np.ones([predicted_points_ypd.shape[0], 1])), axis=1))
        
#         all_points_xyzi = np.concatenate(queue, axis=0)
#         running_time = time.process_time() - start_time
#         all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))
#         o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
#         np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "reference_id_list": {}, "time": time.time(), "running_time": running_time})

# def fun_ridge_regression(exp_path, packet_loss_rate, pro_name, args):
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.linear_model import RidgeCV
#     from sklearn.metrics import mean_squared_error
    
#     point_count = args["point_count"]
#     lidar_range = args["lidar_range"]
#     lidar_coordinate = args["lidar_coordinate"]
#     azimuth_size = args["azimuth_size"]
#     channel_size = args["channel_size"]
#     v_fov = args["v_fov"]
#     v_fov_start = args["v_fov_start"]
#     v_fov_end = args["v_fov_end"]
#     angular_resolution = args["angular_resolution"]
    
#     save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
#     my_util.create_dir(save_dir)

#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         queue = []
#         points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        
#         start_time = time.process_time()
        
#         added_points_ypdac = add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate)
        
#         sector_size = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_size"]
#         sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
#         queue.append(points_xyzi)
        
#         all_sector_list = range(sector_size)
#         drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
#         received_points_ypd = my_util.split_sector_ypd_by_sector_list(added_points_ypdac[:, :3], sector_size, sector_list)
        
#         # received_points_ypd[np.where(received_points_ypd[:, 1] < 0), 1] += 360
        
#         dropped_points_yp = get_yps_by_sector_list(sector_size, drop_sector_list, azimuth_size, v_fov, v_fov_start, v_fov_end, channel_size)
#         # dropped_points_yp[np.where(dropped_points_yp[:, 1] < 0), 1] += 360

#         X_train, X_test, y_train, y_test = train_test_split(received_points_ypd[:, :2], received_points_ypd[:, 2], test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
#         model = RidgeCV(alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cv=5)
#         model.fit(X_train, y_train)
        
#         # y_pred = model.predict(X_test)
#         # rmse = mean_squared_error(y_test, y_pred, squared=False)
#         # print("RMSE on test set: ", rmse)
        
#         predicted_d = model.predict(scaler.transform(dropped_points_yp))
        
#         # features = received_points_ypd[:, :2]
#         # output = received_points_ypd[:, -1]
#         # ridge = Ridge(alpha=0.5)
#         # ridge.fit(features, output)
#         # predicted_d = ridge.predict(dropped_points_yp)
        
#         predicted_points_ypd = np.concatenate((dropped_points_yp, np.array(predicted_d).reshape((-1, 1))), axis=1)
        
#         # remove out of range points
#         predicted_points_ypd = predicted_points_ypd[np.where((predicted_points_ypd[:, 2] > 0) & (predicted_points_ypd[:, 2] < lidar_range))]
        
#         queue.append(np.concatenate((get_xyzs_by_ypds(predicted_points_ypd), -np.ones([predicted_points_ypd.shape[0], 1])), axis=1))
        
#         all_points_xyzi = np.concatenate(queue, axis=0)
        
#         if point_count >= 0:
#             all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
            
#         running_time = time.process_time() - start_time
    
#         all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))

#         o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
#         np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "reference_id_list": {}, "time": time.time(), "running_time": running_time})
        
# def fun_lasso(exp_path, packet_loss_rate, pro_name, args):
#     from sklearn.linear_model import Lasso
    
#     lidar_range = args["lidar_range"]
#     lidar_coordinate = args["lidar_coordinate"]
#     azimuth_size = args["azimuth_size"]
#     channel_size = args["channel_size"]
#     v_fov = args["v_fov"]
#     v_fov_start = args["v_fov_start"]
#     v_fov_end = args["v_fov_end"]
#     angular_resolution = args["angular_resolution"]
    
#     save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
#     my_util.create_dir(save_dir)

#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         queue = []
#         points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        
#         start_time = time.process_time()
        
#         added_points_ypdac = add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate)
        
#         sector_size = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_size"]
#         sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
#         queue.append(my_util.split_sector_xyzi_by_sector_list(points_xyzi, sector_size, sector_list))
        
#         all_sector_list = range(sector_size)
#         drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
#         received_points_ypd = my_util.split_sector_ypd_by_sector_list(added_points_ypdac[:, :3], sector_size, sector_list)
        
#         received_points_ypd[np.where(received_points_ypd[:, 1] < 0), 1] += 360
        
#         dropped_points_yp = get_yps_by_sector_list(sector_size, drop_sector_list, azimuth_size, v_fov, v_fov_start, v_fov_end, channel_size)
#         dropped_points_yp[np.where(dropped_points_yp[:, 1] < 0), 1] += 360

#         features = received_points_ypd[:, :2]
#         output = received_points_ypd[:, -1]
#         lasso = Lasso(alpha=0.1)
#         lasso.fit(features, output)
        
#         predicted_d = z_new = lasso.predict(dropped_points_yp)
        
#         predicted_points_ypd = np.concatenate((dropped_points_yp, np.array(predicted_d).reshape((-1, 1))), axis=1)
        
#         # remove out of range points
#         predicted_points_ypd = predicted_points_ypd[np.where((predicted_points_ypd[:, 2] >= 0) & (predicted_points_ypd[:, 2] < lidar_range))]
        
#         queue.append(np.concatenate((get_xyzs_by_ypds(predicted_points_ypd), -np.ones([predicted_points_ypd.shape[0], 1])), axis=1))
        
#         all_points_xyzi = np.concatenate(queue, axis=0)
        
#         running_time = time.process_time() - start_time
    
#         all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))

#         o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
#         np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "reference_id_list": {}, "time": time.time(), "running_time": running_time})
# ------------------------- obsoleted algorithmes end -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--c', type=str, default="config.yaml")
    return parser.parse_args()

def get_angle_list(fov, start, end, length):
    unit = fov / (length - 1)
    return np.concatenate((np.arange(start, end, unit, dtype=float), [end]))

def get_cs_by_ps(points_p, v_fov, v_fov_start, v_fov_end, channel_size):
    pitch_list = get_angle_list(v_fov, v_fov_start, v_fov_end, channel_size)
    points_pc = np.concatenate((points_p, -np.ones((points_p.shape[0], 1))), axis=1)
    for i, pitch in enumerate(pitch_list):
        points_pc[np.where(((abs(points_pc[:, 0] - pitch) <= 0.1) & (points_pc[:, -1] == -1))), -1] = i
    return points_pc[:, -1]

def get_ypdacs_by_xyzis(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate=[0, 0, 0]):
    points_y = np.arctan(points_xyzi[:, 1] / points_xyzi[:, 0]) / np.pi * 180
    points_xyziy = np.concatenate((points_xyzi, points_y.reshape(-1, 1)), axis=1)
    points_xyziy[np.where((points_xyziy[:, 0] == 0.0) & (points_xyziy[:, 1] > 0)), 4] = 90.0
    points_xyziy[np.where((points_xyziy[:, 0] == 0.0) & (points_xyziy[:, 1] <= 0)), 4] = -90.0
    points_xyziy[np.where(points_xyziy[:, 0] < 0), 4] += 180
    points_xyziy[np.where(points_xyziy[:, 4] < 0), 4] += 360
    
    points_y = points_xyziy[:, -1].reshape(-1, 1)
    points_p = (np.arctan(points_xyzi[:, 2] / (np.sqrt(np.square(points_xyzi[:, 0]) + np.square(points_xyzi[:, 1])))) / np.pi * 180).reshape(-1, 1)
    points_d = np.sqrt(np.sum((points_xyzi[:, :3] - lidar_coordinate) ** 2, axis=1)).reshape(-1, 1)
    points_a = get_as_by_ys(points_y, azimuth_size, angular_resolution).reshape(-1, 1)
    points_c = get_cs_by_ps(points_p, v_fov, v_fov_start, v_fov_end, channel_size).reshape(-1, 1)
    
    assert points_c.any() != -1, "the channel id of points is -1"
    assert points_a.any() != -1, "the azimuth id of points is -1"
    
    return np.concatenate((points_y, points_p, points_d, points_a, points_c), axis=1) # [yaw, pitch, distance, azimuth, channel]


def get_xyzs_by_ypds(points_ypd):
    # yaw, pitch, distance
    points_xyz = np.zeros((points_ypd.shape[0], 3))
    points_xyz[:, 0] = points_ypd[:, 2] * np.cos(points_ypd[:, 0] / 180 * np.pi) * np.cos(points_ypd[:, 1] / 180 * np.pi)
    points_xyz[:, 1] = points_ypd[:, 2] * np.sin(points_ypd[:, 0] / 180 * np.pi) * np.cos(points_ypd[:, 1] / 180 * np.pi)
    points_xyz[:, 2] = points_ypd[:, 2] * np.sin(points_ypd[:, 1] / 180 * np.pi)
    return points_xyz # x, y, z


def get_xyz_by_ypd(yaw, pitch, dist):
    x = dist * np.cos(yaw  / 180 * np.pi) * np.cos(pitch  / 180 * np.pi)
    y = dist * np.sin(yaw  / 180 * np.pi) * np.cos(pitch  / 180 * np.pi)
    z = dist * np.sin(pitch / 180 * np.pi)
    return x, y, z

def get_as_by_ys(points_y, azimuth_size, angular_resolution):
    points_ya = np.concatenate((points_y.reshape(-1, 1), -np.ones((points_y.shape[0], 1))), axis=1)
    points_ya[:, -1] = (np.floor(points_ya[:, 0] / angular_resolution) + 1) % azimuth_size
    return points_ya[:,-1]

def get_statsk_n2_by_arr(arr1, arr2):
    A, B = np.meshgrid(arr1, arr2)
    return np.column_stack((A.ravel(), B.ravel()))

def get_yps_by_sector_list(sector_size, sector_list, azimuth_size, v_fov, v_fov_start, v_fov_end, channel_size):
    all_azimuth_list = get_angle_list(360.0, 0.0, 360.0, azimuth_size)
    pitch_list = get_angle_list(v_fov, v_fov_start, v_fov_end, channel_size)
    sector_range = 360 / sector_size
    azimuth_list = np.zeros((0, 1))
    for sector_id in sector_list:
        azimuth_list = np.concatenate((azimuth_list.reshape(-1, 1), all_azimuth_list[((all_azimuth_list >= sector_id*sector_range) & (all_azimuth_list < (sector_id+1)*sector_range))].reshape(-1, 1)))
    result = get_statsk_n2_by_arr(azimuth_list.reshape(-1, 1), pitch_list.reshape(-1, 1))
    return result

def add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate):
    points_ypdac = get_ypdacs_by_xyzis(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, [0,0,0])
    
    points_ac = np.concatenate((points_ypdac[:,3].reshape(-1, 1), points_ypdac[:, 4].reshape(-1, 1)), axis=1)

    unique_ids = np.unique(points_ac, axis=0, return_index=True)
    
    unique_points_ypdac = points_ypdac[unique_ids[1]]
    
    unique_points_ac = unique_ids[0]
    
    sorted_indices = np.lexsort((unique_points_ac[:, 1], unique_points_ac[:, 0]))
    
    unique_points_ac = unique_points_ac[sorted_indices]

    index = 0
    added_ypdacs = []
    for azimuth_id in range(0, azimuth_size):
        for channel_id in range(0, channel_size):
            if index >= unique_points_ac.shape[0] or unique_points_ac[index, 0] != azimuth_id or unique_points_ac[index, 1] != channel_id:
                added_ypdacs.append([angular_resolution/2*(azimuth_id+1), v_fov/channel_size/2*(channel_id+1), 120.0, azimuth_id, channel_id])
            else:
                index += 1
    added_points_ypdac = np.concatenate((unique_points_ypdac, np.array(added_ypdacs)), axis=0)
    return added_points_ypdac
        

def fun_nearest_neighbor(exp_path, packet_loss_rate, pro_name, args):
    from sklearn.neighbors import KDTree
    
    point_count = args["point_count"]
    lidar_range = args["lidar_range"]
    lidar_coordinate = args["lidar_coordinate"]
    azimuth_size = args["azimuth_size"]
    channel_size = args["channel_size"]
    v_fov = args["v_fov"]
    v_fov_start = args["v_fov_start"]
    v_fov_end = args["v_fov_end"]
    angular_resolution = args["angular_resolution"]
    
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
    my_util.create_dir(save_dir)
    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    file_names = sorted(none_dir.glob("*.bin"))
    assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        queue = []
        points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        
        start_time = time.process_time()
        
        added_points_ypdac = add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate)
        
        sector_size = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_size"]
        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
        queue.append(points_xyzi)
        
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
        received_points_ypd = my_util.split_sector_ypd_by_sector_list(added_points_ypdac[:, :3], sector_size, sector_list)
        
        
        dropped_points_yp = get_yps_by_sector_list(sector_size, drop_sector_list, azimuth_size, v_fov, v_fov_start, v_fov_end, channel_size)
        
        if dropped_points_yp.shape[0] > 0:
            tree_2d = KDTree(received_points_ypd[:, :2])
            distances, indices = tree_2d.query(dropped_points_yp)
            
            predicted_d = received_points_ypd[indices, 2]

            predicted_points_ypd = np.concatenate((dropped_points_yp, np.array(predicted_d).reshape((-1, 1))), axis=1)
            
            # remove out of range points
            predicted_points_ypd = predicted_points_ypd[np.where((predicted_points_ypd[:, 2] > 0) & (predicted_points_ypd[:, 2] < lidar_range))]
            
            queue.append(np.concatenate((get_xyzs_by_ypds(predicted_points_ypd), -np.ones([predicted_points_ypd.shape[0], 1])), axis=1))
        
        all_points_xyzi = np.concatenate(queue, axis=0)
        
        if point_count >= 0:
            all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
        
        running_time = time.process_time() - start_time
    
        all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))

        o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
        np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "reference_id_list": {}, "time": time.time(), "running_time": running_time})

def fun_least_square(exp_path, packet_loss_rate, pro_name, args):
    point_count = args["point_count"]
    lidar_range = args["lidar_range"]
    lidar_coordinate = args["lidar_coordinate"]
    azimuth_size = args["azimuth_size"]
    channel_size = args["channel_size"]
    v_fov = args["v_fov"]
    v_fov_start = args["v_fov_start"]
    v_fov_end = args["v_fov_end"]
    angular_resolution = args["angular_resolution"]
    
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
    my_util.create_dir(save_dir)

    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    file_names = sorted(none_dir.glob("*.bin"))
    assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        queue = []
        points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        
        start_time = time.process_time()
        
        added_points_ypdac = add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate)
        
        sector_size = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_size"]
        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
        queue.append(points_xyzi)
        
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
        received_points_ypd = my_util.split_sector_ypd_by_sector_list(added_points_ypdac[:, :3], sector_size, sector_list)
        
        # received_points_ypd[np.where(received_points_ypd[:, 1] < 0), 1] += 360
        
        dropped_points_yp = get_yps_by_sector_list(sector_size, drop_sector_list, azimuth_size, v_fov, v_fov_start, v_fov_end, channel_size)

        A = np.column_stack((received_points_ypd[:, 0], received_points_ypd[:, 1], np.ones_like(received_points_ypd[:, 0])))
        b = received_points_ypd[:, 2].reshape(-1, 1)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
        
        # dropped_points_yp[np.where(dropped_points_yp[:, 1] < 0), 1] += 360
        
        predicted_d = []
        for x, y in dropped_points_yp:
            z = (w[0]*x + w[1]*y + w[2])[0]
            # print(x, y, z)
            predicted_d.append(z)
        
        predicted_points_ypd = np.concatenate((dropped_points_yp, np.array(predicted_d).reshape((-1, 1))), axis=1)
        
        # remove out of range points
        predicted_points_ypd = predicted_points_ypd[np.where((predicted_points_ypd[:, 2] > 0) & (predicted_points_ypd[:, 2] < lidar_range))]
        
        
        queue.append(np.concatenate((get_xyzs_by_ypds(predicted_points_ypd), -np.ones([predicted_points_ypd.shape[0], 1])), axis=1))
        
        all_points_xyzi = np.concatenate(queue, axis=0)
        
        if point_count >= 0:
            all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
        
        running_time = time.process_time() - start_time
    
        all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))

        o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
        np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "reference_id_list": {}, "time": time.time(), "running_time": running_time})

def main(args):
    m_config = my_util.get_config(args.c)
    execution_name = "spatial_interpolation"
    exp_path = m_config["exp"]["path"]
    packet_loss_rates = m_config["exp"]["packet_loss_rates"]

    # multi process
    if m_config["exp"][execution_name]["process"] is not None:
        with Pool(processes=None) as pool:
            for k, v in m_config["exp"][execution_name]["process"].items():
                if v["enable"]:
                    print("multi-process {} {}...".format(execution_name, k))
                    for packet_loss_rate in packet_loss_rates:
                        pool.apply_async(globals()["fun_{}".format(k)], (exp_path, packet_loss_rate, k, copy.deepcopy(v)))
            pool.close()
            pool.join()
    
    # multi thread
    if m_config["exp"][execution_name]["thread"] is not None:
        thread_list = []
        for k, v in m_config["exp"][execution_name]["thread"].items():
            if v["enable"]:
                print("multi-thread {} {}...".format(execution_name, k))
                for packet_loss_rate in packet_loss_rates:
                    t = threading.Thread(target=globals()["fun_{}".format(k)], args=(exp_path, packet_loss_rate, k, copy.deepcopy(v),))
                    t.daemon = True
                    t.start()
                    thread_list.append(t)
        for t in thread_list:
            t.join()
    
    # single
    if m_config["exp"][execution_name]["single"] is not None:
        for k, v in m_config["exp"][execution_name]["single"].items():
            if v["enable"]:
                print("single {} {}...".format(execution_name, k))
                for packet_loss_rate in packet_loss_rates:
                    globals()["fun_{}".format(k)](exp_path, packet_loss_rate, k, copy.deepcopy(v))

    print("MAIN THREAD STOP")
    
if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print("Total Running Time: {}".format(time.time() - start_time))
