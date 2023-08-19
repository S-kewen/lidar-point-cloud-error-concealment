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

# ------------------------- obsoleted algorithmes start -------------------------
# def main_exp_w():
#     exp_path = m_config["exp"]["path"]
#     packet_loss_rates = m_config["exp"]["packet_loss_rates"]
#     # multi process
#     if m_config["exp"]["stitching"]["process"] is not None:
#         with Pool(processes=None) as pool:
#             for k, v in m_config["exp"]["stitching"]["process"].items():
#                 if k == "closest_both":
#                     print("multi-process stitching {}...".format(k))
#                     for packet_loss_rate in packet_loss_rates:
#                         for w in range(0, 105, 5): # [0, 1)
#                             v["weight"] = w/100
#                             v["is_exp_weight"]= True
#                             print("weight: {}".format(v["weight"]))
#                             pool.apply_async(globals()["fun_{}".format(k)], (exp_path, packet_loss_rate, k, copy.deepcopy(v)))
#             pool.close()
#             pool.join()

# def main_exp_a():
#     exp_path = m_config["exp"]["path"]
#     packet_loss_rates = m_config["exp"]["packet_loss_rates"]
#     with Pool(processes=8) as pool:
#         for k, v in m_config["exp"]["temporal_prediction"]["single"].items():
#             if k == "closest_spatial":
#                 print("multi-process temporal_prediction {}...".format(k))
#                 for packet_loss_rate in packet_loss_rates:
#                     for a in range(1, 14, 1): # [0, 1)
#                         v["interval"] = a
#                         v["is_exp_interval"]= True
#                         pool.apply_async(globals()["fun_{}".format(k)], (exp_path, packet_loss_rate, k, copy.deepcopy(v)))
#         pool.close()
#         pool.join()

# ------------------------- obsoleted algorithmes end -------------------------

# def get_sector_vertex(location, sector_size, sector_id, lidar_range, v_fov):
#     # location["x"], location["y"], location["z"] = 0.0, 0.0, 0.0
#     x1 = np.cos(360 / sector_size / 180 * np.pi) * lidar_range
#     y1 = np.sin(360 / sector_size / 180 * np.pi) * lidar_range

#     x2 = np.cos(v_fov / 180 * np.pi) * lidar_range * np.cos(360 / sector_size / 180 * np.pi)
#     y2 = np.cos(v_fov / 180 * np.pi) * lidar_range * np.sin(360 / sector_size / 180 * np.pi)

#     lidar_height = np.sin(v_fov / 180 * np.pi) * lidar_range
#     vertex_down = np.array([[location["x"] + lidar_range, location["y"], location["z"]],
#                             [location["x"] + x1, location["y"] + y1, location["z"]]])
#     vertex_top = np.array([[location["x"] + x2, location["y"], location["z"]],
#                            [location["x"] + x2, location["y"] + y2, location["z"]]])
#     vertex_top[:, 2] = vertex_top[:, 2] + lidar_height

#     point_list = np.concatenate((vertex_down, vertex_top), axis=0)
#     result_list = []

#     for v in point_list:
#         result_list.append(v)

#     result_list.append([location["x"], location["y"], location["z"]])

#     result_list = np.asarray(result_list)

#     result_list = o3d_util.translate_by_matrix(result_list, np.asarray([-location["x"], -location["y"], -location["z"]]))

#     result_list = o3d_util.rotate_by_matrix(result_list, o3d_util.get_rotation_matrix_from_angles(location["rx"], location["ry"], location["rz"] + 360 / sector_size * sector_id))

#     result_list = o3d_util.translate_by_matrix(result_list, np.asarray([location["x"], location["y"], location["z"]]))

#     return result_list

# def get_overlap_volume(vertex1, vertex2):
#     # https://stackoverflow.com/questions/70452216/how-to-find-the-intersection-of-2-convex-hulls#new-answer
#     from scipy.spatial import ConvexHull
#     import cdd as pcdd

#     v1 = np.column_stack((np.ones(5), vertex1))
#     mat = pcdd.Matrix(v1, number_type='fraction')  # use fractions if possible
#     mat.rep_type = pcdd.RepType.GENERATOR
#     poly1 = pcdd.Polyhedron(mat)

#     # make the V-representation of the second cube; you have to prepend
#     # with a column of ones
#     v2 = np.column_stack((np.ones(5), vertex2))
#     mat = pcdd.Matrix(v2, number_type='fraction')
#     mat.rep_type = pcdd.RepType.GENERATOR
#     poly2 = pcdd.Polyhedron(mat)
#     # H-representation of the first cube
#     h1 = poly1.get_inequalities()
#     # H-representation of the second cube
#     h2 = poly2.get_inequalities()
#     # join the two sets of linear inequalities; this will give the intersection
#     hintersection = np.vstack((h1, h2))
#     # make the V-representation of the intersection
#     mat = pcdd.Matrix(hintersection, number_type='fraction')
#     mat.rep_type = pcdd.RepType.INEQUALITY
#     polyintersection = pcdd.Polyhedron(mat)
#     # get the vertices; they are given in a matrix prepended by a column of ones
#     vintersection = polyintersection.get_generators()
#     # get rid of the column of ones
#     result = 0.0
#     if len(vintersection) > 0:
#         ptsintersection = np.array([vintersection[i][1:4] for i in range(len(vintersection))])
#         try:
#             result = ConvexHull(ptsintersection).volume
#         except:
#             result = 0.0

#     return result

# def fun_cal_verlap_volume(args):
#     dropped_sector = get_sector_vertex(args[0], args[4], args[1], args[5], args[6])
#     previous_sector = get_sector_vertex(args[2], args[4], args[3], args[5], args[6])
#     return get_overlap_volume(dropped_sector, previous_sector)

# def get_possible_sector_list(drop_sector, sector_size, interval):
#     sector_list = np.arange(0, sector_size)
#     idx = np.where(sector_list == drop_sector)[0][0]
#     left = max(0, idx - interval)
#     right = min(len(sector_list), idx + interval + 1)
#     result = sector_list[left:right]
#     if len(result) < 2*interval+1:
#         if left == 0:
#             result = np.concatenate([sector_list[-(2*interval+1 - len(result)):], result])
#         else:
#             result = np.concatenate([result, sector_list[0:2*interval+1 - len(result)]])
#     return result

# def fun_closest_spatial(exp_path, packet_loss_rate, pro_name, args):
#     recursion = args["recursion"]
#     point_count = args["point_count"]
#     sector_size = args["sector_size"]
#     align_location = args["align_location"]
#     align_rotation = args["align_rotation"]
#     max_sector_gap = args["max_sector_gap"]
#     lidar_range = args["lidar_range"]
#     v_fov = args["v_fov"]
#     interval = args["interval"]

#     if args["is_exp_interval"]:
#         save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}_{}".format(pro_name, packet_loss_rate, interval)
#     else:
#         save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
#     my_util.create_dir(save_dir)
#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
#     first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
#     save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         if i == 0:
#             continue
        
#         start_time = time.process_time()
        
#         queue, concealed_sector_list, reference_id_list = [], [], []
#         points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
#         queue.append(points_xyzi)

#         sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
#         npy_file_name = file_name.parent.parent.parent / "location" / "{}.npy".format(file_name.stem)
#         assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)

#         npy_location = np.load(npy_file_name, allow_pickle=True).item()

#         all_sector_list = range(sector_size)
#         drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
#         previous_id = int(file_names[i-1].stem)

#         for drop_sector in drop_sector_list:
#             sector_gap = 0
#             max_closest_val = 0.0
#             max_closest_points_xyzi = np.zeros((0, 4))
            
#             if recursion:
#                 previous_log_file_name = save_dir / "{0:06}_log.npy".format(previous_id)
#                 previous_bin_file_name = save_dir / "{0:06}.bin".format(previous_id)
#             else:
#                 previous_log_file_name = file_name.parent / "{0:06}_log.npy".format(previous_id)
#                 previous_bin_file_name = file_name.parent / "{0:06}.bin".format(previous_id)
                
#             assert previous_log_file_name.exists() and previous_bin_file_name.exists(), "previous file not exists: {}".format(previous_bin_file_name)
            
#             previous_location_file_name = file_name.parent.parent.parent / "location" / "{0:06}.npy".format(previous_id)

#             previous_location = np.load(previous_location_file_name, allow_pickle=True).item()
#             previous_points_xyzi = np.fromfile(previous_bin_file_name, dtype=np.float32, count=-1).reshape([-1, 4])
#             previous_sector_list = np.load(previous_log_file_name, allow_pickle=True).item()["sector_list"]
            
#             possible_sector_list = get_possible_sector_list(drop_sector, sector_size, interval)
#             for previous_sector in previous_sector_list:
#                 if not previous_sector in possible_sector_list:
#                     continue
#                 sector_gap += 1
#                 if sector_gap > max_sector_gap:
#                     break
#                 closest_spatial_val = get_overlap_volume(get_sector_vertex(npy_location, sector_size, drop_sector, lidar_range, v_fov), get_sector_vertex(previous_location, sector_size, previous_sector, lidar_range, v_fov))
                
#                 if closest_spatial_val > max_closest_val or max_closest_points_xyzi.shape[0] == 0:
#                     max_closest_val = closest_spatial_val
                    
#                     new_offset_xyzrs = np.asarray(np.array([npy_location["x"], npy_location["y"], npy_location["z"], npy_location["rx"], npy_location["ry"], npy_location["rz"]])-np.array([previous_location["x"], previous_location["y"], previous_location["z"], previous_location["rx"], previous_location["ry"], previous_location["rz"]]))
#                     if align_location or align_rotation:
#                         previous_points_xyzi = my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, previous_sector_list)
#                         if align_rotation:
#                             previous_points_xyzi = o3d_util.rotate_by_matrix(previous_points_xyzi, o3d_util.get_rotation_matrix_from_angles(new_offset_xyzrs[3], new_offset_xyzrs[4], new_offset_xyzrs[5]))
#                         if align_location:
#                             previous_points_xyzi = o3d_util.translate_by_matrix(previous_points_xyzi, new_offset_xyzrs[0:3])


#                     max_closest_points_xyzi = o3d_util.rotate_by_matrix(my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, [previous_sector]), o3d_util.get_rotation_matrix_from_angles(0, 0, 360/sector_size*(abs(drop_sector-previous_sector))))
#                     reference_id = {"drop_sector": drop_sector, "reference_frame": previous_id, "reference_sector": previous_sector, "default": False}
                        
#             if max_closest_points_xyzi.shape[0] > 0:
#                 queue.append(max_closest_points_xyzi)
#                 reference_id_list.append(reference_id)
#                 concealed_sector_list.append(drop_sector)
                                
#         all_points_xyzi = np.concatenate(queue, axis=0)
        
#         if point_count >= 0:
#             all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
        
#         all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))
        
#         o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
#         running_time = time.process_time() - start_time
#         np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "reference_id_list": reference_id_list, "time": time.time(), "running_time": running_time})  


# def fun_closest_both(exp_path, packet_loss_rate, pro_name, args):
#     recursion = args["recursion"]
#     point_count = args["point_count"]
#     sector_size = args["sector_size"]
#     align_location = args["align_location"]
#     align_rotation = args["align_rotation"]
#     max_sector_gap = args["max_sector_gap"]
#     lidar_range = args["lidar_range"]
#     v_fov = args["v_fov"]
#     weight = args["weight"]
#     interval = args["interval"]
    
    
#     if args["is_exp_weight"]:
#         save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}_{}".format(pro_name, packet_loss_rate, weight)
#     else:
#         save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
#     my_util.create_dir(save_dir)

#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
#     first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
#     save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         if i == 0:
#             continue
        
#         start_time = time.process_time()
        
#         queue, concealed_sector_list, reference_id_list = [], [], []
        
#         points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
#         queue.append(points_xyzi)

#         sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        
#         npy_file_name = file_name.parent.parent.parent / "location" / "{}.npy".format(file_name.stem)
#         assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)

#         npy_location = np.load(npy_file_name, allow_pickle=True).item()

#         all_sector_list = range(sector_size)
#         drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
#         previous_id = int(file_names[i-1].stem)

#         for drop_sector in drop_sector_list:
#             reference_info = None
#             max_closest_val = 0.0
#             max_closest_points_xyzi = np.zeros((0, 4))
#             sector_gap = 0

#             if recursion:
#                 previous_log_file_name = save_dir / "{0:06}_log.npy".format(previous_id)
#                 previous_bin_file_name = save_dir / "{0:06}.bin".format(previous_id)
#             else:
#                 previous_log_file_name = file_name.parent / "{0:06}_log.npy".format(previous_id)
#                 previous_bin_file_name = file_name.parent / "{0:06}.bin".format(previous_id)
                
#             assert previous_log_file_name.exists() and previous_bin_file_name.exists(), "previous file not exists: {}".format(previous_bin_file_name)
            
#             previous_points_xyzi = np.fromfile(previous_bin_file_name, dtype=np.float32, count=-1).reshape([-1, 4])
#             previous_sector_list = np.load(previous_log_file_name, allow_pickle=True).item()["sector_list"]
            
#             possible_sector_list = get_possible_sector_list(drop_sector, sector_size, interval)
#             for previous_sector in previous_sector_list:
#                 if not previous_sector in possible_sector_list:
#                     continue
#                 sector_gap += 1
#                 if sector_gap > max_sector_gap:
#                     break
#                 previous_location_file_name = file_name.parent.parent.parent / "location" / "{0:06}.npy".format(previous_id)
#                 assert npy_file_name.exists(), "npy file not exists: {}".format(previous_location_file_name)
                
#                 previous_location = np.load(previous_location_file_name, allow_pickle=True).item()

#                 closest_time_val = - abs((npy_location["timestamp"] + 1/sector_size*drop_sector) - (previous_location["timestamp"] + 1/sector_size*previous_sector)) * 1000
                
#                 closest_spatial_val = get_overlap_volume(get_sector_vertex(npy_location, sector_size, drop_sector, lidar_range, v_fov), get_sector_vertex(previous_location, sector_size, previous_sector, lidar_range, v_fov))
                
#                 closest_val = (closest_time_val * weight) + (closest_spatial_val * (1 - weight))
                
#                 if closest_val > max_closest_val:
#                     max_closest_val = closest_val
#                     new_offset_xyzrs = np.asarray(np.array([npy_location["x"], npy_location["y"], npy_location["z"], npy_location["rx"], npy_location["ry"], npy_location["rz"]])-np.array([previous_location["x"], previous_location["y"], previous_location["z"], previous_location["rx"], previous_location["ry"], previous_location["rz"]]))
                    
#                     if align_location or align_rotation:
#                         previous_points_xyzi = my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, previous_sector_list)
#                         if align_rotation:
#                             previous_points_xyzi = o3d_util.rotate_by_matrix(previous_points_xyzi, o3d_util.get_rotation_matrix_from_angles(new_offset_xyzrs[3], new_offset_xyzrs[4], new_offset_xyzrs[5]))
#                         if align_location:
#                             previous_points_xyzi = o3d_util.translate_by_matrix(previous_points_xyzi, new_offset_xyzrs[0:3])

                    
#                     max_closest_points_xyzi = o3d_util.rotate_by_matrix(my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, [drop_sector]), o3d_util.get_rotation_matrix_from_angles(0, 0, 360/sector_size*(abs(drop_sector-previous_sector))))
#                     reference_info = {"drop_sector": drop_sector, "reference_frame": previous_id, "reference_sector": previous_sector, "default": False}
                            
#             if reference_info is not None:
#                 queue.append(max_closest_points_xyzi)
#                 reference_id_list.append(reference_info)
#                 concealed_sector_list.append(drop_sector)
          
            
#         all_points_xyzi = np.concatenate(queue, axis=0)
        
#         if point_count >= 0:
#             all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
        
#         all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))
        
#         o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
#         running_time = time.process_time() - start_time
#         np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + concealed_sector_list, "concealed_sector_list": concealed_sector_list, "reference_id_list": reference_id_list, "time": time.time(), "running_time": running_time})

# ------------------------- obsoleted algorithmes end -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--c', type=str, default="config.yaml")
    return parser.parse_args() 
    

def save_first_frame(file_name, point_count, save_dir, sector_size):
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    if point_count >= 0:
        points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(points_xyzi, point_count)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))

    o3d_util.save_ply_by_xyzi(points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))

    np.save(save_dir / "{}_log.npy".format(file_name.stem), {"id": 0, "sector_size": sector_size, "sector_list": list(range(sector_size)), "point_count": point_count, "time": time.time(), "running_time": 0.0})


def fun_copyover_prediction(exp_path, packet_loss_rate, pro_name, args):
    # 1. init the parameters from config.yaml
    recursion = args["recursion"]
    point_count = args["point_count"]
    sector_size = args["sector_size"]

    # 2. init the path
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
    my_util.create_dir(save_dir)
    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    # 3. loading the data
    file_names = sorted(none_dir.glob("*.bin"))
    assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
    first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
    save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
    # 4. start the processing
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        if i == 0:
            continue
        
        start_time = time.process_time()
        
        queue = []
        points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4]) # the received points from the none floder
        queue.append(points_xyzi)

        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]

        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
        previous_id = int(file_names[i-1].stem)
        
        if recursion:
            previous_log_file_name = save_dir / "{0:06}_log.npy".format(previous_id)
            previous_bin_file_name = save_dir / "{0:06}.bin".format(previous_id)
        else:
            previous_log_file_name = file_name.parent / "{0:06}_log.npy".format(previous_id)
            previous_bin_file_name = file_name.parent / "{0:06}.bin".format(previous_id)
            
        assert previous_log_file_name.exists() and previous_bin_file_name.exists() , "previous file not exists: {}".format(previous_bin_file_name)
        
        previous_points_xyzi = np.fromfile(previous_bin_file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        queue.append(my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, drop_sector_list))

        all_points_xyzi = np.concatenate(queue, axis=0)
        
        if point_count >= 0:
            all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
    
        all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))

        o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
        
        running_time = time.process_time() - start_time
        np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "time": time.time(), "running_time": running_time})




def fun_motion_compensated_prediction(exp_path, packet_loss_rate, pro_name, args):
    recursion = args["recursion"]
    point_count = args["point_count"]
    sector_size = args["sector_size"]
    
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate)
    my_util.create_dir(save_dir)

    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)
    file_names = sorted(none_dir.glob("*.bin"))
    assert len(file_names) > 0, "bin files not exists: {}".format(none_dir)
    
    first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
    save_first_frame(first_file_name, point_count, save_dir, sector_size)

    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        if i == 0:
            continue
        
        start_time = time.process_time()

        queue= []
        
        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]

        
        points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        queue.append(points_xyzi)

        npy_file_name = file_name.parent.parent.parent / "location" / "{}.npy".format(file_name.stem)
        
        assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)

        npy_location = np.load(npy_file_name, allow_pickle=True).item()
        
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))

        previous_id = int(file_names[i-1].stem)
        
        if recursion:
            previous_log_file_name = save_dir / "{0:06}_log.npy".format(previous_id)
            previous_bin_file_name = save_dir / "{0:06}.bin".format(previous_id)
        else:
            previous_log_file_name = file_name.parent / "{0:06}_log.npy".format(previous_id)
            previous_bin_file_name = file_name.parent / "{0:06}.bin".format(previous_id)
            
        previous_locaton_file_name = file_name.parent.parent.parent / "location" / "{0:06}.npy".format(previous_id)
        
        assert previous_log_file_name.exists() and previous_bin_file_name.exists() and previous_locaton_file_name.exists() , "previous file not exists: {}".format(previous_bin_file_name)
        
        previous_points_xyzi = np.fromfile(previous_bin_file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        previous_location = np.load(previous_locaton_file_name, allow_pickle=True).item()
        

        new_offset_xyzrs = np.asarray(np.array([npy_location["x"], npy_location["y"], npy_location["z"], npy_location["rx"], npy_location["ry"], npy_location["rz"]])-np.array([previous_location["x"], previous_location["y"], previous_location["z"], previous_location["rx"], previous_location["ry"], previous_location["rz"]]))
                
        # new_offset_xyzrs = new_offset_xyzrs * -1 # Instead of aligning with the lidar center, it moves relatively
        
        previous_points_xyzi = my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, drop_sector_list)
        queue.append(my_util.split_sector_xyzi_by_sector_list(o3d_util.translate_by_matrix(o3d_util.rotate_by_matrix(previous_points_xyzi, o3d_util.get_rotation_matrix_from_angles(new_offset_xyzrs[3], new_offset_xyzrs[4], new_offset_xyzrs[5])), new_offset_xyzrs[0:3]), sector_size, drop_sector_list))
        
        all_points_xyzi = np.concatenate(queue, axis=0)
        
        
        if point_count >= 0:
            all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)

        all_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))
        
        o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))

        running_time = time.process_time() - start_time
        np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "time": time.time(), "running_time": running_time})


def main(args):
    m_config = my_util.get_config(args.c)
    execution_name = "temporal_prediction"
    exp_path = m_config["exp"]["path"]
    packet_loss_rates = m_config["exp"]["packet_loss_rates"]

    # multi process
    if m_config["exp"][execution_name]["process"] is not None:
        with Pool(processes=None) as pool:
            for k, v in m_config["exp"][execution_name]["process"].items():
                if v["enable"]:
                    print("multi-process {} {}...".format(execution_name, k))
                    for packet_loss_rate in packet_loss_rates:
                        #globals()["fun_{}".format(k)](exp_path, packet_loss_rate, k, v)
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
                    # globals()["fun_{}".format(k)](exp_path, packet_loss_rate, k, v)
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
                    globals()["fun_{}".format(k)](exp_path, packet_loss_rate,k, copy.deepcopy(v))

    print("MAIN THREAD STOP")

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print("Total Running Time: {}".format(time.time() - start_time))