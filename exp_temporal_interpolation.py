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
from my_util import MyUtil as my_util

# ------------------------- obsoleted algorithmes start -------------------------
# def fun_identity(exp_path, packet_loss_rate, pro_name, args):
#     point_count = args["point_count"]
#     sector_size = args["sector_size"]

#     save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
#     my_util.create_dir(save_dir)
    
#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     file_count = len(file_names)
#     assert file_count > 0, "bin files not exists: {}".format(none_dir)
    
#     first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
#     save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         if i + 2 >= file_count:
#             continue
#         start_time = time.process_time()
#         fn1_bin = file_name
#         fn2_bin = file_names[i + 2]

#         mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)

#         pred_mid_pc = np.fromfile(fn1_bin, dtype=np.float32, count=-1).reshape([-1, 4])
        
        
#         if point_count != -1:
#             pred_mid_pc = o3d_sampling.adaptive_sampling_by_xyzi(pred_mid_pc, point_count)
        
#         pred_mid_pc.astype(np.float32).tofile(str(save_dir / "{}.bin".format(mid_frame_number)))
        
#         o3d_util.save_ply_by_xyzi(pred_mid_pc, str(save_dir / "{}.ply".format(mid_frame_number)))
#         running_time = time.process_time() - start_time
#         np.save(save_dir / "{}_log.npy".format(mid_frame_number), {"id": i + 1, "point_count": point_count, "first_frame": str(fn1_bin), "mid_frame": str(save_dir / "{}.bin".format(mid_frame_number)), "last_frame": str(fn2_bin), "time": time.time(), "running_time": running_time})


# def fun_triangular(exp_path, packet_loss_rate, pro_name, args):
#     from sklearn.neighbors import KDTree
#     point_count = args["point_count"]
#     sector_size = args["sector_size"]
#     k = args["k"]
    
#     save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
#     my_util.create_dir(save_dir)
    
#     none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

#     file_names = sorted(none_dir.glob("*.bin"))
#     file_count = len(file_names)
#     assert file_count > 0, "bin files not exists: {}".format(none_dir)
    
#     first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
#     save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
#     for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
#         if i + 2 >= file_count:
#             continue
        
#         queue = []
#         fn1_bin = file_name
#         fn2_bin = file_names[i + 2]

#         mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)
        
#         current_frame_pc =  np.fromfile(file_names[i+1], dtype=np.float32, count=-1).reshape([-1, 4])
#         queue.append(current_frame_pc)
#         sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
#         all_sector_list = range(sector_size)
#         drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))

#         fn1_pc = np.fromfile(fn1_bin, dtype=np.float32, count=-1).reshape([-1, 4])
#         fn2_pc = np.fromfile(fn2_bin, dtype=np.float32, count=-1).reshape([-1, 4])
        
#         start_time = time.process_time()
        
#         kd_tree_fn1 = KDTree(fn1_pc[:, :3])
#         dist, indexs_fn1 = kd_tree_fn1.query(fn1_pc[:, :3], k=k)
        
#         kd_tree_fn2 = KDTree(fn2_pc[:, :3])
#         indexs = []
#         for i in indexs_fn1:
#             dist, indexs_fn2 = kd_tree_fn2.query(fn1_pc[i, :3], k=1)
#             indexs.append(indexs_fn2[np.argmin(dist)].reshape(-1))

#         most_frequent_indexs = np.asarray(indexs).reshape(-1)
#         aligned_fn2_pc = fn2_pc[most_frequent_indexs, :]
#         pred_mid_pc = (fn1_pc + aligned_fn2_pc) / 2
        
#         queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
        
#         all_points_xyzi = np.concatenate(queue, axis=0)
        
#         if point_count != -1:
#             all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
        
#         all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(mid_frame_number)))
        
#         o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(mid_frame_number)))

#         running_time = time.process_time() - start_time
#         np.save(save_dir / "{}_log.npy".format(mid_frame_number), {"id": i + 1, "first_frame": str(fn1_bin), "mid_frame": str(save_dir / "{}.bin".format(mid_frame_number)), "last_frame": str(fn2_bin), "time": time.time(), "running_time": running_time})

# ------------------------- obsoleted algorithmes end -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--c', type=str, default="config.yaml")
    return parser.parse_args()

def get_half(transformation):
    result = np.zeros((4, 4))
    result[:3, :] = transformation[:3, :] / 2
    result[3, 3] = 1
    return result

def save_first_frame(file_name, point_count, save_dir, sector_size):
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    if point_count >= 0:
        points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(points_xyzi, point_count)

    points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))

    o3d_util.save_ply_by_xyzi(points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))

    np.save(save_dir / "{}_log.npy".format(file_name.stem), {"id": 0, "sector_size": sector_size, "sector_list": list(range(sector_size)), "point_count": point_count, "time": time.time(), "running_time": 0.0})


def fun_point_matching(exp_path, packet_loss_rate, pro_name, args):
    from sklearn.neighbors import KDTree
    
    point_count = args["point_count"]
    sector_size = args["sector_size"]
    k = args["k"]
    
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
    my_util.create_dir(save_dir)
    
    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    file_names = sorted(none_dir.glob("*.bin"))
    file_count = len(file_names)
    assert file_count > 0, "bin files not exists: {}".format(none_dir)
    
    first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
    save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        if i + 2 >= file_count:
            continue
        
        queue = []
        fn1_bin = file_name
        fn2_bin = file_names[i + 2]

        mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)
        
        
        current_frame_pc =  np.fromfile(file_names[i+1], dtype=np.float32, count=-1).reshape([-1, 4])
        queue.append(current_frame_pc)
        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))


        fn1_pc = np.fromfile(fn1_bin, dtype=np.float32, count=-1).reshape([-1, 4])
        fn2_pc = np.fromfile(fn2_bin, dtype=np.float32, count=-1).reshape([-1, 4])
        
        start_time = time.process_time()
        
        kd_tree = KDTree(fn2_pc[:, :3])
        dist, indexs = kd_tree.query(fn1_pc[:, :3], k=k)

        most_frequent_indexs = indexs.reshape(-1)
        
        aligned_fn2_pc = fn2_pc[most_frequent_indexs, :]
        
        pred_mid_pc = (fn1_pc + aligned_fn2_pc) / 2
        
        queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
        
        all_points_xyzi = np.concatenate(queue, axis=0)
        
        if point_count != -1:
            all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)

        all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(mid_frame_number)))
        
        o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(mid_frame_number)))

        running_time = time.process_time() - start_time
        np.save(save_dir / "{}_log.npy".format(mid_frame_number), {"id": i + 1, "point_count": point_count, "first_frame": str(fn1_bin), "mid_frame": str(save_dir / "{}.bin".format(mid_frame_number)), "last_frame": str(fn2_bin), "time": time.time(), "running_time": running_time})

def fun_iterative_closest_point(exp_path, packet_loss_rate, pro_name, args):
    from o3d_icp import GlobalRegistration
    
    point_count = args["point_count"]
    sector_size = args["sector_size"]
    icp_voxel_size = args["voxel_size"]
    
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
    my_util.create_dir(save_dir)
    
    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    file_names = sorted(none_dir.glob("*.bin"))
    file_count = len(file_names)
    assert file_count > 0, "bin files not exists: {}".format(none_dir)
    
    first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
    save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        if i + 2 >= file_count:
            continue
        start_time = time.process_time()

        queue = []
        fn1_bin = file_name
        fn2_bin = file_names[i + 2]

        mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)
        
        current_frame_pc =  np.fromfile(file_names[i+1], dtype=np.float32, count=-1).reshape([-1, 4])
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

        source_point_list = source_point_list.transform(get_half(result_icp.transformation))

        pred_mid_pc = np.concatenate((np.asarray(source_point_list.points), np.asarray(source_point_list.normals)[:, 0].reshape((-1, 1))), axis=1)
        
        queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
        
        all_points_xyzi = np.concatenate(queue, axis=0)
        
        if point_count != -1:
            all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
        
        all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(mid_frame_number)))
        
        o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(mid_frame_number)))

        running_time = time.process_time() - start_time
        np.save(save_dir / "{}_log.npy".format(mid_frame_number), {"id": i + 1, "first_frame": str(fn1_bin), "mid_frame": str(save_dir / "{}.bin".format(mid_frame_number)), "last_frame": str(fn2_bin), "voxel_size": icp_voxel_size, "time": time.time(), "running_time": running_time})
   
def fun_scene_flow(exp_path, packet_loss_rate, pro_name, args):
    import torch
    from PointINet.models.models import SceneFlow
    point_count = args["point_count"]
    sector_size = args["sector_size"]
    step_t = args["step_t"]
    sampling_type = args["sampling_type"]
    remove_outlier = args["remove_outlier"]
    
    net = SceneFlow()
    net.load_state_dict(torch.load(args["model_path"]))
    net.flow.load_state_dict(torch.load(args["flow_model_path"]))
    net.eval()
    net.cuda()
    
    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
    my_util.create_dir(save_dir)
    
    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    file_names = sorted(none_dir.glob("*.bin"))
    file_count = len(file_names)
    assert file_count > 0, "bin files not exists: {}".format(none_dir)
    
    first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
    save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        if i + 2 >= file_count:
            continue
        start_time = time.process_time()
        
        queue = []
        fn1_bin = file_name
        fn2_bin = file_names[i + 2]

        mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)
        
        current_frame_pc =  np.fromfile(file_names[i+1], dtype=np.float32, count=-1).reshape([-1, 4])
        queue.append(current_frame_pc)
        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))

        with torch.no_grad():
            pc1, color1 = o3d_sampling.get_lidar_by_bin_multi_type(o3d_sampling.bin2xyzi(fn1_bin), 16384, sampling_type, remove_outlier=remove_outlier)
            pc2, color2 = o3d_sampling.get_lidar_by_bin_multi_type(o3d_sampling.bin2xyzi(fn2_bin), 16384, sampling_type, remove_outlier=remove_outlier)
            start_time = time.process_time()

            t = torch.tensor([step_t])
            t = t.cuda().float()

            pred_mid_pc, points_i = net(pc1, pc2, color1, color2, t)

            pred_mid_pc = pred_mid_pc.squeeze(0).permute(1, 0).cpu().numpy()

            pred_mid_pc = np.concatenate((pred_mid_pc, points_i), axis=1)
            
            queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
            
            all_points_xyzi = np.concatenate(queue, axis=0)
                
            if point_count != -1:
                all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)

            all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(mid_frame_number)))
            
            o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(mid_frame_number)))
            
            running_time = time.process_time() - start_time
            np.save(save_dir / "{}_log.npy".format(mid_frame_number), {"id": i + 1, "point_count": point_count, "step": step_t, "first_frame": str(fn1_bin), "mid_frame": str(save_dir / "{}.bin".format(mid_frame_number)), "last_frame": str(fn2_bin), "time": time.time(), "running_time": running_time})


def fun_bidirectional_scene_flow(exp_path, packet_loss_rate, pro_name, args):
    import torch
    from PointINet.models.models import PointINet
    
    point_count = args["point_count"]
    sector_size = args["sector_size"]
    step_t = args["step_t"]
    sampling_type = args["sampling_type"]
    remove_outlier = args["remove_outlier"]
    
    net = PointINet()
    net.load_state_dict(torch.load(args["model_path"]))
    net.flow.load_state_dict(torch.load(args["flow_model_path"]))
    net.eval()
    net.cuda()

    save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(pro_name, packet_loss_rate) 
    my_util.create_dir(save_dir)
    
    none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)

    file_names = sorted(none_dir.glob("*.bin"))
    file_count = len(file_names)
    assert file_count > 0, "bin files not exists: {}".format(none_dir)
    
    first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
    save_first_frame(first_file_name, point_count, save_dir, sector_size)
    
    for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
        if i + 2 >= file_count:
            continue
        
        start_time = time.process_time()
        queue = []
        fn1_bin = file_name
        fn2_bin = file_names[i + 2]

        mid_frame_number = "{0:06}".format(int(fn1_bin.stem) + 1)
        current_frame_pc =  np.fromfile(file_names[i+1], dtype=np.float32, count=-1).reshape([-1, 4])
        queue.append(current_frame_pc)
        sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
        
        with torch.no_grad():
            
            pc1, color1 = o3d_sampling.get_lidar_by_bin_multi_type(o3d_sampling.bin2xyzi(fn1_bin), 16384, sampling_type, remove_outlier=remove_outlier)
            pc2, color2 = o3d_sampling.get_lidar_by_bin_multi_type(o3d_sampling.bin2xyzi(fn2_bin), 16384, sampling_type, remove_outlier=remove_outlier)

            t = torch.tensor([step_t])
            t = t.cuda().float()

            pred_mid_pc = net(pc1, pc2, color1, color2, t)

            pred_mid_pc = pred_mid_pc.squeeze(0).permute(1, 0).cpu().numpy()
            
            queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
            
            all_points_xyzi = np.concatenate(queue, axis=0)
            
            if point_count != -1:
                all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)

            all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(mid_frame_number)))
            
            o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(mid_frame_number)))

            running_time = time.process_time() - start_time
            np.save(save_dir / "{}_log.npy".format(mid_frame_number), {"id": i + 1, "point_count": point_count, "step": step_t, "first_frame": str(fn1_bin), "mid_frame": str(save_dir / "{}.bin".format(mid_frame_number)), "last_frame": str(fn2_bin), "time": time.time(), "running_time": running_time})

def main(args):
    m_config = my_util.get_config(args.c)
    execution_name = "temporal_interpolation"
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
                    globals()["fun_{}".format(k)](exp_path, packet_loss_rate, k, v)

    print("MAIN THREAD STOP")

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print("Total Running Time: {}".format(time.time() - start_time))