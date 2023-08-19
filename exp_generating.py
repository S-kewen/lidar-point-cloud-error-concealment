import time
import random
import numpy as np
import json
from multiprocessing import Pool
import math
from pathlib import Path
from tqdm import tqdm
from o3d_util import O3dUtil as o3d_util
from my_util import MyUtil as my_util
import open3d as o3d
import copy
import argparse

def format_ply_random_by_packet_loss_rate_xyzi(points_xyzi, sector_size, packet_loss_rate, ge_model_flag):
    from sim2net.packet_loss.gilbert_elliott import GilbertElliott
    sector_list= []
    if ge_model_flag:
        recv_rate = 1.0 - packet_loss_rate
        ge_model = GilbertElliott((packet_loss_rate, recv_rate, 0.0, 1.0))
        for i in range(sector_size):
            if ge_model.packet_loss() == False:
                sector_list.append(i)     
    else:
        drop_size = int(packet_loss_rate*sector_size)
        if drop_size > sector_size:
            drop_size = sector_size
        
        for i in range(sector_size):
            sector_list.append(i)

        for i in range(drop_size):
            sector_list.pop(random.randint(0, len(sector_list)-1))

    if packet_loss_rate <= 0.0:
        return points_xyzi[:, :3], points_xyzi[:, 3], sector_list
    
    result_points_xyzi = my_util.split_sector_xyzi_by_sector_list(points_xyzi, sector_size, sector_list)

    return result_points_xyzi[:, :3], result_points_xyzi[:, 3], sector_list



def fun_generation(exp_path, file_name, packet_loss_rate, pro_name, args):
    start_time = time.process_time()
    sector_size = args["sector_size"]
    source_mac = args["source_mac"]
    receive_mac = args["receive_mac"]
    ge_model_flag = args["ge_model_flag"]
    save_dir = Path() / exp_path / args["save_dir"].format(packet_loss_rate)
    my_util.create_dir(save_dir)
    
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    if str(packet_loss_rate).find("ns3") != -1:
        ns3_file_name = Path() / exp_path / str(packet_loss_rate) / "{}.txt".format(file_name.stem)
        sector_list = get_sector_list_by_txt(ns3_file_name, source_mac, receive_mac)
        points_xyz, points_i = format_bin_by_ns3_xyzi(points_xyzi, sector_size, sector_list)
    else:
        points_xyz, points_i, sector_list = format_ply_random_by_packet_loss_rate_xyzi(points_xyzi, sector_size, packet_loss_rate, ge_model_flag)

    if len(sector_list) > 0:
        points_xyzi = np.concatenate((points_xyz, np.asarray(points_i).reshape((-1, 1))), axis=1)
        points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))
        o3d_util.save_ply_by_xyzi(points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))
    else:
        print("no sector_list", file_name)
    running_time = time.process_time() - start_time
    np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "sector_size": sector_size, "sector_list": sector_list, "packet_loss_rate": packet_loss_rate, "time": time.time(), "running_time": running_time})
    
    
def fun_ground_removal(exp_path, file_name, pro_name, args):
    start_time = time.process_time()
    voxel_size = args["voxel_size"]
    ransac_n = args["ransac_n"]
    distance_threshold = args["distance_threshold"]
    num_iterations = args["num_iterations"]
    
    save_dir = Path() / exp_path / args["save_dir"]
    my_util.create_dir(save_dir)
    
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    pcd_filtered = point_list.voxel_down_sample(voxel_size=voxel_size)

    all_indexs = np.arange(len(pcd_filtered.points))

    [planes, ground_indexs] = pcd_filtered.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    non_ground_indexs = list(set(all_indexs) - set(ground_indexs))

    pcd_non_ground = pcd_filtered.select_by_index(non_ground_indexs)
    
    result = np.concatenate((np.asarray(pcd_non_ground.points), np.asarray(pcd_non_ground.normals)[:, 0].reshape(-1, 1)), axis=1)

    result.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))
    o3d_util.save_ply_by_xyzi(result, str(save_dir / "{}.ply".format(file_name.stem)))
    running_time = time.process_time() - start_time
    np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "voxel_size": voxel_size, "ransac_n": ransac_n, "distance_threshold": distance_threshold, "num_iterations": num_iterations, "time": time.time(), "running_time": running_time})
    
def fun_compression(exp_path, file_name, pro_name, args):
    import DracoPy
    start_time = time.process_time()
    save_dir = Path() / exp_path / args["save_dir"]
    
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    colors = np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1)
    colors = (colors * 255).astype(np.uint8)
    binary = DracoPy.encode(points_xyzi[:, :3], colors = colors, preserve_order = True)
    
    buffer_bin = np.frombuffer(binary, dtype=np.uint8)
    buffer_bin.tofile(save_dir / "{}.drc".format(file_name.stem))
    
    compressed_drc = DracoPy.decode(np.fromfile(save_dir / "{}.drc".format(file_name.stem), dtype=np.uint8).tobytes())
    compressed_points_xyzi = np.concatenate((compressed_drc.points, compressed_drc.colors[:, 0].reshape(-1, 1) / 255), axis=1)
    
    compressed_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))
    
    o3d_util.save_ply_by_xyzi(compressed_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))

    running_time = time.process_time() - start_time
    np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "time": time.time(), "running_time": running_time})

def format_bin_by_ns3_xyzi(points_xyzi, sector_size, sector_list):
    if len(sector_list) >= sector_size:
        return points_xyzi[:, :3], points_xyzi[:, 3]
    result_points_xyzi = my_util.split_sector_xyzi_by_sector_list(points_xyzi, sector_size, sector_list)
    return result_points_xyzi[:, :3], result_points_xyzi[:, 3]


def get_sector_list_by_txt(file_name, source_mac, receive_mac):
    sector_list = []
    if file_name.exists():
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                json_data = json.loads(line)
                if json_data["type"] == "rx" and json_data["messageType"] == "PU" and str(json_data["sourceMac"]) == str(source_mac) and str(json_data["receiveMac"]) == str(receive_mac):
                    sector_list.append(int(json_data["segment"]))
    return sector_list


def main(args):
    m_config = my_util.get_config(args.c)
    exp_path = m_config["exp"]["path"]
    packet_loss_rates = m_config["exp"]["packet_loss_rates"]
                
    if m_config["exp"]["pre_process"]["process"]["ground_removal"]["enable"]:
        print("generating ground_removal data...")
        with Pool(processes=None) as pool:
            for file_name in sorted((Path() / exp_path / "velodyne").glob("*.bin")):
                pool.apply_async(fun_ground_removal, (exp_path, file_name, "ground_removal", copy.deepcopy(m_config["exp"]["pre_process"]["process"]["ground_removal"])))
            pool.close()
            pool.join()
            
            
    if m_config["exp"]["pre_process"]["process"]["compression"]["enable"]:
        print("generating compression data...")
        with Pool(processes=None) as pool:
            for file_name in sorted((Path() / exp_path / "velodyne_ground_removal").glob("*.bin")):
                pool.apply_async(fun_compression, (exp_path, file_name, "compression", copy.deepcopy(m_config["exp"]["pre_process"]["process"]["compression"])))
            pool.close()
            pool.join()
    
    
    if m_config["exp"]["pre_process"]["process"]["generation"]["enable"]:
        print("generating none data...")
        for i, packet_loss_rate in tqdm(enumerate(packet_loss_rates), desc="generating", total=len(packet_loss_rates)):
            for file_name in sorted((Path() / exp_path / "velodyne_compression").glob("*.bin")): # here to choose the data to generate !!!!
                fun_generation(exp_path, file_name, packet_loss_rate, "generation", copy.deepcopy(m_config["exp"]["pre_process"]["process"]["generation"]))
                
    print("MAIN THREAD STOP")

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--c', type=str, default="config.yaml")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print("Total Running Time: {}".format(time.time() - start_time))