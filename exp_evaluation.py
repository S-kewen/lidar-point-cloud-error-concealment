import numpy as np
from tqdm import tqdm
import open3d as o3d
from multiprocessing import Pool
from multiprocessing import Process
import threading
from pathlib import Path
from my_util import MyUtil as my_util
from eval_util import EvalUtil as eval_util
import time
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--c', type=str, default="config.yaml")
    return parser.parse_args()

def fun_calculate_metrics(exp_path, target_dir, dir, args):
    is_running_time = args["running_time"]
    is_cd = args["cd"]
    is_hd = args["hd"]
    
    file_names = sorted(dir.glob("*.bin"))

    save_dir = dir / "evaluation"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    running_time_list, cd_list, hd_list  = [], [], []


    for i, file_name in tqdm(enumerate(file_names), desc="calculate {}".format(dir), total=len(file_names)):
        cd, hd = None, None

        ground_truth_frame = target_dir / "{}.bin".format(file_name.stem)
        
        source_points_xyzi = np.fromfile(str(file_name), dtype=np.float32, count=-1).reshape([-1, 4])
        target_points_xyzi = np.fromfile(str(ground_truth_frame), dtype=np.float32, count=-1).reshape([-1, 4])
        
        if is_cd or is_hd:
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(source_points_xyzi[:, :3])
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(target_points_xyzi[:, :3])
            
            source_distances = np.asarray(source.compute_point_cloud_distance(target))
            target_distances = np.asarray(target.compute_point_cloud_distance(source))
            
            

        if is_running_time:
            npy_log_stitching = file_name.parent.parent / "{}_log.npy".format(file_name.stem)
            npy_log_interpolation = file_name.parent / "{}_log.npy".format(file_name.stem)

            if npy_log_stitching.exists():
                stitching_running_time = np.load(npy_log_stitching, allow_pickle=True).item()["running_time"]
            else:
                stitching_running_time = 0.0
            
            if npy_log_interpolation.exists():
                interpolation_running_time = np.load(npy_log_interpolation, allow_pickle=True).item()["running_time"]
            else:
                interpolation_running_time = 0.0
                
            running_time_list.append([file_name.stem, stitching_running_time + interpolation_running_time, stitching_running_time, interpolation_running_time])
            
        if is_cd:
            cd = np.sum(np.square(source_distances))/len(source.points) + np.sum(np.square(target_distances))/len(target.points)
            cd_list.append([file_name.stem, cd])

        if is_hd:
            hd = np.max([np.max(source_distances), np.max(target_distances)])
            hd_list.append([file_name.stem, hd])

    if is_running_time:
        np.save(save_dir / 'running_time.npy', running_time_list)
        
    if is_cd:
        np.save(save_dir / 'cd.npy', cd_list)

    if is_hd:
        np.save(save_dir / 'hd.npy', hd_list)

def main(args):
    m_config = my_util.get_config(args.c)
    exp_path = m_config["exp"]["path"]
    gt_dir = m_config["exp"]["evaluation"]["gt_dir"]
    target_dir = Path() / exp_path / gt_dir

    skip_list = [Path() / exp_path / "cache_frame" / "receiver_none_ns3_c_v2x", Path() / exp_path / "cache_frame" / "receiver_none_ns3_dsrc_ap"]
    
    dirs = []
    base_path = Path() / exp_path / "cache_frame"
    for i in base_path.iterdir():
        dirs.append(i)
    
    with Pool(processes=None) as pool:
        for dir in dirs:
            if dir in skip_list:
                continue
            pool.apply_async(fun_calculate_metrics, (exp_path, target_dir, dir, copy.deepcopy(m_config["exp"]["evaluation"])))
        pool.close()
        pool.join()

    print("MAIN THREAD STOP")

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print("Total Running Time: {}".format(time.time() - start_time))
