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

def fun_calculate_metrics(exp_path, target_dir, file_name, save_dir, args):
    sector_size = args["sector_size"]
    is_snn_rmse = args["snn_rmse"]
    is_running_time = args["running_time"]
    is_acd = args["acd"]
    is_cd = args["cd"]
    is_cd_psnr = args["cd_psnr"]
    is_hd = args["hd"]
    is_emd = args["emd"]

    pre_cal = is_snn_rmse or is_acd or is_cd or is_cd_psnr

    source_distances, target_distances, source_avg_distance, target_avg_distance = None, None, None, None
    snn_rmse, acd, cd, cd_psnr, hd = None, None, None, None, None

    ground_truth_frame = target_dir / "{}.bin".format(file_name.stem)
    log_file_name = file_name.parent / "{}_log.npy".format(file_name.stem) #.parent
    
    source_points_xyzi = np.fromfile(str(file_name), dtype=np.float32, count=-1).reshape([-1, 4])
    target_points_xyzi = np.fromfile(str(ground_truth_frame), dtype=np.float32, count=-1).reshape([-1, 4])
    
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points_xyzi[:, :3])
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points_xyzi[:, :3])
    
    if pre_cal:
        source_distances = np.asarray(source.compute_point_cloud_distance(target))
        target_distances = np.asarray(target.compute_point_cloud_distance(source))
        source_avg_distance = np.sum(np.square(source_distances))/len(source.points)
        target_avg_distance = np.sum(np.square(target_distances))/len(target.points)
    
    if is_snn_rmse:
        snn_rmse = eval_util.cal_snn_rmse_by_xyzis(source_points_xyzi, target_points_xyzi, source_avg_distance, target_avg_distance)
        np.save(save_dir / 'snn_rmse_{}.npy'.format(file_name.stem), [snn_rmse_list])

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

        np.save(save_dir / 'running_time_{}.npy'.format(file_name.stem), [file_name.stem, stitching_running_time + interpolation_running_time, stitching_running_time, interpolation_running_time])

    if is_acd:
        acd = eval_util.cal_acd_by_xyzis(target_points_xyzi, source_points_xyzi, target_distances)
        np.save(save_dir / 'acd_{}.npy'.format(file_name.stem), [file_name.stem, acd])

    if is_cd:
        cd = eval_util.cal_cd_by_xyzis(source_points_xyzi, target_points_xyzi, acd, target_distances)
        np.save(save_dir / 'cd_{}.npy'.format(file_name.stem), [file_name.stem, cd])

    if is_cd_psnr:
        cd_psnr = eval_util.cal_cd_psnr_by_xyzis(source_points_xyzi, target_points_xyzi, cd)
        np.save(save_dir / 'cd_psnr_{}.npy'.format(file_name.stem), [file_name.stem, cd_psnr])

    if is_hd:
        hd = eval_util.cal_hd_by_xyzis(source_points_xyzi, target_points_xyzi)
        np.save(save_dir / 'hd_{}.npy'.format(file_name.stem), [file_name.stem, hd])
        
    if is_emd:
        emd = eval_util.cal_emd_by_xyzis(source_points_xyzi, target_points_xyzi)
        np.save(save_dir / 'emd_{}.npy'.format(file_name.stem), [file_name.stem, emd])

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
    
    for dir in dirs:
        if dir in skip_list:
            continue
        
        save_dir = dir / "evaluation_pre_frame"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            
        file_names = sorted(dir.glob("*.bin"))
        
        with Pool(processes=None) as pool:
            for i, file_name in tqdm(enumerate(file_names), desc="calculate {}".format(dir), total=len(file_names)):
                # fun_calculate_metrics(exp_path, target_dir, file_name, save_dir, copy.deepcopy(m_config["exp"]["evaluation"]))
                pool.apply_async(fun_calculate_metrics, (exp_path, target_dir, file_name, save_dir, copy.deepcopy(m_config["exp"]["evaluation"])))
            pool.close()
            pool.join()

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
