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
from exp_spatial_interpolation import add_undefined_points, get_yps_by_sector_list, get_xyzs_by_ypds
from exp_temporal_interpolation import save_first_frame

def fun_threshold_based_lidar_error_concealment(exp_path, packet_loss_rate, pro_name, args):
    from sklearn.neighbors import KDTree
    import torch
    from PointINet.models.models import PointINet
    
    threadhold_tp_list = args["threadhold_tp_list"]
    threadhold_tn_list = args["threadhold_tn_list"]
    point_count = args["point_count"]
    sector_size = args["sector_size"]
    recursion = args["recursion"]
    
    # 2DNN config
    lidar_range = args["lidar_range"]
    lidar_coordinate = args["lidar_coordinate"]
    azimuth_size = args["azimuth_size"]
    channel_size = args["channel_size"]
    v_fov = args["v_fov"]
    v_fov_start = args["v_fov_start"]
    v_fov_end = args["v_fov_end"]
    angular_resolution = args["angular_resolution"]
    
    # PointINet config
    step_t = args["step_t"]
    sampling_type = args["sampling_type"]
    remove_outlier = args["remove_outlier"]
    
    net = PointINet()
    net.load_state_dict(torch.load(args["model_path"]))
    net.flow.load_state_dict(torch.load(args["flow_model_path"]))
    net.eval()
    net.cuda()
    
    for threadhold_tp in threadhold_tp_list:
        for threadhold_tn in threadhold_tn_list:
            save_dir = Path() / exp_path / "cache_frame" / "receiver_{}_{}_{}_{}".format(pro_name, packet_loss_rate, threadhold_tp, threadhold_tn)
            my_util.create_dir(save_dir)

            none_dir = Path() / exp_path / "cache_frame" / "receiver_none_{}".format(packet_loss_rate)
            file_names = sorted(none_dir.glob("*.bin"))
            file_count = len(file_names)
            assert file_count > 0, "bin files not exists: {}".format(none_dir)
            
            first_file_name = Path() / exp_path / "velodyne_compression" / "{}.bin".format(file_names[0].stem)
            save_first_frame(first_file_name, point_count, save_dir, sector_size)

            for i, file_name in tqdm(enumerate(file_names), desc="{} {}".format(pro_name, packet_loss_rate), total=len(file_names)):
                if i == 0 or i + 1 >= file_count:
                    continue
                
                start_time = time.process_time()
                queue = []
                points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
                queue.append(points_xyzi)
                
                sector_list_p = np.load(file_name.parent / "{}_log.npy".format(file_names[i-1].stem), allow_pickle=True).item()["sector_list"]
                sector_list = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()["sector_list"]
                sector_list_n = np.load(file_name.parent / "{}_log.npy".format(file_names[i+1].stem), allow_pickle=True).item()["sector_list"]
                
                all_sector_list = range(sector_size)
                drop_sector_list = list(set(all_sector_list).difference(set(sector_list)))
                
                
                sector_list_rate_p = 1 - (len(sector_list_p) / sector_size)
                sector_lost_rate = 1 - (len(sector_list) / sector_size)
                sector_lost_rate_n = 1 - (len(sector_list_n) / sector_size)
                
                concealment_types = -1
                if sector_list_rate_p >= threadhold_tp:
                    concealment_types = 1
                    #print("spatial_interpolation")
                    added_points_ypdac = add_undefined_points(points_xyzi, v_fov, v_fov_start, v_fov_end, channel_size, azimuth_size, angular_resolution, lidar_coordinate)
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
                        
                else:
                    if sector_lost_rate_n < threadhold_tn:
                        concealment_types = 3
                        # print("temporal_interpolation")
                        fn1_bin = file_names[i - 1]
                        fn2_bin = file_names[i + 1]
                        with torch.no_grad():
                            pc1, color1 = o3d_sampling.get_lidar_by_bin_multi_type(o3d_sampling.bin2xyzi(fn1_bin), 16384, sampling_type, remove_outlier=remove_outlier)
                            pc2, color2 = o3d_sampling.get_lidar_by_bin_multi_type(o3d_sampling.bin2xyzi(fn2_bin), 16384, sampling_type, remove_outlier=remove_outlier)
                            t = torch.tensor([step_t]).cuda().float()
                            
                            pred_mid_pc = net(pc1, pc2, color1, color2, t)
                            pred_mid_pc = pred_mid_pc.squeeze(0).permute(1, 0).cpu().numpy()
                            
                            queue.append(my_util.split_sector_xyzi_by_sector_list(pred_mid_pc, sector_size, drop_sector_list))
                    else:
                        concealment_types = 2
                        # print("temporal_extrapolation")
                        npy_file_name = file_name.parent.parent.parent / "location" / "{}.npy".format(file_name.stem)
                        assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)
                        npy_location = np.load(npy_file_name, allow_pickle=True).item()
                        
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
                        
                        previous_points_xyzi = my_util.split_sector_xyzi_by_sector_list(previous_points_xyzi, sector_size, drop_sector_list)
                        queue.append(my_util.split_sector_xyzi_by_sector_list(o3d_util.translate_by_matrix(o3d_util.rotate_by_matrix(previous_points_xyzi, o3d_util.get_rotation_matrix_from_angles(new_offset_xyzrs[3], new_offset_xyzrs[4], new_offset_xyzrs[5])), new_offset_xyzrs[0:3]), sector_size, drop_sector_list))
            
                all_points_xyzi = np.concatenate(queue, axis=0)
                
                if point_count != -1:
                    all_points_xyzi = o3d_sampling.adaptive_sampling_by_xyzi(all_points_xyzi, point_count)
                
                all_points_xyzi.astype(np.float32).tofile(str(save_dir / "{}.bin".format(file_name.stem)))
                
                o3d_util.save_ply_by_xyzi(all_points_xyzi, str(save_dir / "{}.ply".format(file_name.stem)))

                running_time = time.process_time() - start_time
                np.save(save_dir / "{}_log.npy".format(file_name.stem), {"id": i + 1, "file_name": str(file_name), "concealment_types": concealment_types, "sector_size": sector_size, "sector_list": sector_list + drop_sector_list, "concealed_sector_list": drop_sector_list, "time": time.time(), "running_time": running_time})

def main(args):
    m_config = my_util.get_config(args.c)
    execution_name = "threshold_based_lidar_error_concealment"
    exp_path = m_config["exp"]["path"]
    packet_loss_rates = m_config["exp"]["packet_loss_rates"]

    if m_config["exp"][execution_name]["enable"]:
        for packet_loss_rate in packet_loss_rates:
            globals()["fun_{}".format(execution_name)](exp_path, packet_loss_rate, execution_name, copy.deepcopy(m_config["exp"][execution_name]))

    print("MAIN THREAD STOP")
    
def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--c', type=str, default="config.yaml")
    # parser.add_argument('--tp', type=int, default="config.yaml")
    # parser.add_argument('--tn', type=int, default="config.yaml")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print("Total Running Time: {}".format(time.time() - start_time))
