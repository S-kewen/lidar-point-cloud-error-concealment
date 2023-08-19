import torch
import torch.nn as nn
import numpy as np

from models.models import PointINet

# import mayavi.mlab as mlab

import argparse
from tqdm import tqdm
import os

import trimesh
from plyfile import PlyData
import pandas as pd
import open3d as o3d

# envpath = '/home/skewen/anaconda3/envs/pointinet/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


def ply2bin(input_path, output_path):
    data = PlyData.read(input_path).elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    # initialize array to store data
    data_np = np.zeros(data_pd.shape, dtype=np.float)
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)


def bin2ply(input_path, output_path):
    xyzr = np.fromfile(input_path, dtype=np.float32).reshape(-1, 4)
    xyz0 = xyzr[:, :3]
    src_keypts = o3d.geometry.PointCloud()
    src_keypts.points = o3d.utility.Vector3dVector(xyz0)
    print(src_keypts)
    o3d.io.write_point_cloud(output_path, src_keypts)


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--npoints', type=int, default=32768)
    parser.add_argument('--pretrain_model', type=str,
                        default='./pretrain_model/interp_kitti.pth')
    parser.add_argument('--pretrain_flow_model', type=str,
                        default='./pretrain_model/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--is_save', type=int, default=1)
    parser.add_argument('--visualize', type=int, default=1)

    return parser.parse_args()


def get_lidar(fn, npoints):
    points = np.fromfile(fn, dtype=np.float32, count=-1).reshape([-1, 4])
    raw_num = points.shape[0]
    if raw_num >= npoints:
        sample_idx = np.random.choice(raw_num, npoints, replace=False)
    else:
        sample_idx = np.concatenate((np.arange(raw_num), np.random.choice(
            raw_num, npoints - raw_num, replace=True)), axis=-1)

    pc = points[sample_idx, :]
    pc = torch.from_numpy(pc).t()
    color = np.zeros([npoints, 3]).astype('float32')
    color = torch.from_numpy(color).t()

    pc = pc.unsqueeze(0).cuda()
    color = color.unsqueeze(0).cuda()

    return pc, color


def demo(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fn1 = './data/demo_data/original/1.bin'  # frist frame
    fn2 = './data/demo_data/original/2.bin'  # last frame

    net = PointINet()
    net.load_state_dict(torch.load(args.pretrain_model))  # loading model
    net.flow.load_state_dict(torch.load(
        args.pretrain_flow_model))  # loading model
    net.eval()
    net.cuda()

    interp_scale = 1  # interp how many frame
    t_array = np.arange(1.0/(interp_scale+1), 1.0, 1.0 /
                        (interp_scale+1), dtype=np.float32)

    with torch.no_grad():
        pc1, color1 = get_lidar(fn1, args.npoints)
        pc2, color2 = get_lidar(fn2, args.npoints)

        for i in range(interp_scale):
            t = t_array[i]
            t = torch.tensor([t])
            t = t.cuda().float()

            pred_mid_pc = net(pc1, pc2, color1, color2, t)

            ini_pc = pc1.squeeze(0).permute(1, 0).cpu().numpy()
            end_pc = pc2.squeeze(0).permute(1, 0).cpu().numpy()

            pred_mid_pc = pred_mid_pc.squeeze(0).permute(1, 0).cpu().numpy()

            if args.visualize == 1:
                point_list_init = o3d.geometry.PointCloud()

                point_list_init.points = o3d.utility.Vector3dVector(ini_pc[:, :3])
                point_list_init.paint_uniform_color([0, 0, 1])

                point_list_end = o3d.geometry.PointCloud()
                point_list_end.points = o3d.utility.Vector3dVector(end_pc[:, :3])
                point_list_end.paint_uniform_color([0, 1, 0])

                point_list_mid = o3d.geometry.PointCloud()
                point_list_mid.points = o3d.utility.Vector3dVector(pred_mid_pc[:, :3])
                point_list_mid.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([point_list_init, point_list_end, point_list_mid])

            if args.is_save == 1:
                save_dir = './data/demo_data/interpolated'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = os.path.join(save_dir, str(
                    t.squeeze().cpu().numpy())+'.bin')
                pred_mid_pc.tofile(save_name)
                print("save interpolated point clouds to:", save_name)


if __name__ == '__main__':
    args = parse_args()
    demo(args)
    # bin2ply('./data/demo_data/original/000000.bin','./data/demo_data/original/000000.ply')
    # ply2bin('/home/skewen/UnrealEngine_4.26/Engine/Binaries/Linux/carla/PythonAPI/examples/output/0/10252.ply','./data/demo_data/original/3.bin')
