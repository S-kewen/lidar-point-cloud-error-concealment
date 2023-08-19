import numpy as np
import time

import torch
from torch import einsum
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join("/home/skewen/lidar-base-point-cloud-error-concealment/Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops"))
from pointnet2_utils import furthest_point_sample

def main():
    fn1_bin = "/mnt/data/skewen/kittiGenerator/output/test_1_50_0_10/object/training/velodyne/000000.bin"
    points_xyzs = np.fromfile(fn1_bin, dtype=np.float32, count=-1).reshape([-1, 4])
    torch_point_xyzs = torch.from_numpy(points_xyzs).unsqueeze(0).transpose(2,1).contiguous()
    torch_point_xyzs = torch_point_xyzs.float().cuda()
    result = furthest_point_sample(torch_point_xyzs, 16384)
    np_result = result.squeeze(0).cpu().numpy()
    points_xyzs = points_xyzs[np_result]
    print(points_xyzs.shape)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time() - start_time))