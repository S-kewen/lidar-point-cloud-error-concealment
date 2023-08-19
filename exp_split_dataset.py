import time
import threading
import open3d as o3d
import copy
import numpy as np
from pathlib import Path
from o3d_icp import GlobalRegistration
import math
from matplotlib import cm
import random

from o3d_util import O3dUtil

import shutil


def main():
    rename = True
    
    frame_ranges = [[150, 749], [250, 849], [350, 949]]
    base_paths = [Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00", Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00", Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00"]
    to_paths = [Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry_600/object/CAR2", Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry_600/object/training", Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry_600/object/CAR3"]
    
    # frame_range = [150, 749]
    # base_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00"
    # to_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry_600/object/CAR2"
    
    # frame_range = [250, 849]
    # base_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00"
    # to_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry_600/object/training"
    
    # frame_range = [350, 949]
    # base_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry/dataset/sequences/00"
    # to_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/kitti_odometry_600/object/CAR3"
    
    for i, frame_range in enumerate(frame_ranges):
        base_path = base_paths[i]
        to_path = to_paths[i]
        
        if not to_path.exists():
            to_path.mkdir(parents=True)

        for path_name in sorted((base_path).glob("**/")):
            if path_name != base_path:
                final_path = to_path / path_name.name
                if not final_path.exists():
                    final_path.mkdir(parents=True)
                for file_name in sorted((path_name).glob("*.*")):
                    frame_number = int(file_name.stem.replace("_log", ""))
                    if frame_number >= frame_range[0] and frame_number <= frame_range[1]:
                        
                        if rename:
                            final_name = "{0:06}{1}".format(frame_number - frame_range[0], file_name.suffix)
                        else: 
                            final_name = file_name.name
                        shutil.copy(str(file_name), str(final_path / final_name))
                        print("{} TO {}".format(file_name, str(final_path / final_name)))
        
        
        with open(to_path / "list.txt", 'w') as f:
            for file_name in sorted((to_path / "velodyne").glob("*.bin")):
                f.write("{}\n".format(file_name.stem))

if __name__ == "__main__":
    main()
