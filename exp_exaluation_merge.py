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

# 20230309_False_1_1_140_0_600
# 20230309_False_1_3_140_0_600
# 20230309_False_1_5_140_0_600
# 20230309_False_1_7_140_0_600

def main():
    exp_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230309_False_1_3_140_0_600/object/training"
    
    metric_list = ["cd", "hd", "running_time"] 

    dirs = []
    base_path = Path() / exp_path / "cache_frame"
    for i in base_path.iterdir():
        dirs.append(i)
    
    for dir in dirs:
        pre_frame_dir = dir / "evaluation_pre_frame"
        if not pre_frame_dir.exists():
            continue
        
        save_dir = dir / "evaluation"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        print(dir)
        for metric in metric_list:
            result_list = []
            file_names = sorted(pre_frame_dir.glob("{}_*.npy".format(metric)))
            for file_name in file_names:
                result_list.append(np.load(file_name, allow_pickle=True).astype(float))
            np.save(save_dir / '{}.npy'.format(metric), result_list)
            print(metric, np.mean(np.load(save_dir / '{}.npy'.format(metric), allow_pickle=True)[:, 1].astype(float)))

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time() - start_time))