import time
import threading
import open3d as o3d
import copy
import numpy as np
from pathlib import Path

'''
Using to find the optimus a for closest spatial algo
'''


def main():
    base_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230306_False_1_1_0_0_200/object/training/cache_frame"
    dirs = ["receiver_closest_spatial_0.2_13"]
    a = 6
    sector_size = 180
    for x in dirs:
        success, fail = 0, 0
        for i, file_name in enumerate(sorted((base_dir / x).glob("*_log.npy"))):
            if i == 0 :
                continue
            reference_id_list = np.load(file_name, allow_pickle=True).item()["reference_id_list"]
            for reference in reference_id_list:
                if reference["default"] == False:
                    sub = min(abs(reference["drop_sector"] - reference["reference_sector"]), abs(reference["drop_sector"] - reference["reference_sector"] - sector_size))
                    if sub <= a:
                        success+=1
                    else:
                        fail+=1
                        #print(reference)
        print("a = {}: {} success: {}, fail: {}, rate: {}".format(a, x, success, fail, success/(success+fail)))

def main_running_time():
    base_dir = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230216_False_1_1_100_0_200/object/training/cache_frame"
    for dir in sorted(base_dir.iterdir()):
        file_names = sorted((dir).glob("*_log.npy"))
        running_time_list = []
        for i, file_name in enumerate(file_names):
            running_time_list.append(np.load(file_name, allow_pickle=True).item()["running_time"])
            # print(np.load(file_name, allow_pickle=True).item()["running_time"])
        
        print("{}: {}".format(dir.name, np.mean(running_time_list)))

if __name__ == "__main__":
    main()