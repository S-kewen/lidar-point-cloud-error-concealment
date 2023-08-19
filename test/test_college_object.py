import time
import threading
import copy
import numpy as np
from pathlib import Path
import math
from matplotlib import cm
import random

from o3d_util import O3dUtil

import shutil
import yaml
import json




base_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/20230208_20_1_50_0_200/object/training/label_2"
easy_count_list, mod_count_list, hard_count_list, unknown_count_list = [], [], [], []
for i, file_name in enumerate(sorted((Path() / base_path).glob("*.txt"))):
    easy, mod, hard, unknown = 0, 0, 0, 0
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(" ")
            if items[0] == "Car":
                if items[2] == "0":
                    easy+=1
                elif items[2] == "1":
                    mod+=1
                elif items[2] == "2":
                    hard+=1
                elif items[2] == "3":
                    unknown+=1
        easy_count_list.append(easy)
        mod_count_list.append(mod)
        hard_count_list.append(hard)
        unknown_count_list.append(unknown)
        
print(np.asarray(easy_count_list).sum())
print(np.asarray(mod_count_list).sum())
print(np.asarray(hard_count_list).sum())
print(np.asarray(unknown_count_list).sum())
# print(easy_count_list, mod_count_list, hard_count_list, unknown_count_list)
# print(np.asarray(car_count_list).sum(), np.asarray(car_count_list).sum()/len(car_count_list))