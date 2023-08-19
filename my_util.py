import time
import open3d as o3d
import copy
import numpy as np
import math
import time
import yaml
from pathlib import Path

class MyUtil(object):
    def __init__(self):
        pass
    
    def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.load(f)
        return config

    def get_config(file_name):
        with open(file_name, 'r') as yamlfile:
            return yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    def create_dir(dir):
        if not dir.exists():
            dir.mkdir(parents=True)
            
    def split_sector_xyzi_by_sector_list(points_xyzi, sector_size, sector_list):
        sector_range = 360 / sector_size
        points_a = np.arctan(points_xyzi[:, 1] / points_xyzi[:, 0]) / math.pi * 180
        points_xyzia = np.concatenate((points_xyzi, points_a.reshape(-1, 1)), axis = 1)

        points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] > 0)), 4] = 90.0
        points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] <= 0)), 4] = -90.0
        points_xyzia[np.where(points_xyzia[:, 0] < 0), 4] += 180
        points_xyzia[np.where(points_xyzia[:, 4] < 0), 4] += 360

        result_points_xyzi = np.zeros((0, 4))
        for sector in sector_list:
            result_points_xyzi = np.concatenate((result_points_xyzi, points_xyzi[np.where((points_xyzia[:, 4] >= sector * sector_range) & (points_xyzia[:, 4] < (sector + 1) * sector_range))]), axis = 0)
        return result_points_xyzi
    
    def split_sector_ypd_by_sector_list(points_ypd, sector_size, sector_list):
        sector_range = 360 / sector_size
        result_points_ypd = np.zeros((0, 3))
        # dont use for loop to concatenate, too slow
        for sector in sector_list:
            result_points_ypd = np.concatenate((result_points_ypd, points_ypd[np.where((points_ypd[:, 0] >= sector * sector_range) & (points_ypd[:, 0] < (sector + 1) * sector_range))]), axis = 0)
        return result_points_ypd
    
    
    def repair_drop_sector_by_xyzi(base_xyzi, intermediate_xyzi, sector_size, rx_sector_list):
        all_sector_list = range(sector_size)
        drop_sector_list = list(set(all_sector_list).difference(set(rx_sector_list)))
        
        rx_points_xyzi = MyUtil.split_sector_xyzi_by_sector_list(base_xyzi, sector_size, rx_sector_list)

        drop_points_xyzi = MyUtil.split_sector_xyzi_by_sector_list(intermediate_xyzi, sector_size, drop_sector_list)

        return np.concatenate((rx_points_xyzi, drop_points_xyzi), axis = 0)
    
    def get_pro_list(config, step_name):
        execution_list = ["process", "thread", "single"]
        result = []
        for e in execution_list:
            if config["exp"][step_name][e] is not None:
                for k, v in config["exp"][step_name][e].items():
                    if v["enable"]:
                        result.append(k)
        return result