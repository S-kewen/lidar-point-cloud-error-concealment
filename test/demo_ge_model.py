
import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
from socket import *
import json
import time
from multiprocessing import Process
import threading
from multiprocessing import Pool
import yaml
import math
from pathlib import Path
from o3d_icp import GlobalRegistration
from o3d_util import O3dUtil
from tqdm import tqdm

from sim2net.packet_loss.gilbert_elliott import GilbertElliott


def main():
    drop = 0
    loss_rate = 0.0
    recv_rate = 1.0 - loss_rate
    GE_parameters = (loss_rate, recv_rate, 0.0, 1.0)
    GE = GilbertElliott(GE_parameters)
    for i in range(100):
        flag = GE.packet_loss()
        if flag:
            drop+=1
    print("drop_flag: {}".format(drop))
    
    for i 
    
    
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time()-start_time))

