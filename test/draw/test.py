import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import yaml
import json

def draw_line_diagram_a_accuracy():
    x = [1, 2, 3, 4, 5, 6]#点的横坐标
    # pointrcnn_easy = [95.6623,95.82780967,95.17123502,95.49738939,95.60661506,95.75720154]#线1的纵坐标
    # pointrcnn_mod = [87.8699,87.67662336,87.4365526527455,87.7747444781028,87.5464249908995,87.5479015098596]#线2的纵坐标
    # pointrcnn_hard = [87.5614,87.1461817918795,87.0210955590806,87.2706093585784,87.3760055192133,87.3012558994349]#线3的纵坐标

    # pointrcnn_easy = [464, 462.41, 378.62, 335.60, 168.71]#线1的纵坐标
    # pointrcnn_mod = [70, 64.33, 57.44, 44.82, 25.63]#线2的纵坐标
    # pointrcnn_hard = [15, 14.49, 12.94, 10.10, 5.76]#线3的纵坐标
    
    result_20 = [99.9128, 99.9128, 99.9128, 99.9128, 99.9128, 99.9128]#线1的纵坐标
    result_40 = [99.8716, 99.8716, 99.8716, 99.8716, 99.8716, 99.8716]#线1的纵坐标
    result_60 = [97.7867, 99.8067, 99.8067, 99.8067, 99.8067, 99.8067]#线1的纵坐标
    result_80 = [92.2874, 97.9823, 99.6564, 99.6564, 99.6564, 99.6564]#线1的纵坐标
    plt.rcParams["font.family"] = 'Times New Roman'
    plt.plot(x,result_20,'s-',color = 'r',label="20%")#s-:方形
    plt.plot(x,result_40,'o-',color = 'g',label="40%")#s-:方形
    plt.plot(x,result_60,'p-',color = 'b',label="60%")#s-:方形
    plt.plot(x,result_80,'x-',color = 'c',label="80%")#s-:方形
    plt.xlabel("Calculate Range")#横坐标名字
    plt.ylabel("Accuracy (%)")#纵坐标名字
    plt.legend(loc = "best")#图例
    
    plt.savefig("result_a_accuracy.svg")
    plt.savefig("result_a_accuracy.eps")
    plt.savefig("result_a_accuracy.png")
    # plt.show()
    plt.close()
    
def draw_line_diagram_a_running_time():
    x = [1, 2, 3, 4, 5, 6]#点的横坐标
    # pointrcnn_easy = [95.6623,95.82780967,95.17123502,95.49738939,95.60661506,95.75720154]#线1的纵坐标
    # pointrcnn_mod = [87.8699,87.67662336,87.4365526527455,87.7747444781028,87.5464249908995,87.5479015098596]#线2的纵坐标
    # pointrcnn_hard = [87.5614,87.1461817918795,87.0210955590806,87.2706093585784,87.3760055192133,87.3012558994349]#线3的纵坐标

    # pointrcnn_easy = [464, 462.41, 378.62, 335.60, 168.71]#线1的纵坐标
    # pointrcnn_mod = [70, 64.33, 57.44, 44.82, 25.63]#线2的纵坐标
    # pointrcnn_hard = [15, 14.49, 12.94, 10.10, 5.76]#线3的纵坐标
    
    result_20 = [36.98266392096499, 44.305651593929994, 52.37132557782501, 59.979792129925, 67.70407588978003, 74.55997643243997]#线1的纵坐标
    result_40 = [78.4096619245699, 92.91578790071506, 108.01951922302, 122.58086018722501, 137.52829497654005, 150.521023758685]#线1的纵坐标
    result_60 = [132.12039793219984, 163.04294561735017, 187.02707647101997, 213.87488000429983, 236.83225042385973, 260.50085425827496]#线1的纵坐标
    result_80 = [207.2977142826299, 279.6129578485649, 340.68995336819984, 395.9314138131002, 463.07700308492997, 521.539302426395]#线1的纵坐标
    plt.rcParams["font.family"] = 'Times New Roman'
    plt.plot(x,result_20,'s-',color = 'r',label="20%")#s-:方形
    plt.plot(x,result_40,'o-',color = 'g',label="40%")#s-:方形
    plt.plot(x,result_60,'p-',color = 'b',label="60%")#s-:方形
    plt.plot(x,result_80,'x-',color = 'c',label="80%")#s-:方形
    plt.xlabel("Calculate Range")#横坐标名字
    plt.ylabel("Running Time (sec)")#纵坐标名字
    plt.legend(loc = "best")#图例
    
    plt.savefig("result_a_running_time.svg")
    plt.savefig("result_a_running_time.eps")
    plt.savefig("result_a_running_time.png")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    start_time = time.time()
    draw_line_diagram_a_accuracy()
    draw_line_diagram_a_running_time()
    print("Total time: {}".format(time.time() - start_time))