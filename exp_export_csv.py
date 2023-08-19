import numpy as np
import time
from pathlib import Path
import yaml
import json
from my_util import MyUtil as my_util
import math

def get_mean_by_npy(file_name):
    print(file_name)
    return np.mean(np.load(file_name, allow_pickle=True)[:, 1].astype(float))

def export_latency(exp_path, ns3_dir, source_mac, receive_mac, save_name):
    csv_list = []
    latency_list = []
    for i, file_name in enumerate(sorted((exp_path / ns3_dir).glob("**/*.txt"))):
        with open(file_name, "r") as f:
            lines = f.readlines()
            latency = 0.0
            count = 0
            for line in lines:
                json_data = json.loads(line)
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == str(source_mac) and json_data["receiveMac"] == str(receive_mac):
                    if json_data["type"] == "rx":
                        latency += (json_data["recvTime"] - json_data["sendTime"])*1000
                        count += 1
            result = latency/count
            latency_list.append(result)
            csv_list.append([str(frame_number), result])
    np.savetxt(save_name, csv_list, fmt='%s',delimiter=',')

def export_packet_loss_rate(exp_path, ns3_dir, source_mac, receive_mac, save_name, sector_size=180):
    packet_loss_rate_list = []
    csv_list = []
    for i, file_name in enumerate(sorted((exp_path / ns3_dir).glob("**/*.txt"))):
        with open(file_name, "r") as f:
            lines = f.readlines()
            rxCount = 0
            for line in lines:
                json_data = json.loads(line)
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == str(source_mac) and json_data["receiveMac"] == str(receive_mac):
                    if json_data["type"] == "rx":
                        rxCount += 1

            result = (1-(rxCount/sector_size))*100
            packet_loss_rate_list.append(result)
            csv_list.append([str(frame_number), result])
    np.savetxt(save_name, csv_list, fmt='%s',delimiter=',')

def export_distance(exp_path, frame_size, agent_list, edge_server_location):
    from scipy.spatial.distance import euclidean
    csv_dir = Path().cwd() / "csv_{}".format(exp_path.parent.name)
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
    
    for agent in agent_list:
        csv_list = []
        for i in range(frame_size):
            #print("[{}] {}".format(i, file_name))
            npy_file_name = exp_path / agent / "location" / "{0:06}.npy".format(i)
            if not npy_file_name.exists():
                continue
            npy_location = np.load(npy_file_name, allow_pickle=True).item()
            location = np.asarray([npy_location["x"], npy_location["y"], npy_location["z"]])
            
            csv_list.append([str(i+1), euclidean(edge_server_location, location)])
        if csv_list != []:
            np.savetxt(csv_dir / "distance_frame_{}.csv".format(agent), csv_list, fmt='%s',delimiter=',')


def export_throughput(exp_path, ns3_dir, frame_size, agent_list, save_name):
    csv_list = []
    for i in range(frame_size):
        total_size = 0.0
        g_firstReceived = 0.0
        g_lastReceived = -1.0
        for agent in agent_list:
            file_name = exp_path / agent / ns3_dir / "{0:06}.txt".format(i)
            if not file_name.exists():
                continue
            with open(file_name, "r") as f:
                lines = f.readlines()
                for line in lines:
                    json_data = json.loads(line)
                    if json_data["type"] == "rx":
                        total_size += float(json_data["size"])*8
                        if g_firstReceived == 0.0:
                            g_firstReceived = float(json_data["recvTime"])
                        if float(json_data["recvTime"]) > g_lastReceived:
                            g_lastReceived = float(json_data["recvTime"])           
        frame_number = i + 1
        if total_size > 0 and (g_lastReceived - g_firstReceived)!=0:
            result = total_size/(g_lastReceived - g_firstReceived)/1e6
        else:
            result = 0.0
        csv_list.append([str(frame_number), min(result, 100)])
        np.savetxt(save_name, csv_list, fmt='%s',delimiter=',')


def main():
    # agent_list = ["training", "CAR2", "CAR3", "CAR4", "CAR5", "CAR6", "CAR7"]
    # dataset_list = ["20230413_False_1_6_110_0_600_1", "20230413_False_1_6_110_0_600_2", "20230413_False_1_6_110_0_600_3", "20230413_False_1_6_110_0_600_4", "20230413_False_1_6_110_0_600_5", "20230413_False_1_6_110_0_600_6", "20230413_False_1_6_110_0_600_7", "20230413_False_1_6_110_0_600_8", "20230413_False_1_6_110_0_600_9", "20230413_False_1_6_110_0_600_10"]
    agent_list = ["CAR3"]
    dataset_list = ["20230413_False_1_6_110_0_600_1"]
    cache_frame_dir = "cache_frame"
    ns3_dir = "ns3_dsrc_ap_{}"
    frame_size = 600
    

    
    # generate all algo results
    for dataset in dataset_list:
        exp_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/{}/object".format(dataset)
        
        csv_dir = Path().cwd() / "csv_{}".format(dataset)
        my_util.create_dir(csv_dir)
        
        # generate packet loss rate, latency, throughput
        # for i, v in enumerate(agent_list):
            # export_packet_loss_rate(exp_path / v, ns3_dir.format(i+1), i, i+1, csv_dir / "{}_{}_{}.csv".format(v, "packet_loss_rate", ns3_dir.format(i+1)))
            # export_latency(exp_path / v, ns3_dir.format(i+1), i, i+1, csv_dir / "{}_{}_{}.csv".format(v, "latency", ns3_dir.format(i+1)))
            # export_throughput(exp_path, ns3_dir.format(i+1), frame_size, agent_list, csv_dir / "{}_{}_{}.csv".format(v, "throughput", ns3_dir.format(i+1)))
    
        for v in agent_list:
            base_path = exp_path / v / cache_frame_dir
            dirs = []
            dirs.append(exp_path / v / "velodyne")
            dirs.append(exp_path / v / "velodyne_compression")
            for x in base_path.iterdir():
                dirs.append(x)

            # pre-frame
            each_frame_metrics_list = ["cd", "hd", "running_time"]
            for metrics in each_frame_metrics_list:
                for x in base_path.iterdir():
                    file_name = x / "evaluation/{}.npy".format(metrics)
                    if not file_name.exists():
                        continue
                    result_array = np.load(file_name, allow_pickle=True).astype(float)
                    np.savetxt(csv_dir / "{}_{}_{}.csv".format(v, metrics, x.name), result_array, fmt='%s',delimiter=',')
            
            # avg result for vehicles {1, 3, 5, 7}
            avg_metrics_list = ["cd", "hd", "running_time", "obj_pointrcnn_car_0.5_easy_3d", "obj_pointrcnn_car_0.5_mod_3d", "obj_pointrcnn_car_0.5_hard_3d", "obj_pointrcnn_car_0.5_avg_iou"]
            for metrics in avg_metrics_list:
                csv_list = []
                for x in dirs:
                    file_name = x / "evaluation/{}.npy".format(metrics)
                    if not file_name.exists():
                        continue
                    result = get_mean_by_npy(file_name)
                    csv_list.append([x.name, result])
                np.savetxt(csv_dir / "{}_{}_avg.csv".format(v, metrics), csv_list, fmt='%s',delimiter=',')

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total time: {}".format(time.time() - start_time))