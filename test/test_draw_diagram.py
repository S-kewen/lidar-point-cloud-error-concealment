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


def draw_heat_map(arrays, save_name, title=""):
    import matplotlib.pyplot as plt
    x_labels = ["None", "Location-Stiching", "MC-Stiching", "ICP"]
    y_labels = ["Identity", "Align-ICP", "Scene-Flow", "PointINet"]

    harvest = np.array(np.around(arrays, decimals=2))

    plt.xticks(np.arange(len(y_labels)), labels=y_labels,
               rotation=45, rotation_mode="anchor", ha="right")
    plt.yticks(np.arange(len(x_labels)), labels=x_labels)
    plt.title(title)

    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = plt.text(j, i, harvest[i, j],
                            ha="center", va="center", color="w")

    plt.imshow(harvest)
    plt.colorbar()
    plt.tight_layout()

    save_dir = Path() / "figs" / "heat_map"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    plt.savefig(save_dir / save_name)
    # plt.show()
    plt.close()


def draw_column_chart(x_labels, bar_dict, y_label, y_label_size, bar_label_list, save_name, title=""):
    import matplotlib.pyplot as plt

    width_coe = []
    my_length = len(bar_dict.keys())
    if my_length % 2 == 0:
        for i in range(int(my_length / -2), int(my_length / 2 + 1)):
            if i < 0:
                width_coe.append(i + 0.5)
            elif i > 0:
                width_coe.append(i - 0.5)
    else:
        for i in range(int((my_length - 1) / -2), int((my_length - 1) / 2 + 1)):
            width_coe.append(i)

    x = np.arange(len(x_labels))

    width = 0.1 #0.15

    plt.figure(dpi=300)
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.rc('figure', facecolor='w')

    for idx, key in enumerate(bar_dict.keys()):
        mylabel = bar_label_list[idx]
        plt.bar(x+width*width_coe[idx], bar_dict[key], width, label=mylabel)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xticks(x, x_labels)
    plt.yticks()

    # plt.xlim(x1, x2)  #x1 is min, x2 is max
    # plt.ylim(0, 100.0)

    plt.ylabel(y_label, fontsize=y_label_size)
    plt.yscale("log")

    plt.legend(loc="upper left", fancybox=True)#, ncol=len(bar_dict) how many item each row
    plt.tight_layout()
    plt.title(title)
    save_dir = Path() / "figs" / "column_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    plt.savefig(save_dir / save_name)
    # plt.show()
    plt.close()


def draw_line_chart(x_label, x_label_size, x_ticks, y_label, y_label_size, line_labels, data_list, save_name, title=""):
    import matplotlib.pyplot as plt
    x = range(len(x_ticks))
    for i, data in enumerate(data_list):
        plt.plot(x, data, label=line_labels[i])

    plt.legend()
    plt.xticks(x, x_ticks)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(x_label, fontsize=x_label_size)  # X轴标签
    plt.ylabel(y_label, fontsize=y_label_size)  # Y轴标签
    plt.title(title)  # 标题

    save_dir = Path() / "figs" / "line_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    plt.savefig(save_dir / save_name)
    # plt.show()
    plt.close()


def draw_latency_diagram(exp_path, ns3_dir, source_mac, receive_mac):
    import matplotlib.pyplot as plt
    base_path = exp_path / ns3_dir
    csv_dir = Path().cwd() / "csv_{}_{}".format(base_path.parent.parent.parent.name, ns3_dir)
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
    
    csv_list = []
    latency_list = []
    id_list = []
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        with open(file_name, "r") as f:
            lines = f.readlines()
            latency = 0.0
            for line in lines:
                json_data = json.loads(line)
                sector_number = int(json_data["segment"])+1
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == str(source_mac) and json_data["receiveMac"] == str(receive_mac):
                    if json_data["type"] == "rx":
                        latency += (json_data["recvTime"] - json_data["sendTime"])*1000

            result = latency/len(lines)
            latency_list.append(result)
            id_list.append(frame_number)
            csv_list.append([str(frame_number), result])
    np.savetxt(csv_dir / "latency.csv", csv_list, fmt='%s',delimiter=',')

def draw_packet_loss_rate_diagram(exp_path, sector_size, ns3_dir, source_mac, receive_mac):
    base_path = exp_path / ns3_dir
    csv_dir = Path().cwd() / "csv_{}".format(base_path.parent.parent.parent.name)
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
    
    packet_loss_rate_list = []
    id_list = []
    csv_list = []
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
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
            id_list.append(frame_number)
            csv_list.append([str(frame_number), result])
            

    np.savetxt(csv_dir / "{}_packet_loss_rate_{}.csv".format(exp_path.name, ns3_dir), csv_list, fmt='%s',delimiter=',')
    print("packet_loss_rate_list: {}".format(np.mean(packet_loss_rate_list)))
    
def draw_avg_packet_loss_rate_diagram(exp_path, agent_list, sector_size, ns3_dir, receive_mac, frame_size):
    packet_loss_rate_list = []
    for node_id, agent in enumerate(agent_list):
        base_path = exp_path / agent / ns3_dir
        if not base_path.exists():
            continue
        file_count = 0
        for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
            with open(file_name, "r") as f:
                lines = f.readlines()
                rxCount = 0
                for line in lines:
                    json_data = json.loads(line)
                    if json_data["sourceMac"] == str(node_id) and json_data["receiveMac"] == str(receive_mac):
                        if json_data["type"] == "rx":
                            rxCount += 1

                result = (1-(rxCount/sector_size))*100
                packet_loss_rate_list.append(result)
                file_count+=1
        for x in range(frame_size - file_count):
            packet_loss_rate_list.append(100)
    print("{} {}v packet_loss_rate_list: {}".format(exp_path.parent.name, receive_mac, np.mean(packet_loss_rate_list)))
    

def draw_distance_diagram_frame(exp_path, frame_size, agent_list, edge_server_location):
    import matplotlib.pyplot as plt
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


# def draw_packet_loss_rate_diagram_distance():
#     import matplotlib.pyplot as plt
#     from scipy.spatial.distance import euclidean
#     edge_server_location = np.array([-23.50, 16.62, 0.005])
    
#     base_path = Path() / m_config["exp"]["path"] / "ns3"
#     packet_size = m_config["exp"]["sector_size"]
#     packet_loss_rate_list = []
#     x_list = []
#     for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
#         #print("[{}] {}".format(i, file_name))
#         with open(file_name, "r") as f:
#             npy_file_name = file_name.parent.parent / "location" / "{}.npy".format(file_name.stem)
#             assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)
            
#             npy_location = np.load(npy_file_name, allow_pickle=True).item()
#             location = np.asarray([npy_location["x"], npy_location["y"], npy_location["z"]])
            
#             lines = f.readlines()
#             rxCount = 0
#             for line in lines:
#                 json_data = json.loads(line)
#                 frame_number = int(json_data["frame"])+1
#                 if json_data["sourceMac"] == "0" and json_data["receiveMac"] == "1":
#                     if json_data["type"] == "rx":
#                         rxCount += 1

#             result = (1-(rxCount/packet_size))*100
#             print("[{}] {} = {}".format(i, file_name, rxCount))
#             packet_loss_rate_list.append(result)
#             x_list.append(euclidean(edge_server_location, location))

#     x = x_list
#     k1 = packet_loss_rate_list
#     plt.plot(x, k1)  # s-:方形
#     plt.xlabel("Distance (m)")  # 横坐标名字
#     plt.ylabel("Packet Loss Rate (%)")  # 纵坐标名字
#     plt.legend(loc="best")  # 图例

#     save_dir = Path() / "figs" / "line_chart"
#     if not save_dir.exists():
#         save_dir.mkdir(parents=True)
#     plt.savefig(save_dir / "packet_loss_rate_distance.png")
#     plt.show()
    
    
# def draw_throughpu_diagram_distance():
#     import matplotlib.pyplot as plt
#     from scipy.spatial.distance import euclidean
#     edge_server_location = np.array([-23.50, 16.62, 0.005])
    
#     base_path = Path() / m_config["exp"]["path"] / "ns3"
#     packet_size = m_config["exp"]["sector_size"]
#     print(packet_size)
#     y_list = []
#     x_list = []
#     total_size = 0.0
#     g_firstReceived = 1.0
#     g_lastReceived = 0.0
#     for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
#         #print("[{}] {}".format(i, file_name))
#         with open(file_name, "r") as f:
#             npy_file_name = file_name.parent.parent / "location" / "{}.npy".format(file_name.stem)
#             assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)
            
#             npy_location = np.load(npy_file_name, allow_pickle=True).item()
#             location = np.asarray([npy_location["x"], npy_location["y"], npy_location["z"]])
            
#             lines = f.readlines()
#             for line in lines:
#                 json_data = json.loads(line)
#                 frame_number = int(json_data["frame"])+1
#                 if json_data["sourceMac"] == "0" and json_data["receiveMac"] == "1":
#                     if json_data["type"] == "rx":
#                         total_size += float(json_data["size"])*8
#                 if float(json_data["recvTime"]) < g_firstReceived:
#                     g_firstReceived = float(json_data["recvTime"])
#                 if float(json_data["recvTime"]) > g_lastReceived:
#                     g_lastReceived = float(json_data["recvTime"])

#             result = total_size/(g_lastReceived - g_firstReceived)/1e6
#             #"Average Throughput:\t" << (double(g_rxPackets)*(double(packetSize)*8)/double(g_lastReceived.GetSeconds() - g_firstReceived.GetSeconds()))/1e6 << " Mbps"
#             y_list.append(result)
#             x_list.append(euclidean(edge_server_location, location))

#     x = x_list
#     k1 = y_list
#     plt.plot(x, k1)  # s-:方形
#     plt.xlabel("Distance (m)")  # 横坐标名字
#     plt.ylabel("Throughput (Mbps)")  # 纵坐标名字
#     plt.legend(loc="best")  # 图例

#     save_dir = Path() / "figs" / "line_chart"
#     if not save_dir.exists():
#         save_dir.mkdir(parents=True)
#     plt.savefig(save_dir / "throughput_distance.png")
#     plt.show()
    
# def draw_throughpu_diagram_distance_frame():
#     import matplotlib.pyplot as plt
#     from scipy.spatial.distance import euclidean
#     edge_server_location = np.array([-23.50, 16.62, 0.005])
    
#     base_path = Path() / m_config["exp"]["path"] / "ns3"
#     packet_size = m_config["exp"]["sector_size"]
#     print(packet_size)
#     y_list = []
#     x_list = []
#     for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
#         #print("[{}] {}".format(i, file_name))
#         total_size = 0.0
#         g_firstReceived = 1.0
#         g_lastReceived = 0.0
#         with open(file_name, "r") as f:
#             npy_file_name = file_name.parent.parent / "location" / "{}.npy".format(file_name.stem)
#             assert npy_file_name.exists(), "npy file not exists: {}".format(npy_file_name)
            
#             npy_location = np.load(npy_file_name, allow_pickle=True).item()
#             location = np.asarray([npy_location["x"], npy_location["y"], npy_location["z"]])
            
#             lines = f.readlines()
#             for line in lines:
#                 json_data = json.loads(line)
#                 frame_number = int(json_data["frame"])+1
#                 if json_data["sourceMac"] == "0" and json_data["receiveMac"] == "1":
#                     if json_data["type"] == "rx":
#                         total_size += float(json_data["size"])*8
#                 if float(json_data["recvTime"]) < g_firstReceived:
#                     g_firstReceived = float(json_data["recvTime"])
#                 if float(json_data["recvTime"]) > g_lastReceived:
#                     g_lastReceived = float(json_data["recvTime"])

#             result = total_size/(g_lastReceived - g_firstReceived)/1e6
#             #"Average Throughput:\t" << (double(g_rxPackets)*(double(packetSize)*8)/double(g_lastReceived.GetSeconds() - g_firstReceived.GetSeconds()))/1e6 << " Mbps"
#             y_list.append(result)
#             x_list.append(euclidean(edge_server_location, location))

#     x = x_list
#     k1 = y_list
#     plt.plot(x, k1)  # s-:方形
#     plt.xlabel("Distance (m)")  # 横坐标名字
#     plt.ylabel("Throughput (Mbps)")  # 纵坐标名字
#     plt.legend(loc="best")  # 图例

#     save_dir = Path() / "figs" / "line_chart"
#     if not save_dir.exists():
#         save_dir.mkdir(parents=True)
#     plt.savefig(save_dir / "throughput_distance_frame.png")
#     plt.show()
    
# def draw_throughpu_diagram(exp_path, ns3_dir, source_mac, receive_mac):
#     import matplotlib.pyplot as plt
#     base_path = exp_path / ns3_dir
#     y_list = []
#     x_list = []
#     total_size = 0.0
#     g_firstReceived = 1.0
#     g_lastReceived = -1.0
    
#     csv_list = []
#     for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
#         #print("[{}] {}".format(i, file_name))
#         with open(file_name, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 json_data = json.loads(line)
#                 frame_number = int(json_data["frame"])+1
#                 if json_data["sourceMac"] == str(source_mac) and json_data["receiveMac"] == str(receive_mac):
#                     if json_data["type"] == "rx":
#                         total_size += float(json_data["size"])*8
#                 if float(json_data["recvTime"]) < g_firstReceived:
#                     g_firstReceived = float(json_data["recvTime"])
#                 if float(json_data["recvTime"]) > g_lastReceived:
#                     g_lastReceived = float(json_data["recvTime"])

#             result = total_size/(g_lastReceived - g_firstReceived)/1e6
#             #"Average Throughput:\t" << (double(g_rxPackets)*(double(packetSize)*8)/double(g_lastReceived.GetSeconds() - g_firstReceived.GetSeconds()))/1e6 << " Mbps"
#             y_list.append(result)
#             x_list.append(frame_number)
#             csv_list.append([str(frame_number), result])
#     np.savetxt(csv_dir / "throughpu.csv", csv_list, fmt='%s',delimiter=',')
    
#     x = x_list
#     k1 = y_list
#     plt.plot(x, k1)  # s-:方形
#     plt.xlabel("Frame Number")  # 横坐标名字
#     plt.ylabel("Throughput (Mbps)")  # 纵坐标名字
#     plt.legend(loc="best")  # 图例

#     save_dir = Path() / "figs" / "line_chart"
#     if not save_dir.exists():
#         save_dir.mkdir(parents=True)
#     plt.savefig(save_dir / "throughput.png")
#     plt.show()


def draw_throughput_diagram_frame_receiver(exp_path, ns3_dir, frame_size, agent_list):
    csv_dir = Path().cwd() / "csv_{}_{}".format(exp_path.parent.name, ns3_dir)
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
    y_list = []
    x_list = []
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
        y_list.append(result)
        x_list.append(frame_number)
        csv_list.append([str(frame_number), result])
        np.savetxt(csv_dir / "throughpu_frame_receiver.csv", csv_list, fmt='%s',delimiter=',')
  
def draw_throughput_diagram_frame(exp_path, ns3_dir, source_mac, receive_mac):
    import matplotlib.pyplot as plt
    base_path = exp_path / ns3_dir
    csv_dir = Path().cwd() / "csv_{}".format(base_path.parent.parent.parent.name)
    y_list = []
    x_list = []
    csv_list = []
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        total_size = 0.0
        g_firstReceived = 1.0
        g_lastReceived = -1.0
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                json_data = json.loads(line)
                frame_number = int(json_data["frame"])+1
                if str(json_data["sourceMac"]) == str(source_mac) and str(json_data["receiveMac"]) == str(receive_mac):
                    if json_data["type"] == "rx":
                        total_size += float(json_data["size"])*8
                if float(json_data["recvTime"]) < g_firstReceived:
                    g_firstReceived = float(json_data["recvTime"])
                if float(json_data["recvTime"]) > g_lastReceived:
                    g_lastReceived = float(json_data["recvTime"])

            result = total_size/(g_lastReceived - g_firstReceived)/1e6
            #"Average Throughput:\t" << (double(g_rxPackets)*(double(packetSize)*8)/double(g_lastReceived.GetSeconds() - g_firstReceived.GetSeconds()))/1e6 << " Mbps"
            y_list.append(result)
            x_list.append(frame_number)
            csv_list.append([str(frame_number), result])
    np.savetxt(csv_dir / "throughpu_frame.csv", csv_list, fmt='%s',delimiter=',')

    x = x_list
    k1 = y_list
    plt.plot(x, k1)  # s-:方形
    plt.xlabel("Frame Number")  # 横坐标名字
    plt.ylabel("Throughput (Mbps)")  # 纵坐标名字
    plt.legend(loc="best")  # 图例

    save_dir = Path() / "figs" / "line_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    plt.savefig(save_dir / "throughput_frame.png")
    plt.show()


def export_sinr_csv(exp_path, ns3_dir, source_mac, receive_mac):
    base_path = exp_path / ns3_dir
    csv_dir = Path().cwd() / "csv_{}".format(base_path.parent.parent.parent.name)
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
    
    y_axis_list = []
    id_list = []
    csv_list = []
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        signal, noise, snr = 0.0, 0.0, 0.0
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                json_data = json.loads(line)
                frame_number = int(json_data["frame"])+1
                if str(json_data["sourceMac"]) == str(source_mac) and str(json_data["receiveMac"]) == str(receive_mac):
                    signal += float(json_data["signal"])
                    noise += float(json_data["noise"])
                    
                    snr += math.log(10, float(json_data["signal"])/float(json_data["noise"]))*10

            result = snr/len(lines)
            y_axis_list.append(result)
            id_list.append(frame_number)
            csv_list.append([str(frame_number), result])
            
    np.savetxt(csv_dir / "{}_sinr.csv_{}".format(exp_path.name, ms3_dir), csv_list, fmt='%s',delimiter=',')
    print("y_axis_list: {}".format(np.mean(y_axis_list)))


def draw_sector_received_diagram(exp_path, sector_size, ns3_dir, source_mac, receive_mac):
    import matplotlib.pyplot as plt
    base_path = exp_path / ns3_dir
    id_list = []
    for i in range(sector_size):
        id_list.append(i+1)
    received_count_list = []
    for i in range(sector_size):
        received_count_list.append(0)
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        with open(file_name, "r") as f:
            lines = f.readlines()
            rxCount = 0
            for line in lines:
                json_data = json.loads(line)
                sector_number = int(json_data["segment"])+1
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == str(source_mac) and json_data["receiveMac"] == str(receive_mac):
                    if json_data["type"] == "rx":
                        received_count_list[int(json_data["segment"])] += 1

    x = id_list
    k1 = received_count_list
    plt.plot(x, k1)  # s-:方形
    plt.xlabel("Sector Number")  # 横坐标名字
    plt.ylabel("Received Count")  # 纵坐标名字
    plt.legend(loc="best")  # 图例

    save_dir = Path() / "figs" / "line_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    plt.savefig(save_dir / "sector_received_count.png")
    plt.show()



def main():
    # v_list = ["training", "CAR2", "CAR3", "CAR4", "CAR5", "CAR6", "CAR7"]
    # dataset_list = ["20230413_False_1_6_110_0_600_1", "20230413_False_1_6_110_0_600_2", "20230413_False_1_6_110_0_600_3", "20230413_False_1_6_110_0_600_4", "20230413_False_1_6_110_0_600_5", "20230413_False_1_6_110_0_600_6", "20230413_False_1_6_110_0_600_7", "20230413_False_1_6_110_0_600_8", "20230413_False_1_6_110_0_600_9", "20230413_False_1_6_110_0_600_10"]
    
    v_list = ["training", "CAR2", "CAR3"]
    dataset_list = ["kitti_odometry_600"]
    
    for dataset in dataset_list:
        exp_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/{}/object".format(dataset)
        cache_frame_dir = "cache_frame_best"
        for v in v_list:
            base_path = exp_path / v / cache_frame_dir
            dirs = []
            dirs.append(exp_path / v / "velodyne")
            dirs.append(exp_path / v / "velodyne_compression")
            for x in base_path.iterdir():
                dirs.append(x)
            
            csv_dir = Path().cwd() / "csv_{}_{}".format(dataset, cache_frame_dir)
            my_util.create_dir(csv_dir)

            each_frame_metrics_list = ["cd", "hd", "running_time"]
            
            avg_metrics_list = ["cd", "hd", "running_time", "obj_pointrcnn_car_0.5_easy_3d", "obj_pointrcnn_car_0.5_mod_3d", "obj_pointrcnn_car_0.5_hard_3d", "obj_pointrcnn_car_0.5_avg_iou"]

            # pre-frame
            for metrics in each_frame_metrics_list:
                for x in base_path.iterdir():
                    file_name = x / "evaluation/{}.npy".format(metrics)
                    if not file_name.exists():
                        continue
                    result_array = np.load(file_name, allow_pickle=True).astype(float)
                    np.savetxt(csv_dir / "{}_{}_{}.csv".format(v, metrics, x.name), result_array, fmt='%s',delimiter=',')
            
            # avg result for vehicles {1, 3, 5, 7}
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
    
    # sector_size = 180
    # ns3_dir = "ns3_nr_c_v2x_{}" # CHECK !!!!!!!!!!!
    # source_mac = 1
    # receive_mac = 2 # CHECK
    # frame_size = 600
    
    # # dataset_list = ["20230413_False_1_6_110_0_600_1", "20230413_False_1_6_110_0_600_2", "20230413_False_1_6_110_0_600_3", "20230413_False_1_6_110_0_600_4", "20230413_False_1_6_110_0_600_5", "20230413_False_1_6_110_0_600_6", "20230413_False_1_6_110_0_600_7", "20230413_False_1_6_110_0_600_8", "20230413_False_1_6_110_0_600_9", "20230413_False_1_6_110_0_600_10"]
    # # agent_list = ["training", "CAR2", "CAR3", "CAR4", "CAR5", "CAR6", "CAR7"]
    
    # dataset_list = ["kitti_odometry_600"]
    # agent_list = ["training", "CAR2", "CAR3"]
    # # for i, dataset in enumerate(dataset_list):
    # #     exp_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/{}/object".format(dataset) # CHECK
    # #     agent_list = ["training", "CAR2", "CAR3", "CAR4", "CAR5", "CAR6", "CAR7"]
    # #     for i in [1, 2, 3, 4, 5, 6, 7]:
    # #         draw_avg_packet_loss_rate_diagram(Path() / exp_path, agent_list, sector_size, ns3_dir.format(i), i, frame_size)
    
    # # A0
    # for i, dataset in enumerate(dataset_list):
    #     exp_path = Path() / "/mnt/data2/skewen/kittiGenerator/output/{}/object".format(dataset) # CHECK
    #     for i in [1, 2, 3]: # , 4, 5, 6, 7
    #         # draw_latency_diagram(exp_path / "training", ns3_dir.format(i), 0, i)
    #         draw_packet_loss_rate_diagram(exp_path / agent_list[i-1], sector_size, ns3_dir.format(i), i-1, i)
    #         # draw_throughput_diagram_frame_receiver(exp_path, ns3_dir.format(i), frame_size, agent_list)
    #         # export_sinr_csv(exp_path / agent_list[i-1], ns3_dir.format(i), i-1, i)

    
    # draw_distance_diagram_frame(exp_path, frame_size, agent_list, np.asarray([-64, 20.33, 0.0]))
    # draw_sector_received_diagram(exp_path, sector_size, "ns3_nr_c_v2x_2", source_mac, receive_mac)
    print("Total time: {}".format(time.time() - start_time))