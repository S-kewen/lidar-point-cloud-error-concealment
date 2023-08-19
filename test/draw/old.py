import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import yaml
import json

with open(Path().cwd().parent / "config.yaml", 'r') as yamlfile:
    m_config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_mean_by_npy(file_name):
    print("get_mean_by_npy: {}".format(file_name))
    return np.mean(np.load(file_name, allow_pickle=True)[:, 1].astype(float))


def cal_metrics(exp_path, interpolation, packet_loss_rate_list):
    caches = ["none", "location_stiching", "mc_stiching", "icp"]
    metrics_list = ["snn-rmse", "runningTime", "acd", "cd", "cd-psnr", "hd"]

    # metrics_list = ["snn-rmse", "runningTime", "acd", "cd", "cd-psnr", "hd", "pointnet++_cls", "obj_pointrcnn_car_0.7_easy_3d", "obj_pointrcnn_car_0.7_mod_3d",
    #                 "obj_pointrcnn_car_0.7_hard_3d", "obj_pointrcnn_car_0.5_easy_3d", "obj_pointrcnn_car_0.5_mod_3d", "obj_pointrcnn_car_0.5_hard_3d"]

    all_list = []
    for metrics in metrics_list:
        result_list = []
        for cache in caches:
            cache_list = []
            for packet_loss_rate in packet_loss_rate_list:
                file_name = Path() / exp_path / "cache_frame" / "receiver_{}_{}".format(cache, packet_loss_rate) / \
                    interpolation / "evaluate/{}.npy".format(metrics)
                # print(file_name)
                result = get_mean_by_npy(file_name)
                print("{}={}".format(file_name, result))
                cache_list.append(result)
            result_list.append(cache_list)
        all_list.append(result_list)

    return all_list


def draw_heat_map(arrays, save_name, title=""):
    x_labels = ["None", "Location-Stiching", "MC-Stiching", "ICP"]
    y_labels = ["Identity", "Align-ICP", "Scene-Flow", "PointINet"]

    harvest = np.array(np.around(arrays, decimals=2))

    plt.xticks(np.arange(len(y_labels)), labels=y_labels,
               rotation=45, rotation_mode="anchor", ha="right")
    plt.yticks(np.arange(len(x_labels)), labels=x_labels)
    plt.title(title)

    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = plt.text(j, i, harvest[i, j], ha="center", va="center", color="w")

    plt.imshow(harvest)
    plt.colorbar()
    plt.tight_layout()

    save_dir = Path() / "figs" / "heat_map"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    plt.savefig(save_dir / save_name)
    # plt.show()
    plt.close()


def draw_column_chart(x_labels, bar_dict, y_label, y_label_size, save_name, title=""):

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

    width = 0.15

    plt.figure(dpi=300)
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.rc('figure', facecolor='w')

    for idx, key in enumerate(bar_dict.keys()):
        if key == "CMI_N":
            mylabel = "$CMI_{new}$"
        else:
            mylabel = key
        plt.bar(x+width*width_coe[idx], bar_dict[key], width, label=mylabel)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xticks(x, x_labels)
    plt.yticks()

    # plt.xlim(x1, x2)  #x1 is min, x2 is max
    # plt.ylim(0, 100.0)

    plt.ylabel(y_label, fontsize=y_label_size)
    plt.yscale("log")

    plt.legend(loc="upper left", fancybox=True, ncol=len(bar_dict))
    plt.tight_layout()
    plt.title(title)
    save_dir = Path() / "figs" / "column_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    plt.savefig(save_dir / save_name)
    # plt.show()
    plt.close()


def draw_line_chart(x_label, x_label_size, x_ticks, y_label, y_label_size, line_labels, data_list, save_name, title=""):
    x = range(len(x_ticks))
    # pl.xlim(-1, 11) # 限定横轴的范围
    # pl.ylim(-1, 110) # 限定纵轴的范围
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


def draw_latency_diagram():
    base_path = Path() / m_config["exp"]["path"] / "ns3"
    latency_list = []
    id_list = []
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        print("[{}] {}".format(i, file_name))
        with open(file_name, "r") as f:
            lines = f.readlines()
            latency = 0.0
            for line in lines:
                json_data = json.loads(line)
                sector_number = int(json_data["segment"])+1
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == "0" and json_data["receiveMac"] == "2":
                    if json_data["type"] == "rx":
                        latency += (json_data["recvTime"] - json_data["sendTime"])*1000
                    else:
                        latency += 100
                    
            latency_list.append(latency/len(lines))
            id_list.append(frame_number)

    x = id_list
    k1 = latency_list
    plt.plot(x, k1)  # s-:方形
    plt.xlabel("Frame Number")  # 横坐标名字
    plt.ylabel("Latency (ms)")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    
    save_dir = Path() / "figs" / "line_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    plt.savefig(save_dir / "latency.png")
    plt.show()


def draw_packet_loss_rate_diagram():
    base_path = Path() / m_config["exp"]["path"] / "ns3"
    packet_size = m_config["simulator"]["sectorSize"]
    packet_loss_rate_list = []
    id_list = []
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        print("[{}] {}".format(i, file_name))
        with open(file_name, "r") as f:
            lines = f.readlines()
            rxCount = 0
            for line in lines:
                json_data = json.loads(line)
                sector_number = int(json_data["segment"])+1
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == "0" and json_data["receiveMac"] == "2":
                    if json_data["type"] == "rx":
                        rxCount += 1
                    
            packet_loss_rate_list.append((1-(rxCount/packet_size))*100)
            id_list.append(frame_number)

    x = id_list
    k1 = packet_loss_rate_list
    plt.plot(x, k1)  # s-:方形
    plt.xlabel("Frame Number")  # 横坐标名字
    plt.ylabel("Packet Loss Rate (%)")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    
    save_dir = Path() / "figs" / "line_chart"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    plt.savefig(save_dir / "packet_loss_rate.png")
    plt.show()
    
def draw_sector_received_diagram():
    base_path = Path() / m_config["exp"]["path"] / "ns3"
    packet_size = m_config["simulator"]["sectorSize"]
    id_list = []
    for i in range(packet_size):
        id_list.append(i+1)
    received_count_list = []
    for i in range(packet_size):
        received_count_list.append(0)
    for i, file_name in enumerate(sorted((base_path).glob("**/*.txt"))):
        print("[{}] {}".format(i, file_name))
        with open(file_name, "r") as f:
            lines = f.readlines()
            rxCount = 0
            for line in lines:
                json_data = json.loads(line)
                sector_number = int(json_data["segment"])+1
                frame_number = int(json_data["frame"])+1
                if json_data["sourceMac"] == "0" and json_data["receiveMac"] == "2":
                    if json_data["type"] == "rx":
                        received_count_list[int(json_data["segment"])] +=1


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

    exp_path = m_config["exp"]["path"]
    
    packet_loss_rate_list = m_config["exp"]["packetLossRateList"]

    x_labels = ["None", "Location-Stiching", "MC-Stiching", "ICP"]

    y_label_list = ["SNN-RMSE", "Running Time (s)", "Asymmetric Chamfer Distance (ACD)",
                    "Chamfer Distance (CD)", "Chamfer Distance-PSNR (dB)", "Hausdorff Dist. (HD)", "Cls Avg IoU", "Car Easy 3D AP (%) [IoU: 70%]", "Car Mod 3D AP (%) [IoU: 70%]", "Car Hard 3D AP (%) [IoU: 70%]", "Car Easy 3D AP (%) [IoU: 50%]", "Car Mod 3D AP (%) [IoU: 50%]", "Car Hard 3D AP (%) [IoU: 50%]"]

    y_label_size_list = [20, 20, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    interpolation_list = m_config["exp"]["interpolationList"]

    all_data_list = []  # (I4, M13, C4, P5)
    for interpolation in interpolation_list:
        all_data_list.append(cal_metrics(exp_path, interpolation, packet_loss_rate_list))

    all_data_list = np.asarray(all_data_list)
    print(all_data_list.shape)

    # start draw column chart
    # for i in range(all_data_list.shape[0]):  # every interpolation
    #     for j in range(all_data_list.shape[1]):  # every meterics
    #         data_list = all_data_list[i, j, :, :].T
    #         bar_dict = {
    #             '0%': data_list[0, :],  # None[NI, NA, NS, NP]
    #             '20%': data_list[1, :],
    #             '40%': data_list[2, :],
    #             '60%': data_list[3, :],
    #             '80%': data_list[4, :],
    #         }
    #         draw_column_chart(x_labels, bar_dict,
    #                           y_label_list[j], y_label_size_list[j], "{}-{}.png".format(interpolation_list[i], y_label_list[j]).replace(":", ""))

    # start draw column chart and heat map
    for i in range(all_data_list.shape[1]):  # every metrics
        for j in range(all_data_list.shape[3]):  # every packet loss rate
            temp_list = all_data_list[:, i, :, j]
            bar_dict = {}
            for i, interpolation in enumerate(interpolation_list):
                bar_dict[interpolation] = temp_list[i]
            print("temp_list: {}".format(temp_list.shape))
            # bar_dict = {
            #     'identity': temp_list[0],  # [C1, C2, C3, C4]
            #     'align-ICP': temp_list[1],
            #     'scene-flow': temp_list[2],
            #     'pointinet': temp_list[3]
            # }
            draw_column_chart(x_labels, bar_dict, y_label_list[i], y_label_size_list[i],
                              "{}_{}.png".format(packet_loss_rate_list[j], y_label_list[i]).replace(":", ""), "")
            draw_heat_map(temp_list.T, "{}_{}.png".format(
                packet_loss_rate_list[j], y_label_list[i]).replace(":", ""), "{}% - {}".format(packet_loss_rate_list[j]*100, y_label_list[i]))

    # start draw line chart
    # y_label_list = ["SNN-RMSE", "Running Time (s)", "Asymmetric Chamfer Distance (ACD)",
    #                 "Chamfer Distance (CD)", "Chamfer Distance-PSNR (dB)", "Hausdorff Dist. (HD)", "Cls Avg IoU (%)", "Car Easy 3D AP (%) [IoU: 70%]", "Car Mod 3D AP (%) [IoU: 70%]", "Car Hard 3D AP (%) [IoU: 70%]", "Car Easy 3D AP (%) [IoU: 50%]", "Car Mod 3D AP (%) [IoU: 50%]", "Car Hard 3D AP (%) [IoU: 50%]"]
    # y_label_size_list = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

    # x_label = "Packet Loss Rate (%)"
    # x_label_size = 15
    # x_ticks = ["0", "20", "40", "60", "80"]

    # line_labels = ["Identity", "Align-ICP", "Scene-Flow", "PointINet"]
    # for i, y_label in enumerate(y_label_list):  # Location-Stiching with all interpolation method
    #     draw_line_chart(x_label, x_label_size, x_ticks, y_label, y_label_size_list[i], line_labels, all_data_list[:,
    #                                                                                                               i, 1, :], "{}_Location-Stiching.png".format(y_label).replace(":", ""), "Cache: Location-Stiching")

    # line_labels = ["None", "Location-Stiching", "MC-Stiching", "ICP"]
    # for i, y_label in enumerate(y_label_list):  # PointINet with all cache method
    #     draw_line_chart(x_label, x_label_size, x_ticks, y_label,
    #                     y_label_size_list[i], line_labels, all_data_list[-1, i, :, :], "{}_PointINet.png".format(y_label).replace(":", ""), "Interpolation: PointINet")


if __name__ == "__main__":
    start_time = time.time()
    main()
    # draw_latency_diagram()
    # draw_packet_loss_rate_diagram()
    # draw_sector_received_diagram()
    print("Total time: {}".format(time.time() - start_time))
