import numpy as np
from pathlib import Path
data_dict = {}

base_dir_list = ["result_dsrc_ap", "result_dsrc_ap_lec" ]

dir_list = ["csv_20230413_False_1_6_110_0_600_1", "csv_20230413_False_1_6_110_0_600_2", "csv_20230413_False_1_6_110_0_600_3", "csv_20230413_False_1_6_110_0_600_4", "csv_20230413_False_1_6_110_0_600_5", "csv_20230413_False_1_6_110_0_600_6", "csv_20230413_False_1_6_110_0_600_7", "csv_20230413_False_1_6_110_0_600_8", "csv_20230413_False_1_6_110_0_600_9", "csv_20230413_False_1_6_110_0_600_10"]


for base_dir in base_dir_list:
    for dir in dir_list:
        csv_files = sorted((Path() / base_dir / dir).glob("*_*_avg.csv"))
        for file in csv_files:
            with open(file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(',')
                    key = file.name + "_" + key
                    if key in data_dict:
                        data_dict[key].append(float(value))
                    else:
                        data_dict[key] = [float(value)]

result_dict = {}
for key, values in data_dict.items():
    result_dict[key] = np.mean(values)

for key, value in sorted(result_dict.items()):
    print(f'{key},{value}')
