def main():
    # for v in range(1, 8):
    #     result = ""
    #     for i in range(1, 11):
    #         result += "cd /home/skewen/lidar-base-point-cloud-error-concealment && conda activate stitching && python exp_generating.py --c config/config_{}_{}v.yaml && conda activate interpolation && CUDA_VISIBLE_DEVICES=3 python exp_temporal_interpolation.py --c config/config_{}_{}v.yaml && ".format(i, v, i, v)
    #     print(result)

    result = ""
    for v in range(1, 8):
        for i in range(1, 11):
            result += "cd /home/skewen/lidar-base-point-cloud-error-concealment && conda activate interpolation && python exp_evaluation.py --c  config/config_{}_{}v.yaml &&".format(i, v)
    print(result)
if __name__ == "__main__":
    main()