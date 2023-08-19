# LiDAR-Point-Cloud-Error-Concealment
This is an error concealment system for LiDAR point clouds, which can be used to generate experimental results. If you want to generate your own dataset as input, please use our [co-simulator](https://github.com/S-kewen/carla-generator).

You can download datasets here, from: (a) [our co-simulator](https://drive.google.com/file/d/1r5gZ2dFfh7-EKhhnmYpLIlK9aHTo9Oyc/view?usp=sharing) and (b) [KITTI Odometry](https://drive.google.com/file/d/1FEkYaSIWDklQkaZdKXcB0hRaofKNyGJx/view?usp=sharing).

<div align=center>
<!-- <img src="doc/input_output_diagram.svg" width="400px"> -->
<img src="doc/lec_diagram.svg" width="800px">
</div>

In this repository, we provide the following functions:
- Packet loss analysis based on NS-3 network simulator and Gilbert-Elliot model.
- Some point cloud preprocessing, such as downsampling, ground removal, compression, etc.
- A variety of different error concealment approaches, such as Temporal Prediction (TP),  Spatial Interpolation (SI), Temporal Interpolation (TI), Threshold-based LiDAR Error Concealment (TLEC), and LiDAR Error Concealment (LEC).
- Low-level metrics evaluation, such Chamfer and Hausdorff distances, running time, etc.
- High-level metrics evaluation, such object detection accuracy and average IoU.
- Support for CARLA and KITTI datasets.
## Installation
### Requirements
All the codes are tested in the following environment:
* OS (Ubuntu 20.04)
* Conda 22.9.0
* Python 3.6.13/3.7.13/3.9.13
* Cuda 11.7
* Pytorch 1.4.0
## Quick demo
You can download our conda environment [here](https://drive.google.com/drive/folders/1KRNBL0MH5Lpmcs6krxjcFsEfEPaQaNrF?usp=sharing) or follow the steps below to install:
### a. Clone repository
```
git clone https://github.com/S-kewen/lidar-point-cloud-error-concealment
cd lidar-point-cloud-error-concealment
```

### b. Build environment for generation, temporal prediction, and spatial interpolation
```
conda create -n ec39 python==3.9.13 -y
conda activate ec39
pip install -r requirements_ec39.txt
conda install -c conda-forge gmp -y && pip install pycddlib
```

### c. Build environment for temporal interpolation, LEC, and evaluation
```
conda create -n ec37 python==3.7.13 -y
conda activate ec37
pip install -r requirements_ec37.txt
```
#### 1) Install PointINet
```
cd PointINet
pip install -r requirements.txt
cd ..
```
#### 2) Install Pytorch
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
conda install -c conda-forge -c fvcore fvcore -y
conda install -c bottler nvidiacub -y
```
#### 3) Install Kaolin
```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
git checkout v0.1
sed -i '219d' setup.py # delete line 219
python setup.py develop
cd ..
```
#### 4) Install Pytorch3d
```
conda install jupyter -y
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black 'isort<5' flake8 flake8-bugbear flake8-comprehensions
conda install pytorch3d=0.3.0 -c pytorch3d -y
pip install pytorch3d==0.3.0
```
#### 5) Install PyTorchEMD
```
git clone https://github.com/daerduoCarey/PyTorchEMD.git
cd PyTorchEMD
python setup.py install
cd ..
```
#### 6) Install Shape-Measure
```
git clone https://github.com/FengZicai/Shape-Measure.git
cd Shape-Measure
python setup.py install
cd ..
```
### d. Build environment for object detection
```
conda create -n pointrcnn python==3.6.13 -y
conda activate pointrcnn
cd PointRCNN
pip install -r requirements.txt
sh build_and_install.sh
cd ..
```

### e. Run
#### Generating incomplete frame
```
conda activate ec39 && python exp_generating.py --c {your_config_file}
```

#### Temproal Prediction (TP)
```
conda activate ec39 && python exp_temporal_prediction.py --c {your_config_file}
```

#### Spatial Interpolation (SI)
```
conda activate ec39 && python exp_spatial_interpolation.py --c {your_config_file}
```

#### Temproal Interpolation (TI)
```
conda activate ec37 && python exp_temporal_interpolation.py --c {your_config_file}
```

#### Optimal (OPT)
```
conda activate ec37 && python exp_opt.py --c {your_config_file}
```

#### Threshold-based LiDAR Error Concealment (TLEC)
```
conda activate ec37 && python exp_tlec_single.py --c {your_config_file} --tp {your_threshold_Tp} --tn {your_threshold_Tn}
```

#### LiDAR Error Concealment (LEC)
```
conda activate ec37 && python exp_lec.py --c {your_config_file}
```
If you want to train your LEC model, please perform the following training command:
```
conda activate ec37 && python exp_lec_training.py --c {your_config_file}
```

#### Low-level evaluation
```
conda activate ec37 && python exp_evaluation.py --c {your_config_file}
```

#### High-level evaluation
To fit our dataset, we modified [PointRCNN](https://github.com/sshaoshuai/PointRCNN).
```
cd PointRCNN/tools
conda activate pointrcnn && CUDA_VISIBLE_DEVICES=0 python exp_rcnn.py --c {your_config_file}
```

For specific configuration, please refer to [config.yaml](config.yaml).

<!-- ## Citation 
If you find this project useful in your research, please consider cite:
```
XXXX
``` -->

<!-- ## Limitations 
We are happy to improve this project together, please submit your pull request if you fixed these limitations.
- [ ] Calib: Our sensors are installed in a fixed location and can not provide calibration replacement.
- [ ] LiDAR Intensity: The CARLA LiDAR sensor only provides virtual intensity without considering the material. -->

## Contribution
welcome to contribute to this repo, please feel free to contact us with any potential contributions.