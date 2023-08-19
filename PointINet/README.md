# PointINet: Point Cloud Frame Interpolation Network

## Introduction

The repository contains the source code and pre-trained models of our paper (published on AAAI 2021): `PointINet: Point Cloud Frame Interpolation Network`.

<div align="center">
<img src="https://github.com/ispc-lab/PointINet/blob/main/figs/interpolation.png"  width = "500" height = "500"/>
</div>

## Environment

Our code is developed and tested on the following environment:

- Python 3.6
- PyTorch 1.4.0
- Cuda 10.1
- Numpy 1.19
- kaolin v0.1
- pytorch3d v0.3.0
- Mayavi 4.8.2
- jsoncpp 1.8.3
- wandb 0.13.1

We utilized several open source library to implement the code:

- [kaolin](https://github.com/NVIDIAGameWorks/kaolin/tree/v0.1)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/v0.3.0)
- [PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD) (only for evaluation)
- [Mayavi](https://docs.enthought.com/mayavi/mayavi/) (only for visualization of demo)
- [wandb](https://app.wandb.ai/) (to record training process)

## install command
### pytorch 1.4.1
`conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch`

### kaolin v0.1
conda install cycler

git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git

git checkout v0.1

python setup.py develop

### pytorch3d v0.3.0
https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/INSTALL.md

git checkout v0.3.0

### mayavi
conda install mayavi=4.8.2

pip install mayavi=4.7.1

pip install PyQt5=5.15.6

jupyter nbextension install --py mayavi --user

jupyter nbextension enable --py mayavi --user

### nuscenes & vtk
pip install nuscenes-devkit=1.1.9

pip install vtk=8.2.0

### PyQt DEBUG MODE
`
export QT_DEBUG_PLUGINS=1
`

### jsoncpp
`
conda install jsoncpp=1.8.3
`
### wandb
`pip install wandb=0.13.1`

### install issue
* ModuleNotFoundError: No module named 'vtkIOParallelPython'
`conda install jsoncpp=1.8.3`

## Usage

### Dataset

We utilize two large scale outdoor LiDAR dataset:

- [Kitti odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [nuScenes dataset](https://www.nuscenes.org/)

To facilitate the implementation, we split the LiDAR point clouds in nuScenes dataset by scenes and the results are saved in `data/scene-split`. Besides, all of the LiDAR files in nuScenes dataset are stored in one single folder (include sweeps and samples).

For the pre-training of FlowNet3D, please refer to [FlowNet3D](https://github.com/xingyul/flownet3d) to download the pre-processed dataset (Flythings3D and Kitti scene flow dataset).

### Demo

We provide a demo to visualize the result, please run

```bash
python demo.py --is_save IS_SAVE --visualize VISUALIZE
```

### Training

#### Training of FlowNet3D

To train FlowNet3D, firstly train on Flythings3D dataset

```bash
python train_sceneflow.py --batch_size BATCH_SIZE --gpu GPU --dataset Flythings3D --root DATAROOT --save_dir CHECKPOINTS_SAVE_DIR --train_type init
```

Then train it on Kitti scene flow dataset

```bash
python train_sceneflow.py --batch_size BATCH_SIZE --gpu GPU --dataset Kitti --root DATAROOT --pretrain_model PRETRAIN_MODEL --save_dir CHECKPOINTS_SAVE_DIR --train_type init
```

After that train it on Kitti odometry dataset based on the model pretrained on Kitti scene flow dataset.

```bash
python train_sceneflow.py --batch_size BATCH_SIZE --gpu GPU --dataset Kitti --root DATAROOT --pretrain_model PRETRAIN_MODEL --save_dir CHECKPOINTS_SAVE_DIR --train_type refine
```

Also train it on nuScenes dataset based on the model pretrained on Kitti scene flow dataset.

```bash
python train_sceneflow.py --batch_size BATCH_SIZE --gpu GPU --dataset nuscenes --root DATAROOT --pretrain_model PRETRAIN_MODEL --save_dir CHECKPOINTS_SAVE_DIR --train_type refine
```

#### Training of PointINet

We only train the PointINet on Kitti odometry dataset, run

```bash
python train_interp.py --batch_size BATCH_SIZE --gpu GPU --dataset kitti --root DATAROOT --pretrain_model FLOWNET3D_PRETRAIN_MODEL --freeze 1
```

### Testing

To test on Kitti odometry dataset, run

```bash
python test.py --gpu GPU --dataset kitti --root DATAROOT --pretrain_model POINTINET_PRETRAIN_MODEL --pretrain_flow_model FLOWNET3D_PRETRAIN_MODEL
```

To test on nuScenes dataset, run

```bash
python test.py --gpu GPU --dataset nuscenes --root DATAROOT --pretrain_model POINTINET_PRETRAIN_MODEL --pretrain_flow_model FLOWNET3D_PRETRAIN_MODEL --scenelist TEST_SCENE_LIST
```

## Citation

    @InProceedings{Lu2020_PointINet,
        author = {Lu, Fan and Chen, Guang and Qu, Sanqing and Li, Zhijun and Liu, Yinlong and Knoll, Alois},
        title = {PointINet: Point Cloud Frame Interpolation Network},
        booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
        year = {2021}
    }
