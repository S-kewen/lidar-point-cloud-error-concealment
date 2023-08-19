import os
import numpy as np
import torch.utils.data as torch_data
import kitti_utils
import cv2
from PIL import Image
from data_utils import *

USE_INTENSITY = False


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train', mode='TRAIN', npoints=16384, lidar_dir="velodyne"):
        self.split = split
        self.mode = mode
        self.classes = ['Car']
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'object', 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        self.num_sample = self.image_idx_list.__len__()

        self.npoints = npoints

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, lidar_dir)
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return kitti_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def filtrate_objects(self, obj_list):
        type_whitelist = self.classes
        if self.mode == 'TRAIN':
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue

            valid_obj_list.append(obj)
        return valid_obj_list

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        '''
        get a frame by index
        '''
        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)

        # get valid point (projected points should be in image)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])  # points_xyzs
        pts_intensity = pts_lidar[:, 3]  # points_i

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth,
                                             img_shape)  # get point indexs from rgb image [skewen]: filter rgb camera fov

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]

        pts_intensity = pts_intensity[pts_valid_flag]
        # np.concatenate((pts_rect, pts_intensity.reshape((-1, 1))), axis=1).astype(np.float32).tofile(str("/home/skewen/lidar-base-point-cloud-error-concealment/PointRCNN/pointnet2_lib_fg/tools/output/bin/fov.bin"))

        if len(pts_rect) == 0:
            return {'sample_id': -1}
        if self.npoints < len(pts_rect):
            pts_depth = pts_rect[:, 2]
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            #choice = np.random.choice(np.arange(0, len(pts_rect), dtype=np.int32), self.npoints, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
            if self.npoints > len(pts_rect):
                if len(choice) == 0:
                    print("len(pts_rect): {}".format(len(pts_rect)))
                extra_choice = np.random.choice(choice, self.npoints - len(pts_rect),
                                                replace=(self.npoints - len(pts_rect)) > len(choice))
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        ret_pts_rect = pts_rect[choice, :]  # filter points xyz

        original_input = pts_lidar[choice, :]

        ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5] #filter points i

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id}
        sample_info['original_input'] = original_input


        # np.concatenate((ret_pts_rect, ret_pts_features.reshape((-1, 1))), axis=1).astype(np.float32).tofile(str("/home/skewen/lidar-base-point-cloud-error-concealment/PointRCNN/pointnet2_lib_fg/tools/output/bin/fov_downsampling.bin"))
        if self.mode == 'TEST':
            if USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = ret_pts_features
            return sample_info

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))

        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        # prepare input
        if USE_INTENSITY:
            pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = ret_pts_rect

        # generate training labels
        fov_point_xyzis = np.concatenate((ret_pts_rect, ret_pts_features.reshape((-1, 1))), axis=1)
        cls_labels = self.generate_training_labels(pts_input, gt_boxes3d)
        #assert 1==0
        cls_point_xyzi = np.concatenate((fov_point_xyzis, cls_labels.reshape((-1, 1))), axis=1)
        fg_points_gt = cls_point_xyzi[np.where(cls_point_xyzi[:, -1]>0),:4]

        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = ret_pts_rect
        sample_info['cls_labels'] = cls_labels
        # sample_info['fg_points_gt'] = fg_points_gt
        return sample_info

    # @staticmethod
    # def generate_training_labels(pts_rect, gt_boxes3d):
    #     #pts_rect: downsamplinged input
    #     cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
    #     gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
    #     extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
    #     extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
    #     for k in range(gt_boxes3d.shape[0]):
    #         box_corners = gt_corners[k]
    #         fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
    #         cls_label[fg_pt_flag] = 1

    #         # enlarge the bbox3d, ignore nearby points
    #         extend_box_corners = extend_gt_corners[k]
    #         fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
    #         ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
    #         cls_label[ignore_flag] = -1

    #     return cls_label



    # @staticmethod
    # def generate_training_labels(pts_rect, gt_boxes3d):
    #     calibs = {"R0_rect": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "Tr_velo_to_cam": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]} #[skewen]: fixed calib config
    #     cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
    #     for bbox3d in gt_boxes3d:
    #         x, y, z, h, w, l, rotation_y = bbox3d[0], bbox3d[1], bbox3d[2], bbox3d[3], bbox3d[4], bbox3d[5], bbox3d[6]
    #         fg_pt_flag = in_hull(pts_rect, compute_box_3d(calibs, h, w, l, x, y, z, rotation_y)) > 0
    #         print("fg_pt_flag.shape: {}".format(fg_pt_flag.shape))
    #         cls_label[fg_pt_flag] = 1
    #     return cls_label


    @staticmethod
    def generate_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        for bbox3d in gt_corners:
            fg_pt_flag = kitti_utils.in_hull_fg(pts_rect[:, :3], bbox3d)
            cls_label[fg_pt_flag] = 1
        # for c in cls_label:
        #     print("!!!!: {}".format(c))
        #     if c>0:
        #         print("!!!!: {}".format(c))
        return cls_label

    def collate_batch(self, batch):
        batch_size = batch.__len__()
        ans_dict = {}
        if batch[0]["sample_id"] == -1:
            return None
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict