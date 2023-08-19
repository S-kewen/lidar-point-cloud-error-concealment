import open3d as o3d
import numpy as np
import math


class EvalUtil(object):
    def __init__(self):
        pass

    def get_euclidean_distance(p1, p2):
        return np.linalg.norm(np.asarray(p1) - np.asarray(p2))
    
    def get_point_cloud_by_xyzis(pointer_xyzis):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pointer_xyzis[:, :3])
        return point_cloud
    
    def cal_snn_rmse_by_xyzis(source_points_xyzi, target_points_xyzi):
        if source_avg_distance is None or target_avg_distance is None:
            source = EvalUtil.get_point_cloud_by_xyzis(source_points_xyzi)
            target = EvalUtil.get_point_cloud_by_xyzis(target_points_xyzi)

            source_distances = np.asarray(source.compute_point_cloud_distance(target))
            source_avg_distance = np.sum(np.square(source_distances))/len(source.points)  # MSE(P, Q)

            target_distances = np.asarray(target.compute_point_cloud_distance(source))
            target_avg_distance = np.sum(np.square(target_distances))/len(target.points)  # MSE(P, Q)

        return math.sqrt((source_avg_distance + target_avg_distance)/2)

    def cal_acd_by_xyzis(source_points_xyzi, target_points_xyzi, source_distances=None):
        source = EvalUtil.get_point_cloud_by_xyzis(source_points_xyzi)

        if source_distances is None:
            target = EvalUtil.get_point_cloud_by_xyzis(target_points_xyzi)
            source_distances = np.asarray(source.compute_point_cloud_distance(target))
        return np.sum(np.square(source_distances))/len(source.points)

    def cal_cd_by_xyzis(source_points_xyzi, target_points_xyzi, acd_source=None, target_distances=None):
        if acd_source is None:
            acd_source = EvalUtil.cal_acd_by_xyzis(source_points_xyzi, target_points_xyzi)
        if target_distances is None:
            source = EvalUtil.get_point_cloud_by_xyzis(source_points_xyzi)
            target = EvalUtil.get_point_cloud_by_xyzis(target_points_xyzi)
            np.asarray(target.compute_point_cloud_distance(source))
        return (acd_source + EvalUtil.cal_acd_by_xyzis(target_points_xyzi, source_points_xyzi, target_distances))/2
    
    
    def cal_hd_by_xyzis_open3d(source_points_xyzi, target_points_xyzi, target_distances=None):
        source = EvalUtil.get_point_cloud_by_xyzis(source_points_xyzi)
        target = EvalUtil.get_point_cloud_by_xyzis(target_points_xyzi)
        source_distances = np.asarray(source.compute_point_cloud_distance(target))
        if target_distances is None:
            target_distances = np.asarray(target.compute_point_cloud_distance(source))
        return np.max([np.max(source_distances), np.max(target_distances)])

    def cal_cd_psnr_by_xyzis(source_points_xyzi, target_points_xyzi, cd=None):
        point1 = [target_points_xyzi[:, 0].min(), target_points_xyzi[:, 1].min(), target_points_xyzi[:, 2].min()]
        point2 = [target_points_xyzi[:, 0].max(), target_points_xyzi[:, 1].max(), target_points_xyzi[:, 2].max()]

        max_diameter = EvalUtil.get_euclidean_distance(point1, point2)
        if cd is None:
            cd = EvalUtil.cal_cd_by_xyzis(source_points_xyzi, target_points_xyzi)
        result = np.square(max_diameter) / cd
        return 10 * math.log(result, 10)

    def cal_hd_by_xyzis(source_points_xyzi, target_points_xyzi):
        from hausdorff import hausdorff_distance
        return hausdorff_distance(source_points_xyzi, target_points_xyzi, distance='euclidean')

    def cal_emd_by_xyzis(source_points_xyzi, target_points_xyzi):
        import torch
        from ShapeMeasure.distance import EMDLoss

        emd_util = EMDLoss()
        p1 = torch.from_numpy(source_points_xyzi).cuda()  # .double()
        p2 = torch.from_numpy(target_points_xyzi).cuda()  # .double()

        p1.requires_grad = True
        p2.requires_grad = True

        emd_list = emd_util(p1, p2)
        return torch.mean(emd_list)