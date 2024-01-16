import torch
import numpy as np
import copy
import open3d

def draw_registration_result(source, target, trans):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(trans)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def draw_registration_result_key(source, target, key_source, key_target):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    key_source.paint_uniform_color([1, 0, 0])
    key_target.paint_uniform_color([0, 1, 0])
    # source_temp.translate((3, 0, 0), relative=True)
    # key_source.translate((3, 0, 0), relative=True)

    open3d.visualization.draw_geometries([source_temp, target_temp, key_source, key_target])

def draw_registration_corr_in(source, target, source_keypoint, target_keypoint, transformation, correspondence_set):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_keypoint_transform = copy.deepcopy(target_keypoint)
    source_keypoint_temp = copy.deepcopy(source_keypoint)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    target_keypoint_transform.transform(transformation)
    target_keypoint_transform = np.array(target_keypoint_transform.points).astype(np.float32)
    source_np = np.array(source_keypoint.points).astype(np.float32)
    correspondence = np.array(correspondence_set).astype(np.int32)
    distance = np.sqrt(
        np.sum(np.power(target_keypoint_transform[correspondence[:, 1]] - source_np[correspondence[:, 0]], 2), axis=1))
    correspondence_inlier = correspondence[distance <= 0.1]
    correspondence_outlier = correspondence[distance > 0.1]
    print("correspondence_inlier",correspondence_inlier.shape)
    correspondence_inlier = open3d.utility.Vector2iVector(correspondence_inlier)
    correspondence_outlier = open3d.utility.Vector2iVector(correspondence_outlier)

    source_temp.translate((5, 0, 0), relative=True)
    source_keypoint_temp.translate((5, 0, 0), relative=True)
    # source_temp.transform(transformation)
    inlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp,
                                                                                       target_keypoint,
                                                                                       correspondence_inlier)
    inlier_corr_line.paint_uniform_color([0, 1, 0])
    outlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp,
                                                                                        target_keypoint,
                                                                                        correspondence_outlier)
    outlier_corr_line.paint_uniform_color([1, 0, 0])
    open3d.visualization.draw_geometries([source_temp, target_temp, inlier_corr_line])

def draw_registration_corr(source, target, source_keypoint, target_keypoint, transformation, correspondence_set):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_keypoint_transform = copy.deepcopy(target_keypoint)
    source_keypoint_temp = copy.deepcopy(source_keypoint)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    target_keypoint_transform.transform(transformation)
    target_keypoint_transform = np.array(target_keypoint_transform.points).astype(np.float32)
    source_np = np.array(source_keypoint.points).astype(np.float32)
    correspondence = np.array(correspondence_set).astype(np.int32)
    distance = np.sqrt(
        np.sum(np.power(target_keypoint_transform[correspondence[:, 1]] - source_np[correspondence[:, 0]], 2), axis=1))
    correspondence_inlier = correspondence[distance <= 0.1]
    correspondence_outlier = correspondence[distance > 0.1]
    print("correspondence_inlier",correspondence_inlier.shape)
    correspondence_inlier = open3d.utility.Vector2iVector(correspondence_inlier)
    correspondence_outlier = open3d.utility.Vector2iVector(correspondence_outlier)

    source_temp.translate((5, 0, 0), relative=True)
    source_keypoint_temp.translate((5, 0, 0), relative=True)
    # source_temp.transform(transformation)
    inlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp,
                                                                                       target_keypoint,
                                                                                       correspondence_inlier)
    inlier_corr_line.paint_uniform_color([0, 1, 0])
    outlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp,
                                                                                        target_keypoint,
                                                                                        correspondence_outlier)
    outlier_corr_line.paint_uniform_color([1, 0, 0])
    open3d.visualization.draw_geometries([source_temp, target_temp, inlier_corr_line])

def draw_registration_corr2(source, target, source_keypoint, target_keypoint, transformation):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_keypoint_transform = copy.deepcopy(target_keypoint)
    source_keypoint_temp = copy.deepcopy(source_keypoint)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    target_keypoint_transform.transform(transformation)
    target_keypoint_transform = np.array(target_keypoint_transform.points).astype(np.float32)
    source_np = np.array(source_keypoint.points).astype(np.float32)
    distance = np.sqrt(
        np.sum(np.power(target_keypoint_transform - source_np, 2), axis=1))
    correspondence = torch.cat([torch.arange(target_keypoint_transform.shape[0])[:, None].cuda(), torch.arange(target_keypoint_transform.shape[0])[:, None].cuda()], dim=-1)
    correspondence = correspondence.cpu().numpy()
    correspondence_inlier = correspondence[distance <= 0.1]
    correspondence_outlier = correspondence[distance > 0.1]
    print("correspondence_inlier",correspondence_inlier.shape)
    correspondence_inlier = open3d.utility.Vector2iVector(correspondence_inlier)
    correspondence_outlier = open3d.utility.Vector2iVector(correspondence_outlier)

    source_temp.translate((5, 0, 0), relative=True)
    source_keypoint_temp.translate((5, 0, 0), relative=True)
    # source_temp.transform(transformation)
    inlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp,
                                                                                       target_keypoint,
                                                                                       correspondence_inlier)
    inlier_corr_line.paint_uniform_color([0, 1, 0])
    outlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp,
                                                                                        target_keypoint,
                                                                                        correspondence_outlier)
    outlier_corr_line.paint_uniform_color([1, 0, 0])
    open3d.visualization.draw_geometries([source_temp, target_temp, outlier_corr_line, inlier_corr_line])