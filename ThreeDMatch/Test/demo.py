import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import sys

sys.path.append('../../')
import script.common as cm
from script.tools import get_pcd, get_keypts
from sklearn.neighbors import KDTree
import importlib
import open3d
import copy


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

'''
vicinity=0.3
'''
def build_patch_input(pcd, keypts, vicinity=0.3, num_points_per_patch=2048):
    refer_pts = keypts.astype(np.float32)
    pts = np.array(pcd.points).astype(np.float32)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)   # vicinity 会不会是这个导致不准的？？ 尤其是对于我们训练集的数据！！！！ 0.3和1.0差别
    local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
    for i in range(num_patches):
        local_neighbors = pts[ind_local[i], :]
        if local_neighbors.shape[0] >= num_points_per_patch:
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
        else:
            fix_idx = np.asarray(range(local_neighbors.shape[0]))   # 可能会筛选出重复的点，是否会有影响？
            while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
            random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
                                          replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
            local_neighbors = local_neighbors[choice_idx]
            local_neighbors[-1, :] = refer_pts[i, :]
        local_patches[i] = local_neighbors

    return local_patches


def noise_Gaussian(points, std):
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    out = points + noise
    return out

def noise_Gaussian_Z(points, std):
    noise = np.random.normal(0, std, points[:,2].shape)
    # noise[noise>0.05]=0.05
    # noise[noise < -0.05] = -0.05
    points[:,2] = points[:,2] + noise
    return points

def noise_ramdom(points):
    noise = np.random.rand([points.shape[0],points.shape[1]])*0.1-0.05
    out = points + noise
    return out

def noise_Gaussian_proportion(points, std, proportion):
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    select=np.random.rand(points.shape[0])
    out = points
    out[select<proportion]=points[select<proportion]+noise[select<proportion]
    return out

def unsampling_points(points):
    n = np.random.choice(len(points), num_keypoints, replace=False)
    keypoints= points[n]
    return keypoints, n

def noise_Gaussian_replace(points, std, proportion):
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    select=np.random.rand(points.shape[0])
    out = points
    out[select<proportion]=points[select<proportion]+noise[select<proportion]
    return out

def prepare_patch(pcd):

    pcd_np = np.array(pcd.points).astype(np.float32)
    keypoints, n = unsampling_points(pcd_np)
    local_patches = build_patch_input(pcd, keypoints)  # [num_keypts, 1024, 4]
    return local_patches, keypoints, n


def generate_descriptor(model, pcd):
    model.eval()
    with torch.no_grad():
        local_patches, keypoints, n = prepare_patch(pcd)
        input_ = torch.tensor(local_patches.astype(np.float32))
        B = input_.shape[0]
        input_ = input_.cuda()
        model = model.cuda()
        # calculate descriptors
        desc_list = []
        start_time = time.time()
        desc_len = 64
        step_size = 4
        iter_num = np.int32(np.ceil(B / step_size))
        for k in range(iter_num):
            if k == iter_num - 1:
                desc = model(input_[k * step_size:, :, :])
            else:
                desc = model(input_[k * step_size: (k + 1) * step_size, :, :])
            desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
            del desc
        step_time = time.time() - start_time
        print(f'Finish {B} descriptors spend {step_time:.4f}s')
        desc = np.concatenate(desc_list, 0).reshape([B, desc_len])

    return desc, keypoints, n

def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 512]
    """

    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:
            result.append([i, sourceNNidx[i][0]])
    return np.array(result)

def register2Fragments(keypoints1, keypoints2, descriptor1, descriptor2, gtTrans = None):

    source_keypts = keypoints1
    target_keypts = keypoints2
    source_desc = np.nan_to_num(descriptor1)
    target_desc = np.nan_to_num(descriptor2)
    if source_desc.shape[0] > num_keypoints:
        rand_ind = np.random.choice(source_desc.shape[0], num_keypoints, replace=False)
        source_keypts = source_keypts[rand_ind]
        target_keypts = target_keypts[rand_ind]
        source_desc = source_desc[rand_ind]
        target_desc = target_desc[rand_ind]

    if gtTrans is not None:
        # find mutually cloest point.
        corr = calculate_M(source_desc, target_desc)
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.geometry.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gtTrans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < 0.10)
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1
        print(f"num_inliers:{num_inliers}")
        print(f"inlier_ratio:{inlier_ratio}")

    # calculate the transformation matrix using RANSAC, this is for Registration Recall.
    source_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(source_keypts)
    target_pcd = open3d.geometry.PointCloud()
    target_pcd.points = open3d.utility.Vector3dVector(target_keypts)
    s_desc = open3d.pipelines.registration.Feature()
    s_desc.data = source_desc.T
    t_desc = open3d.pipelines.registration.Feature()
    t_desc.data = target_desc.T

    # registration method: registration_ransac_based_on_feature_matching
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd,
        target_pcd,
        s_desc,
        t_desc,
        mutual_filter=True,
        max_correspondence_distance=0.05,
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),      # 检查两个点云是否用相似的边长构建多边形
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05),
        ],
        criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
    )

    return result.transformation, result.correspondence_set

def read_log_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pairs = len(lines) // 5
    for i in range(num_pairs):
        line_id = i * 5
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragments = int(split_line[2])
        transform = []
        for j in range(1, 5):
            transform.append(lines[line_id + j].split())
        # transform is the pose from test_pair[1] to test_pair[0]
        transform = np.array(transform, dtype=np.float32)
        test_pairs.append(dict(test_pair=test_pair, num_fragments=num_fragments, transform=transform))
    return test_pairs

def draw_registration_result(source, target):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    # source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_corr(source, target,source_keypoint, target_keypoint, transformation, correspondence_set):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_keypoint_transform = copy.deepcopy(source_keypoint)
    source_keypoint_temp = copy.deepcopy(source_keypoint)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])


    source_keypoint_transform.transform(transformation)
    source_transform_np = np.array(source_keypoint_transform.points).astype(np.float32)
    target_np = np.array(target_keypoint.points).astype(np.float32)
    correspondence = np.array(correspondence_set).astype(np.int32)
    distance = np.sqrt(np.sum(np.power(source_transform_np[correspondence[:,0]] - target_np[correspondence[:,1]], 2), axis=1))
    correspondence_inlier = correspondence[distance<=0.1]
    correspondence_outlier = correspondence[distance > 0.1]
    print(correspondence_inlier.shape,correspondence.shape)
    correspondence_inlier = open3d.utility.Vector2iVector(correspondence_inlier)
    correspondence_outlier = open3d.utility.Vector2iVector(correspondence_outlier)

    source_temp.translate((5, 0, 0), relative=True)
    source_keypoint_temp.translate((5, 0, 0), relative=True)
    # source_temp.transform(transformation)
    inlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp, target_keypoint, correspondence_inlier)
    inlier_corr_line.paint_uniform_color([0, 1, 0])
    outlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint_temp, target_keypoint, correspondence_outlier)
    outlier_corr_line.paint_uniform_color([1, 0, 0])
    open3d.visualization.draw_geometries([source_temp, target_temp,outlier_corr_line,inlier_corr_line])


def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)

from typing import Tuple, List, Optional, Union, Any
def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte



if __name__ == '__main__':

    # dynamically load the model
    module_file_path = '../model.py'
    shutil.copy2(os.path.join('.', '../../module/SphereNet.py'), module_file_path)
    module_name = ''
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.SphereNet(0.3, 15, 40, 20, '3DMatch')
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load('../../pretrain/3DMatch_best.pkl'))
    num_keypoints = int(sys.argv[1])

    id1 = 0
    id2 = 2
    desc_name = 'SpinNet'
    scene = '7-scenes-redkitchen'
    pcdpath1 = f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/{scene}/cloud_bin_{id1}.ply"
    pcdpath2 = f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/{scene}/cloud_bin_{id2}.ply"
    gtpath = f'/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/{scene}-evaluation/'
    # gtpath = f'/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/3DLoMatch/{scene}/'
    descpath = f"/home/zhaoguiyu/code/SphereNet/ThreeDMatch/Test/{desc_name}_desc_{12281301}/{scene}"
    interpath = f"../../data/3DMatch/intermediate-files-real/{scene}/"
    keyptspath = interpath
    # id1 = 10
    # id2 = 11
    # pcdpath1 = f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika/cloud_bin_{id1}.ply"
    # pcdpath2 = f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika/cloud_bin_{id2}.ply"
    # gtpath = f'/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation/'
    gt_logs = read_log_file(os.path.join(gtpath, f"gt.log"))
    for i in gt_logs:
        if i['test_pair']==[id1,id2]:
            gt_transformation=i['transform']
            break
    # pcdpath1 = sys.argv[2]
    # pcdpath2 = sys.argv[3]
    transformation = np.linalg.inv(gt_transformation)
    pcd1 = open3d.io.read_point_cloud(pcdpath1)
    pcd2 = open3d.io.read_point_cloud(pcdpath2)

    src_pcd_np1 = np.array(pcd1.points).astype(np.float32)
    tgt_pcd_np1 = np.array(pcd2.points).astype(np.float32)
    src_pcd_np = noise_Gaussian_Z(src_pcd_np1,0.05)
    tgt_pcd_np = noise_Gaussian_Z(tgt_pcd_np1, 0.05)
    src_pcd = open3d.geometry.PointCloud()
    src_pcd.points = open3d.utility.Vector3dVector(src_pcd_np)
    tgt_pcd = open3d.geometry.PointCloud()
    tgt_pcd.points = open3d.utility.Vector3dVector(tgt_pcd_np)

    #
    descriptor1, keypoints1, n = generate_descriptor(model, pcd1)
    descriptor2, keypoints2, m = generate_descriptor(model, pcd2)
    # keypoints1=src_pcd_np1[n]
    # keypoints2=tgt_pcd_np1[m]
    corr = calculate_M(descriptor1, descriptor2)
    start_time = time.time()
    transformation, correspondence_set = register2Fragments(keypoints1, keypoints2, descriptor1, descriptor2)
    print(f"Finish in {time.time() - start_time}s")
    source_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(keypoints1)
    target_pcd = open3d.geometry.PointCloud()
    target_pcd.points = open3d.utility.Vector3dVector(keypoints2)
    draw_registration_corr(pcd1, pcd2, source_pcd, target_pcd, gt_transformation, corr)
    draw_registration_corr(pcd1, pcd2, source_pcd, target_pcd, gt_transformation, correspondence_set)

    draw_registration_result(pcd1, pcd2)
