import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import sys

sys.path.append('../')
from sklearn.neighbors import KDTree
import importlib
import open3d
import copy
import time
from module.HCF_module import HCF_module
from script.registration_metric import calculate_IR2, compute_registration_error
from script.tools import read_pose_file
from script.visualization import draw_registration_corr2, draw_registration_result


def build_patch_input(pcd, keypts, vicinity=0.3, num_points_per_patch=2048):
    refer_pts = keypts.astype(np.float32)
    pts = np.array(pcd.points).astype(np.float32)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)  # vicinity 会不会是这个导致不准的？？ 尤其是对于我们训练集的数据！！！！ 0.3和1.0差别
    local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
    for i in range(num_patches):
        local_neighbors = pts[ind_local[i], :]
        if local_neighbors.shape[0] >= num_points_per_patch:
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
        else:
            fix_idx = np.asarray(range(local_neighbors.shape[0]))  # 可能会筛选出重复的点，是否会有影响？
            while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
            random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
                                          replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
            local_neighbors = local_neighbors[choice_idx]
            local_neighbors[-1, :] = refer_pts[i, :]
        local_patches[i] = local_neighbors

    return local_patches


def unsampling_points(points):
    num_keypoints1 = num_keypoints
    if num_keypoints1 > len(points):
        num_keypoints1 = len(points)
    n = np.random.choice(len(points), num_keypoints1, replace=False)
    keypoints = points[n]
    return keypoints, n


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
            desc_list.append(desc.view(desc.shape[0], desc_len).detach())
            del desc
        step_time = time.time() - start_time
        # print(f'Finish {B} descriptors spend {step_time:.4f}s')
        desc = torch.cat(desc_list, 0).view([B, desc_len])

    return desc, keypoints, n


if __name__ == '__main__':
    visualization_bool =True
    # dynamically load the model
    module_file_path = '../ThreeDMatch/model.py'
    shutil.copy2(os.path.join('.', '../module/SphereNet.py'), module_file_path)
    module_name = ''
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.SphereNet(0.3, 15, 40, 20, '3DMatch')
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load('../pretrain/3DMatch_best.pkl'))
    num_keypoints = int(sys.argv[1])
    HCF = HCF_module(inlier_threshold=0.1,
                     num_node="all",
                     use_mutual=False,
                     d_thre=0.1,
                     num_iterations=10,
                     ratio=0.2,
                     nms_radius=0.1,
                     max_points=8000,
                     k=200,
                     filter_ratio=0.5,
                     final_k=15, )

    # 3DCSR dataset directory. Please replace it with your path
    directory = "/media/zhaoguiyu/新加卷/cross-source-dataset/cross-source-dataset/kinect_sfm"
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    RR_sum = 0
    PIR_sum = 0
    IR_sum = 0
    PFMR_sum = 0
    FMR_sum = 0
    successful_sum = 0
    RTE_sum = 0
    RRE_sum = 0
    num = 0
    time_feat = 0
    time_pose = 0
    scene_result_list = []
    # These point clouds has a large amount of data, which leads to a large number of gpu memory occupation.
    wrong_list = ["hard/labcorner", "hard/lab6", "hard/lab4", "easy/duck", "easy/duckpart"]
    for folder in folders:

        RR_sum_one_scene = 0
        successful_sum_one_scene = 0
        PIR_sum_one_scene = 0
        PFMR_sum_one_scene = 0
        FMR_sum_one_scene = 0
        IR_sum_one_scene = 0
        RTE_sum_one_scene = 0
        RRE_sum_one_scene = 0
        num_one_scene = 0
        scenes = [scene for scene in os.listdir(f"{directory}/{folder}") if
                  os.path.isdir(os.path.join(f"{directory}/{folder}", scene))]
        for scene in scenes:
            if f"{folder}/{scene}" in wrong_list:
                continue
            pcdpath1 = f"{directory}/{folder}/{scene}/kinect.ply"
            pcdpath2 = f"{directory}/{folder}/{scene}/sfm.ply"
            gtpath = f"{directory}/{folder}/{scene}/T_gt.txt"

            gt_transformation = read_pose_file(gtpath)
            gt_transformation = np.asarray(gt_transformation)
            gt_transformation = np.linalg.inv(gt_transformation)
            pcd1 = open3d.io.read_point_cloud(pcdpath1)
            pcd2 = open3d.io.read_point_cloud(pcdpath2)
            pcd1 = pcd1.voxel_down_sample(0.025)
            pcd2 = pcd2.voxel_down_sample(0.025)

            # draw_registration_result(pcd1, pcd2, gt_transformation)
            T1 = time.time()
            descriptor1, keypoints1, n = generate_descriptor(model, pcd1)
            descriptor2, keypoints2, m = generate_descriptor(model, pcd2)
            T2 = time.time()
            # keypoints1=src_pcd_np1[n]
            # keypoints2=tgt_pcd_np1[m]
            keypoints1 = torch.from_numpy(keypoints1).cuda()
            keypoints2 = torch.from_numpy(keypoints2).cuda()

            transformation, _, src_node_corr, ref_node_corr = HCF(keypoints1[None], keypoints2[None], descriptor1[None], descriptor2[None])
            transformation = transformation[0].cpu().numpy()
            src_node_corr = src_node_corr[0].cpu().numpy()
            ref_node_corr = ref_node_corr[0].cpu().numpy()
            IR, inliers_num = calculate_IR2(src_node_corr, ref_node_corr, gt_transformation)

            # import open3d
            # if visualization_bool:
            #     print("对应关系")
            #     src = open3d.geometry.PointCloud()
            #     src.points = open3d.utility.Vector3dVector(src_node_corr)
            #     tgt = open3d.geometry.PointCloud()
            #     tgt.points = open3d.utility.Vector3dVector(ref_node_corr)
            #     src1 = open3d.geometry.PointCloud()
            #     src1.points = open3d.utility.Vector3dVector(src_node_corr)
            #     tgt1 = open3d.geometry.PointCloud()
            #     tgt1.points = open3d.utility.Vector3dVector(ref_node_corr)
            #     draw_registration_corr2(pcd1, pcd2, src1, tgt1, np.linalg.inv(gt_transformation))
            #     print("最后的配准结果")
            #     draw_registration_result(pcd1, pcd2, transformation)
            #     draw_registration_result(pcd1, pcd2, gt_transformation)

            T3 = time.time()
            time_feat = time_feat + T2 - T1
            time_pose = time_pose + T3 - T2
            rre, rte = compute_registration_error(gt_transformation, transformation)
            num = num + 1
            if IR > 0.05:
                FMR = 1
            else:
                FMR = 0

            if rre > 15 or rte > 0.3:
                RR = 0
            else:
                RR = 1
            if RR == 1:
                RTE_sum = RTE_sum + rte
                RRE_sum = RRE_sum + rre
                successful_sum = successful_sum + 1

            RR_sum = RR_sum + RR
            IR_sum = IR_sum + IR
            FMR_sum = FMR_sum + FMR
            print(f"IR: {IR:.3f}, FMR: {FMR}， inlier_num: {inliers_num}, "
                  f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}, RR : {RR}")

            num_one_scene = num_one_scene + 1
            if RR == 1:
                RTE_sum_one_scene = RTE_sum_one_scene + rte
                RRE_sum_one_scene = RRE_sum_one_scene + rre
                successful_sum_one_scene = successful_sum + 1
            else:
                print(f"{folder}/{scene}")
            RR_sum_one_scene = RR_sum_one_scene + RR
            IR_sum_one_scene = IR_sum_one_scene + IR
            FMR_sum_one_scene = FMR_sum_one_scene + FMR

        print(
            f"scene {folder}, mean PIR : {PIR_sum_one_scene / num_one_scene}, mean PFMR : {PFMR_sum_one_scene / num_one_scene}, "
            f"mean IR : {IR_sum_one_scene / num_one_scene}， mean FMR : {FMR_sum_one_scene / num_one_scene},"
            f"mean RR : {RR_sum_one_scene / num_one_scene}")

        if successful_sum_one_scene != 0:
            print(
                f"scene {folder}, mean rre and rte : {RRE_sum_one_scene / successful_sum_one_scene} deg, {RTE_sum_one_scene / successful_sum_one_scene}")
            scene_result_list.append(
                f"scene {folder}, mean PIR : {PIR_sum_one_scene / num_one_scene}, mean PFMR : {PFMR_sum_one_scene / num_one_scene}, "
                f"mean IR : {IR_sum_one_scene / num_one_scene}， mean FMR : {FMR_sum_one_scene / num_one_scene},"
                f"mean RR : {RR_sum_one_scene / num_one_scene},"
                f"scene {folder}, mean rre and rte : {RRE_sum_one_scene / successful_sum_one_scene} deg, {RTE_sum_one_scene / successful_sum_one_scene}")
        else:
            print(f"scene {folder}, mean rre and rte : Nan deg, Nan")
            scene_result_list.append(
                f"scene {folder}, mean PIR : {PIR_sum_one_scene / num_one_scene}, mean PFMR : {PFMR_sum_one_scene / num_one_scene}, "
                f"mean IR : {IR_sum_one_scene / num_one_scene}， mean FMR : {FMR_sum_one_scene / num_one_scene},"
                f"mean RR : {RR_sum_one_scene / num_one_scene},"
                f"scene {folder}, mean rre and rte : Nan deg, Nan")
    for i in scene_result_list:
        print(i)
    print(f"All scenes, mean PIR : {PIR_sum / num}, mean PFMR : {PFMR_sum / num}, "
          f"mean IR : {IR_sum / num}， mean FMR : {FMR_sum / num}, "
          f"mean RR : {RR_sum / num}")
    print(f"All scenes, mean rre and rte : {RRE_sum / successful_sum} deg, {RTE_sum / successful_sum}")
    print(f"All scenes, mean time feat and pose : {time_feat / num}, {time_pose / num}")
