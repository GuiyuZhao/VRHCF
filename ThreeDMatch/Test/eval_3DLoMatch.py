import sys
sys.path.append('../../')
import numpy as np
import time
import os
from script.tools import get_keypts, get_desc, loadlog, read_trajectory_info
from module.HCF_module import HCF_module
import torch
from script.registration_metric import calculate_IR2, compute_registration_error, computeTransformationErr


def Registration(register_method, id1, id2, keyptspath, descpath, desc_name, index):
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    source_keypts = get_keypts(keyptspath, cloud_bin_s)
    target_keypts = get_keypts(keyptspath, cloud_bin_t)

    source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
    target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
    source_desc = np.nan_to_num(source_desc)
    target_desc = np.nan_to_num(target_desc)
    source_desc = source_desc / (np.linalg.norm(source_desc, axis=1, keepdims=True) + 1e-6)
    target_desc = target_desc / (np.linalg.norm(target_desc, axis=1, keepdims=True) + 1e-6)

    if source_desc.shape[0] > num_keypoints:
        rand_ind = np.random.choice(source_desc.shape[0], num_keypoints, replace=False)
        source_keypts = source_keypts[rand_ind]
        target_keypts = target_keypts[rand_ind]
        source_desc = source_desc[rand_ind]
        target_desc = target_desc[rand_ind]

    source_keypts = torch.from_numpy(source_keypts).cuda()
    target_keypts = torch.from_numpy(target_keypts).cuda()
    source_desc = torch.from_numpy(source_desc).cuda()
    target_desc = torch.from_numpy(target_desc).cuda()

    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    if key not in gtLog.keys():
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
        rre = 0
        rte = 0
        rmse = 0
    else:
        # find mutually cloest point.
        gt_transformation = gtLog[key]
        gt_flag = 1
        gt_transformation = np.linalg.inv(gt_transformation)
        gt_transformation = torch.from_numpy(gt_transformation).cuda()
        transformation, _, src_node_corr, ref_node_corr = register_method(source_keypts[None], target_keypts[None],
                                                                          source_desc[None], target_desc[None])
        transformation = transformation[0].cpu().numpy()
        src_node_corr = src_node_corr[0].cpu().numpy()
        ref_node_corr = ref_node_corr[0].cpu().numpy()
        gt_transformation = gt_transformation.cpu().numpy()
        inlier_ratio, num_inliers = calculate_IR2(src_node_corr, ref_node_corr, gt_transformation)
        rre, rte = compute_registration_error(gt_transformation, transformation)
        rmse = computeTransformationErr(np.linalg.inv(gt_transformation) @ transformation, gt_traj_cov[index, :, :])

    return num_inliers, inlier_ratio, gt_flag, rre, rte, rmse


if __name__ == '__main__':

    RR_sum = 0
    IR_sum = 0
    FMR_sum = 0
    successful_sum = 0
    RTE_sum = 0
    RRE_sum = 0
    num = 0
    scene_result_list = []

    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    desc_name = 'SpinNet'
    timestr = sys.argv[1]
    num_keypoints = int(sys.argv[2])
    IR_list = []
    RR_list = []
    FMR_list = []

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

    for scene in scene_list:

        RR_sum_one_scene = 0
        successful_sum_one_scene = 0
        FMR_sum_one_scene = 0
        IR_sum_one_scene = 0
        RTE_sum_one_scene = 0
        RRE_sum_one_scene = 0
        num_one_scene = 0

        # 3DCSR dataset directory. Please replace it with your path
        overlappath=f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/3DLoMatch/{scene}/"
        pcdpath = f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/{scene}/"
        interpath = f"/home/zhaoguiyu/code/SphereNet/data/3DMatch/intermediate-files-real/{scene}/"
        gtpath = f'/home/zhaoguiyu/code/SphereNet/data/3DMatch/fragments/3DLoMatch/{scene}/'
        keyptspath = interpath  # os.path.join(interpath, "keypoints/")
        descpath = os.path.join(".", f"{desc_name}_desc_{timestr}/{scene}")
        gtLog = loadlog(gtpath)
        _, gt_traj_cov = read_trajectory_info(os.path.join(gtpath, "gt.info"))

        # register each pair
        num_frag = len(os.listdir(pcdpath))
        print(f"Start Evaluate Descriptor {desc_name} for {scene}")
        start_time = time.time()
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                index = num_one_scene
                num_inliers, IR, gt_flag, rre, rte, rmse = Registration(HCF, id1, id2, keyptspath, descpath,
                                                                              desc_name, index)
                if gt_flag == 1:
                    num = num + 1
                    if IR > 0.05:
                        FMR = 1
                    else:
                        FMR = 0
                    if rmse > 0.04:
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

                    num_one_scene = num_one_scene + 1
                    if RR == 1:
                        RTE_sum_one_scene = RTE_sum_one_scene + rte
                        RRE_sum_one_scene = RRE_sum_one_scene + rre
                        successful_sum_one_scene = successful_sum + 1

                    RR_sum_one_scene = RR_sum_one_scene + RR
                    IR_sum_one_scene = IR_sum_one_scene + IR
                    FMR_sum_one_scene = FMR_sum_one_scene + FMR
                    print(f"IR: {IR:.3f}, FMR: {FMR}， inlier_num: {num_inliers}, "
                          f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}, RR : {RR}")

        print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")
        print(
            f"scene {scene},"
            f"mean IR : {IR_sum_one_scene / num_one_scene}， mean FMR : {FMR_sum_one_scene / num_one_scene},"
            f"mean RR : {RR_sum_one_scene / num_one_scene}")

        if successful_sum_one_scene != 0:
            IR_list.append(IR_sum_one_scene / num_one_scene)
            FMR_list.append(FMR_sum_one_scene / num_one_scene)
            RR_list.append(RR_sum_one_scene / num_one_scene)
            print(
                f"scene {scene}, "
                f"mean rre and rte : {RRE_sum_one_scene / successful_sum_one_scene} deg, {RTE_sum_one_scene / successful_sum_one_scene}")
            scene_result_list.append(
                f"scene {scene},"
                f"mean IR : {IR_sum_one_scene / num_one_scene}， mean FMR : {FMR_sum_one_scene / num_one_scene},"
                f"mean RR : {RR_sum_one_scene / num_one_scene},"
                f"scene {scene}, mean rre and rte : {RRE_sum_one_scene / successful_sum_one_scene} deg, {RTE_sum_one_scene / successful_sum_one_scene}")
        else:
            print(f"scene {scene}, mean rre and rte : Nan deg, Nan")
            scene_result_list.append(
                f"scene {scene},"
                f"mean IR : {IR_sum_one_scene / num_one_scene}， mean FMR : {FMR_sum_one_scene / num_one_scene},"
                f"mean RR : {RR_sum_one_scene / num_one_scene},"
                f"scene {scene}, mean rre and rte : Nan deg, Nan")

    for i in scene_result_list:
        print(i)
    print(f"All scenes, "
          f"mean IR : {IR_sum / num}， mean FMR : {FMR_sum / num}, "
          f"mean RR : {RR_sum / num}")
    print(f"All scenes, mean rre and rte : {RRE_sum / successful_sum} deg, {RTE_sum / successful_sum}")

