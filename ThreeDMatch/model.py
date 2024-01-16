import sys
import time

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import module.SphericalCNN as NET
import module.SphericalVoxelization as VS
import script.common as cm
from script.common import switch


class SphereNet(nn.Module):
    def __init__(self, des_r, rad_n, azi_n, ele_n, dataset, use_interpolation=True, use_MSF=False):
        super(SphereNet, self).__init__()
        self.des_r = des_r
        self.rad_n = rad_n
        self.azi_n = azi_n
        self.ele_n = ele_n
        self.dataset = dataset
        self.use_interpolation = use_interpolation
        self.use_MSF = use_MSF
        if use_MSF:
            self.conv_net = NET.SCNN_MSF(inchan=3, dim=64)
        else:
            self.conv_net = NET.SCNN(inchan=1, dim=64)
        self.Spherical_Voxelization = VS.Spherical_Voxelization(des_r, rad_n, azi_n, ele_n, use_interpolation, use_MSF)
        self.bn_xyz_raising = nn.BatchNorm2d(16)
        self.bn_mapping = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        self.xyz_raising = nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, input):

        #################################
        # 1. Spheroid Generator
        #################################
        superpoints = input[:, -1, :].unsqueeze(1)  # get the superpoints
        points = input[:, :, 0:3] - superpoints[:, :, 0:3]  # realize translation invariance
        for case in switch(self.dataset):
            if case('3DMatch'):
                z_axis = cm.Construct_LRF(points, ref_point=input[:, -1, :3])
                z_axis = cm.l2_norm(z_axis, axis=1)
                R = cm.RodsRotatFormula(z_axis,
                                        torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(z_axis.shape[0], 1))  # 根旋转公式
                points_LRF = torch.matmul(points, R)
                points_LRF = cm.Determine_direction_Z(points_LRF)
                break
            if case('KITTI'):
                points_LRF = points
                break

        x = self.Spherical_Voxelization(points_LRF)
        # points_LSRF = model.xyz2spherical(points_LRF)  # to local spherical reference frame (LSRF)
        # SHOT_feature = model.get_SHOT_space_inter(points_LSRF, self.des_r, self.rad_n, self.azi_n, self.ele_n)
        # x = SHOT_feature.view(SHOT_feature.shape[0], 1, self.rad_n, self.azi_n, self.ele_n)

        del points_LRF
        del points

        #################################
        # 2. Spherical Feature Extractor
        #################################
        x = self.conv_net(x)
        x = F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))

        return x

    def get_parameter(self):
        return list(self.parameters())
