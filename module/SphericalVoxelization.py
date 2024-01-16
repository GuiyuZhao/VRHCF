import torch
import math
import torch.nn as nn
import script.common as common


class Spherical_Voxelization(nn.Module):
    """

    """
    def __init__(self, des_r, rad_n, azi_n, ele_n, use_interpolation=True, use_MSF=False):
        super(Spherical_Voxelization, self).__init__()
        self.des_r = des_r
        self.rad_n = rad_n
        self.azi_n = azi_n
        self.ele_n = ele_n
        self.use_interpolation = use_interpolation
        self.use_MSF = use_MSF

    def forward(self, points):

        points_SCS = common.xyz2spherical(points)  # to Spherical coordinate system

        if self.use_interpolation is True:   # use interpolation or not
            initial_feature = get_SHOT_space_inter(points_SCS, self.des_r, self.rad_n, self.azi_n, self.ele_n)
        else:
            initial_feature = get_SHOT_space(points_SCS, self.des_r, self.rad_n, self.azi_n, self.ele_n)

        x = initial_feature.view(initial_feature.shape[0], 1, self.rad_n, self.azi_n, self.ele_n)

        if self.use_MSF is True:  # use multiscale feature fusion or not
            x = Multiscale_SHOT(x)

        del initial_feature
        del points

        return x


def get_SHOT_space_inter(pts, radius, rad_n, azi_n, ele_n):
    """
    :param pts: all points, [B. N. 3]
    :return:
    pts_SHOT_space: [B. N. 1. rad_n, azi_n, ele_n]
    """
    device = pts.device
    B, N, C = pts.shape
    pts_SHOT_space = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float).to(device).repeat(
        [B, N, rad_n, azi_n, ele_n])  # (B, rad_n, azi_n, ele_n)

    # Step size of r, a, and e to generate interpolation line
    rad_n_start=torch.tensor(radius/rad_n/2).to(device)
    rad_n_step=torch.tensor(radius/rad_n).to(device)
    azi_n_start=torch.tensor(2*math.pi/azi_n/2).to(device)
    azi_n_step=torch.tensor(2*math.pi/azi_n).to(device)
    ele_n_start=torch.tensor(math.pi/ele_n/2).to(device)
    ele_n_step=torch.tensor(math.pi/ele_n).to(device)

    SHOT_space_azi = torch.arange(azi_n_start, 2*math.pi, azi_n_step, dtype=torch.float).to(device).view(1, 1, azi_n).repeat([B, N, 1])
    SHOT_space_rad = torch.arange(rad_n_start,radius, rad_n_step, dtype=torch.float).to(device).view(1, 1, rad_n).repeat([B, N, 1])
    SHOT_space_ele = torch.arange(ele_n_start, math.pi, ele_n_step, dtype=torch.float).to(device).view(1, 1, ele_n).repeat([B, N, 1])

    # Calculate the distance to the interpolation line
    r_distance=torch.abs(pts[:,:,0].view(B,N,1).expand(B,N,rad_n)-SHOT_space_rad)
    a_distance=torch.abs(pts[:,:,1].view(B,N,1).expand(B,N,azi_n)-SHOT_space_azi)
    e_distance =torch.abs(pts[:, :, 2].view(B,N,1).expand(B,N,ele_n) - SHOT_space_ele)

    # The radius interpolation is adjusted for special cases, noting that pts uses broadcast here
    r_distance[(pts[:,:,0]<rad_n_start).view(B,N,1).expand(B,N,rad_n)*(r_distance <=rad_n_start)]=0
    r_distance[(pts[:,:,0]<rad_n_start).view(B,N,1).expand(B,N,rad_n)*(r_distance>rad_n_step)*(r_distance <=rad_n_step+rad_n_start)] = rad_n_step
    # r_distance[r_distance >radius-radius/rad_n/2]=0
    r_distance[((pts[:,:,0]>radius-rad_n_start)*(pts[:,:,0]<=radius)).view(B,N,1).expand(B,N,rad_n)*(r_distance>rad_n_step)*(r_distance <=rad_n_step+rad_n_start)] = rad_n_step
    # Longitude interpolation special case adjustment
    a_distance[(pts[:,:,1]<azi_n_start).view(B,N,1).expand(B,N,azi_n)*(2*math.pi-azi_n_step<a_distance)] = azi_n_step-a_distance[(pts[:,:,1]<azi_n_start).view(B,N,1).expand(B,N,azi_n)*(a_distance<azi_n_step*0.75)]
    # The 0.75 is for precision purposes, but it's actually 0.5, which is where the starting point is
    a_distance[((pts[:,:,1]>2*math.pi-azi_n_start)*(pts[:,:,1]<=2*math.pi)).view(B,N,1).expand(B,N,azi_n)*(2*math.pi-azi_n_step<a_distance)] = azi_n_step-a_distance[(pts[:,:,1]>2*math.pi-azi_n_start*(pts[:,:,1]<=2*math.pi)).view(B,N,1).expand(B,N,azi_n)*(a_distance<azi_n_step*0.75)]

    # Latitude interpolation special case adjustment
    e_distance[(pts[:,:,2]<ele_n_start).view(B,N,1).expand(B,N,ele_n)*(e_distance <=ele_n_start)]=0
    e_distance[(pts[:,:,2]<ele_n_start).view(B,N,1).expand(B,N,ele_n)*(ele_n_step<e_distance)*(e_distance<=ele_n_step+ele_n_start)] = ele_n_step
    e_distance[((pts[:,:,2]>math.pi-ele_n_start)*(pts[:,:,2]<=math.pi)).view(B,N,1).expand(B,N,ele_n)*(e_distance <=ele_n_step*0.75)]=0
    e_distance[((pts[:,:,2]>math.pi-ele_n_start)*(pts[:,:,2]<=math.pi)).view(B,N,1).expand(B,N,ele_n)*(ele_n_step<e_distance)*(e_distance<=ele_n_start+ele_n_step)]=ele_n_step

    # Sort finds two interpolation points for each dimension
    r_distance,r_idx=torch.sort(r_distance,2,descending=False)
    a_distance,a_idx=torch.sort(a_distance,2,descending=False)
    e_distance,e_idx=torch.sort(e_distance,2,descending=False)
    r_distance=r_distance[:,:,0:2]
    r_idx=r_idx[:,:,0:2]
    a_distance=a_distance[:,:,0:2]
    a_idx=a_idx[:,:,0:2]
    e_distance=e_distance[:,:,0:2]
    e_idx=e_idx[:,:,0:2]

    # Calculate the total weight of interpolation in each dimension
    r_distance=1-r_distance/(radius/rad_n)
    a_distance = 1 - a_distance / (2 * math.pi / azi_n)
    e_distance = 1 - e_distance / (math.pi / ele_n)

    # Set 0 if out of range
    r_distance[pts[:, :, 0] > radius] = 0
    a_distance[pts[:, :, 1] > 2 * math.pi] = 0
    e_distance[pts[:, :, 2] > math.pi] = 0

    r_idx=r_idx.view(B,N,2,1).repeat(1,1,1,4).view(B,N,8)
    a_idx=a_idx.view(B,N,2,1).repeat(1,1,1,2).view(B,N,4).repeat(1,1,2)
    e_idx = e_idx.repeat(1,1,4)
    idx=torch.stack((r_idx,a_idx,e_idx),dim=3).view(-1,3)
    B_idx=torch.arange(B).to(device).view(B,1).repeat(1,8*N).view(8*N*B,1)
    N_idx=torch.arange(N).to(device).view(N,1).repeat(1,8).view(8*N).repeat(B).view(8*N*B,1)
    idx = torch.cat((B_idx,N_idx,idx), dim=1)

    r_distance=r_distance.view(B,N,2,1).repeat(1,1,1,4).view(B,N,8)
    a_distance=a_distance.view(B,N,2,1).repeat(1,1,1,2).view(B,N,4).repeat(1,1,2)
    e_distance = e_distance.repeat(1,1,4)
    weight=(r_distance*a_distance*e_distance).view(B*N*8)

    # Interpolation weight summation
    pts_SHOT_space[idx[:,0],idx[:,1],idx[:,2],idx[:,3],idx[:,4]]=weight
    pts_SHOT_space = torch.sum(pts_SHOT_space, dim=1)

    del SHOT_space_azi
    del SHOT_space_rad
    del SHOT_space_ele
    del r_distance
    del a_distance
    del e_distance
    del r_idx
    del a_idx
    del e_idx
    del B_idx
    del N_idx
    del weight

    return pts_SHOT_space

def Multiscale_SHOT (feature):

    device = feature.device
    B, C, N, M, K = feature.shape
    feature1 = feature[:, :, 0:5, :, :]
    feature2 = feature[:, :, 0:10:2, :, :] + feature[:, :, 1:10:2, :, :]
    feature3 = feature[:, :, 0:15:3, :, :] + feature[:, :, 1:15:3, :, :] + feature[:, :, 2:15:3, :, :]
    mutiscale_feature=torch.cat((feature1, feature2,feature3), dim=1)

    return mutiscale_feature


def get_SHOT_space(pts, radius, rad_n, azi_n, ele_n):
    """
    :param pts: all points, [B. N. 3]
    :return:
    pts_SHOT_space: [B. N. 1. rad_n, azi_n, ele_n]
    """
    device = pts.device
    B, N, C = pts.shape
    pts_SHOT_space = torch.zeros(1, 1, 1, 1, 1, dtype=torch.long).to(device).repeat(
        [B, N, rad_n, azi_n, ele_n])

    SHOT_space_azi = torch.arange(azi_n, dtype=torch.int).to(device).view(1, azi_n, 1).repeat([rad_n, 1, ele_n])
    SHOT_space_rad = torch.arange(rad_n, dtype=torch.int).to(device).view(rad_n, 1, 1).repeat([1, azi_n, ele_n])
    SHOT_space_ele = torch.arange(ele_n, dtype=torch.int).to(device).view(1, 1, ele_n).repeat([rad_n, azi_n, 1])

    # [B. N.  rad_n, azi_n, ele_n] Which area is the statistical point located in
    pts_position = (torch.div(pts[:, :, 0], (radius / rad_n)).floor()) * 1 + (
        torch.div(pts[:, :, 1], (2 * math.pi / azi_n)).floor()) * 100 + (
                       torch.div(pts[:, :, 2], (math.pi / ele_n)).floor()) * 10000
    pts_position = pts_position.view(B, N, 1, 1, 1).repeat([1, 1, rad_n, azi_n, ele_n])
    SHOT_space_position = SHOT_space_rad * 1 + SHOT_space_azi * 100 + SHOT_space_ele * 10000  # [B. N.  rad_n, azi_n, ele_n]
    SHOT_space_position = SHOT_space_position.to(device).view(1, 1, rad_n, azi_n, ele_n).repeat([B, N, 1, 1, 1])

    pts_SHOT_space[SHOT_space_position == pts_position] = 1

    # Find the corresponding area, add up to get points
    pts_SHOT_space = torch.sum(pts_SHOT_space, dim=1)

    del pts_position
    del SHOT_space_position
    del SHOT_space_rad
    del SHOT_space_azi
    del SHOT_space_ele

    return pts_SHOT_space