
import torch
# import numpy as np
# import torch.nn as nn
# from torch.autograd import Function
# import math
import sys
sys.path.append("/workspace/PointNeXt/")
# import os, sys
from typing import Tuple
from torch.autograd import Function
from openpoints.cpp import pointnet2_cuda
import pdb
# import open3d as o3d

# import logging
import pdb
# import csv


def find_indices(a, b, batch):
    try:
        c = []
        for i in range(0, batch):
            c_matrix = []
            for b_row in b[i,:]:
                # 找到与b_row相等的行的索引
                a_row=a[i,:]
                idx = (a_row == b_row).all(dim=1).nonzero(as_tuple=True)[0]
                if idx.size()[0] == 1:
                    c_matrix.append(idx.item())
                else:
                    c_matrix.append(idx[0].item())
            c.append(c_matrix)
    except ValueError:
        print(ValueError)
        pdb.set_trace()
    
    return torch.tensor(c)

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        # pdb.set_trace()
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


# ################ block partition with different directions and count the size
def part2_and_count(xyz, FPS_th, TreeDepth):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    direction = TreeDepth % 3
    # xyz.size(), torch.Size([1, 1024, 3])
    xyz_onedirec = xyz[:, :, direction]
    xyz_not5 = xyz_onedirec[xyz_onedirec != 5.0]
    max_val = torch.max(xyz_not5)
    min_val = torch.min(xyz_not5)
    mid_val = (max_val + min_val)/2

    # double check the direction, incase the mid_val is equal to max or min
    if mid_val == max_val:
        xyz_onedirec = xyz[:, :, ((TreeDepth+1) % 3)]
        xyz_not5 = xyz_onedirec[xyz_onedirec != 5.0]
        max_val = torch.max(xyz_not5)
        min_val = torch.min(xyz_not5)
        mid_val = (max_val + min_val)/2
        assert max_val != min_val

    xyz_lessMid_indx = torch.nonzero((xyz_onedirec < mid_val))
    xyz_largMid_indx = torch.nonzero((xyz_onedirec >= mid_val) & (xyz_onedirec != 5))
    
    size_2dTensor  = [0 for _ in range(2)]
    xyz_2dList = [0 for _ in range(2)]
    FPS_2dList = [0 for _ in range(2)]

    size_2dTensor[0] = xyz_lessMid_indx.size()[0]
    size_2dTensor[1] = xyz_largMid_indx.size()[0]
    FPS_2dList[0] = 1 if size_2dTensor[0] <= FPS_th else 0
    FPS_2dList[1] = 1 if size_2dTensor[1] <= FPS_th else 0
    # pdb.set_trace()
    xyz_lessMid = torch.full_like(xyz, 5)
    xyz_lessMid[:, xyz_lessMid_indx[:,1], :] = xyz[:, xyz_lessMid_indx[:,1], :]
    xyz_largMid = torch.full_like(xyz, 5)
    xyz_largMid[:, xyz_largMid_indx[:,1], :] = xyz[:, xyz_largMid_indx[:,1], :]
    xyz_2dList[0] = xyz_lessMid
    xyz_2dList[1] = xyz_largMid

    return size_2dTensor, xyz_2dList, FPS_2dList

def selectAfromB(A, B):
    A_2d = A.squeeze(0)
    B_2d = B.squeeze(0)
    matches = (B_2d[:, None] == A_2d).all(-1).any(1)
    C = B_2d[matches]
    count = C.shape[0]
    return C.unsqueeze(0), count


def mygroup(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    :param ctx:
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
    pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
    # pdb.set_trace()
    return idx


def Tree_group_depth10_config(xyz, nsample, new_xyz, radius):
    B, N, _ = xyz.size()

    # tree FPS
    # firstly tree block partition

    indx_all = torch.tensor([], device='cuda')
    try:
        for i in range(0, B): # deep=1
            indx_batch = torch.tensor([], device='cuda')
            indices_batch = torch.tensor([], device='cuda')
            size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count(xyz[i, :, :].unsqueeze(0),FPS_th, 0)
            for j in range(0, 2):
                if (size_2dL_L1[j] != 0):
                    xyz_L1 = xyz_2dL_L1[j]
                    if (FPS_2dL_L1[j]==1):
                        centerXYZ, centerXYZ_size = selectAfromB(xyz_L1, new_xyz[i,:,:].unsqueeze(0))
                        if(centerXYZ_size != 0):
                            group_from_xyz = xyz_L1
                            group_from_xyz_N = xyz_L1.size(1)
                            idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                            pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                            indx_batch = torch.cat((indx_batch, idx), dim=1)

                        else:
                            continue
                    else: # deep = 2
                        size_2dL_L2, xyz_2dL_L2, FPS_2dL_L2 = part2_and_count(xyz_L1, FPS_th, 1)
                        for k in range(0, 2):
                            if (size_2dL_L2[k] != 0):
                                xyz_L2 = xyz_2dL_L2[k]
                                # if (FPS_2dL_L2[k]==1):
                                centerXYZ, centerXYZ_size = selectAfromB(xyz_L2, new_xyz[i,:,:].unsqueeze(0))
                                if(centerXYZ_size != 0):
                                    group_from_xyz = xyz_L1 # group source
                                    group_from_xyz_N = group_from_xyz.size(1)
                                    idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                    pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                    indx_batch = torch.cat((indx_batch, idx), dim=1)

                                else:
                                    continue
                                # else: # deep = 3
                                #     size_2dL_L3, xyz_2dL_L3, FPS_2dL_L3 = part2_and_count(xyz_L2, FPS_th, 2) # actually the FPS_th2 is useless here
                                #     for l in range(0, 2):
                                #         if (size_2dL_L3[l] != 0):
                                #             xyz_L3 = xyz_2dL_L3[l]
                                #             if (FPS_2dL_L3[l]==1):
                                #                 centerXYZ, centerXYZ_size = selectAfromB(xyz_L3, new_xyz[i,:,:].unsqueeze(0))
                                #                 if(centerXYZ_size != 0):
                                #                     group_from_xyz = xyz_L2 # group source
                                #                     group_from_xyz_N = group_from_xyz.size(1)
                                #                     idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                     pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                     indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                     if(i==23):
                                #                         print(k,l,'depth=', 3, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                 else:
                                #                     continue
                                #             else: # deep = 4
                                #                 size_2dL_L4, xyz_2dL_L4, FPS_2dL_L4 = part2_and_count(xyz_L3, FPS_th, 3) # actually the FPS_th2 is useless here
                                #                 for m in range(0, 2):
                                #                     # if([i,k,l,m] == [23,0,1,0]):
                                #                     #     pdb.set_trace()
                                #                     if (size_2dL_L4[m]!=0):
                                #                         xyz_L4 = xyz_2dL_L4[m]
                                #                         if (FPS_2dL_L4[m]==1):
                                #                             centerXYZ, centerXYZ_size = selectAfromB(xyz_L4, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                             if(centerXYZ_size != 0):
                                #                                 group_from_xyz = xyz_L3 # group source
                                #                                 group_from_xyz_N = group_from_xyz.size(1)
                                #                                 idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                 pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                 indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                 if(i==23):
                                #                                     print(k,l,m,'depth=', 4, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                             else:
                                #                                 continue
                                #                         else: #deep=5
                                #                             size_2dL_L5, xyz_2dL_L5, FPS_2dL_L5 = part2_and_count(xyz_L4, FPS_th, 4) # actually the FPS_th2 is useless here
                                #                             for n in range(0, 2):
                                #                                 if (size_2dL_L5[n]!=0):
                                #                                     xyz_L5 = xyz_2dL_L5[n]
                                #                                     if (FPS_2dL_L5[n]==1):
                                #                                         centerXYZ, centerXYZ_size = selectAfromB(xyz_L5, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                                         if(centerXYZ_size != 0):
                                #                                             group_from_xyz = xyz_L4 # group source
                                #                                             group_from_xyz_N = group_from_xyz.size(1)
                                #                                             idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                             pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                             indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                             if(i==23):
                                #                                                 print(k,l,m,n, 'depth=', 5, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                                         else:
                                #                                             continue
                                #                                     else:#deep 6
                                #                                         size_2dL_L6, xyz_2dL_L6, FPS_2dL_L6 = part2_and_count(xyz_L5, FPS_th, 5) # actually the FPS_th2 is useless here
                                #                                         for o in range(0, 2):
                                #                                             if (size_2dL_L6[o] != 0):
                                #                                                 xyz_L6 = xyz_2dL_L6[o]
                                #                                                 if (FPS_2dL_L6[o]==1):
                                #                                                     centerXYZ, centerXYZ_size = selectAfromB(xyz_L6, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                                                     if(centerXYZ_size != 0):
                                #                                                         group_from_xyz = xyz_L5 # group source
                                #                                                         group_from_xyz_N = group_from_xyz.size(1)
                                #                                                         idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                                         pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                                         indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                                         if(i==23):
                                #                                                             print(k,l,m,n,o,'depth=', 6, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                                                     else:
                                #                                                         continue
                                #                                                 else:#deep 7
                                #                                                     # if([i,k,l,m,n,o] == [23,0,0,1,1,1]):
                                #                                                     #     pdb.set_trace()
                                #                                                     size_2dL_L7, xyz_2dL_L7, FPS_2dL_L7 = part2_and_count(xyz_L6,FPS_th,6) # actually the FPS_th2 is useless here
                                #                                                     for p in range(0, 2):
                                #                                                         if (size_2dL_L7[p] != 0):
                                #                                                             xyz_L7 = xyz_2dL_L7[p]
                                #                                                             if (FPS_2dL_L7[p]==1):
                                #                                                                 centerXYZ, centerXYZ_size = selectAfromB(xyz_L7, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                                                                 if(centerXYZ_size != 0):
                                #                                                                     group_from_xyz = xyz_L6 # group source
                                #                                                                     group_from_xyz_N = group_from_xyz.size(1)
                                #                                                                     idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                                                     pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                                                     indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                                                     if(i==23):
                                #                                                                         print(k,l,m,n,o,p, 'depth=', 7, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                                                                 else:
                                #                                                                     continue
                                #                                                             else:#deep 8
                                #                                                                 size_2dL_L8, xyz_2dL_L8, FPS_2dL_L8 = part2_and_count(xyz_L7,FPS_th,7) # actually the FPS_th2 is useless here
                                #                                                                 for q in range(0, 2):
                                #                                                                     if (size_2dL_L8[q] != 0):
                                #                                                                         xyz_L8 = xyz_2dL_L8[q]
                                #                                                                         if (FPS_2dL_L8[q]==1):
                                #                                                                             centerXYZ, centerXYZ_size = selectAfromB(xyz_L8, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                                                                             if(centerXYZ_size != 0):
                                #                                                                                 group_from_xyz = xyz_L7 # group source
                                #                                                                                 group_from_xyz_N = group_from_xyz.size(1)
                                #                                                                                 idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                                                                 pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                                                                 indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                                                                 if(i==23):
                                #                                                                                     print(k,l,m,n,o,p,q,'depth=', 8, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                                                                             else:
                                #                                                                                 continue
                                #                                                                         else:#deep 9
                                #                                                                             size_2dL_L9, xyz_2dL_L9, FPS_2dL_L9 = part2_and_count(xyz_L8, FPS_th,8) # actually the FPS_th2 is useless here
                                #                                                                             for r in range(0, 2):
                                #                                                                                 if (size_2dL_L9[r] != 0):
                                #                                                                                     xyz_L9 = xyz_2dL_L9[r]
                                #                                                                                     if (FPS_2dL_L9[r]==1):
                                #                                                                                         centerXYZ, centerXYZ_size = selectAfromB(xyz_L9, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                                                                                         if(centerXYZ_size != 0):
                                #                                                                                             group_from_xyz = xyz_L8 # group source
                                #                                                                                             group_from_xyz_N = group_from_xyz.size(1)
                                #                                                                                             idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                                                                             pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                                                                             indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                                                                             if(i==23):
                                #                                                                                                 print(k,l,m,n,o,p,q,r,'depth=', 9, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                                                                                         else:
                                #                                                                                             continue
                                #                                                                                     else:#deep 10
                                #                                                                                         size_2dL_L10, xyz_2dL_L10, FPS_2dL_L10 = part2_and_count(xyz_L9, FPS_th,9) # actually the FPS_th2 is useless here
                                #                                                                                         for s in range(0, 2):
                                #                                                                                             if (size_2dL_L10[s] != 0):
                                #                                                                                                 xyz_L10 = xyz_2dL_L10[s]
                                #                                                                                                 centerXYZ, centerXYZ_size = selectAfromB(xyz_L10, new_xyz[i,:,:].unsqueeze(0)) # group source
                                #                                                                                                 if(centerXYZ_size != 0):
                                #                                                                                                     group_from_xyz = xyz_L9 # group source
                                #                                                                                                     group_from_xyz_N = group_from_xyz.size(1)
                                #                                                                                                     idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                #                                                                                                     pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                #                                                                                                     indx_batch = torch.cat((indx_batch, idx), dim=1)
                                #                                                                                                     if(i==23):
                                #                                                                                                         print(k,l,m,n,o,p,q,r,s, 'depth=', 10, ', indx.size=', idx.size(1), 'indx_batch.size=', indx_batch.size(1))
                                #                                                                                                 else:
                                #                                                                                                     continue
                                                                                                                            
                            else:
                                continue
                else:
                    continue
            # if(i==23):
            #     pdb.set_trace()

            indx_all = torch.cat((indx_all, indices_batch), dim=0)
            print('batch=', i, ', indx_batch.size=', indx_batch.size(1))
        # groupXyz_indx = find_indices(xyz, groupXyz, B)
            # pdb.set_trace()
    except RuntimeError:
            print(RuntimeError)
            pdb.set_trace()

    return indx_all

def check_unique(a):
    return torch.tensor([row.unique().numel() == row.numel() for row in a]).int()

def blockNum_count(xyz_input, fps_out, blockNum):
    # pdb.set_trace()
    fps_out = fps_out.to(torch.int64).to('cuda:0')
    query_original = torch.gather(xyz_input, 1, fps_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    xyz_0_temp = query_original < 0
    xyz_0_xless05 = torch.abs(query_original)<0.5
    if blockNum == 2:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]
    elif blockNum == 4:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]
    elif blockNum == 8:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]
    elif blockNum == 16:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05[:,:,0]
    else: # now Num is 32
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05[:,:,0]+16*xyz_0_xless05[:,:,1]
    B, N = fps_out.size()
    finalPredictNuminBlock = []
    for b in range(B):
        domain_temp = []
        for i in range(blockNum):
            domain_temp.append(torch.nonzero(xyz_0_temp[b,:]==i).size()[0]) # multiple the number with scale
        finalPredictNuminBlock.append(domain_temp)
    return finalPredictNuminBlock

def find_indices_new(xyz_0, new_xyz):
    # 初始化一个空的索引张量
    indices = torch.zeros((new_xyz.shape[0], new_xyz.shape[1]), dtype=torch.long)
    
    # 遍历每个batch
    for batch_idx in range(new_xyz.shape[0]):
        # 遍历new_xyz中的每个点
        for point_idx in range(new_xyz.shape[1]):
            # 找到new_xyz中的点在xyz_0中的索引
            point = new_xyz[batch_idx, point_idx, :]
            index = torch.where((xyz_0[batch_idx, :, :] == point).all(dim=1))[0][0]
            indices[batch_idx, point_idx] = index
    
    return indices


# xyz_0    = torch.load('./data/PointNeXt-S/model_encoder_encoder_1_0_grouper/XyzinGroup.pt')
# new_xyz = torch.load('./data/PointNeXt-S/model_encoder_encoder_1_0_grouper/newXyzinGroup.pt')
# grouped_xyz = torch.load('./data/PointNeXt-S/model_encoder_encoder_1_0_grouper/grouped_xyz.pt')

xyz_0    = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/support_xyz.pth')
new_xyz = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/query_xyz.pth')
network_idx = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/idx.pth')


# xyz_0    = torch.load('./data/TensorData_S3DIS/support_xyz_grouper_train_24k.pt')
# pdb.set_trace()
# xyz_out0 = torch.load('./data/TensorData_FilterPrune-v4-1-4/model_encoder_encoder_1_0_grouper/grouped_xyz.pt')

nsample = 32
FPS_th = 64 # 这个值不能太小 容易产生负数。。。

xyz_out_original = mygroup(0, 0.15, 32, xyz_0, new_xyz)

# pdb.set_trace()

# 原本串行的方法，用来做对照
# xyz_out_ours = TreeBlock_fps_depth10_config(xyz_0, 512, FPS_th).to('cuda:0') #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)
# 迭代方式
xyz_out_ours =  Tree_group_depth10_config(xyz_0, 32, new_xyz, 0.15) #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)
# indxOfNewXYZ = find_indices_new(xyz_0, new_xyz)
pdb.set_trace()

# bn_ori = blockNum_count(xyz_0, xyz_out_original, BlocCNTkNum)
# xyz_out_original = xyz_out_original.to(torch.int64)
# xyz_out_original_point = torch.gather(xyz_0, 1, xyz_out_original.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)

# blockNumList = [2, 4, 8, 16, 32]
# scaleList = [2, 4, 8, 16, 32]
# coreList = [1, 2, 4, 8, 16, 32, 64]


# =============================
# individual test
# =============================

BlockNum_MAE = 32
# # xyz_out_ours = MultiSteram_blockwise_fps_test(xyz_0, 512, 16, 16, 1, 1) #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)
# xyz_out_ours = MultiSteram_blockwise_fps_test(xyz_0, 512, 8, 16, 1, 1) #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)

# bn_16b = blockNum_count(xyz_0, xyz_out_16block, BlockNum)
# pdb.set_trace()
bn_our = blockNum_count(xyz_0, xyz_out_ours, BlockNum_MAE)
bn_ori = blockNum_count(xyz_0, xyz_out_original, BlockNum_MAE)

sum_16b_total = 0
sum_our_total = 0

md_16b_total = 0
md_our_total = 0
md_new_total = 0

xyz_out_original = xyz_out_original.to(torch.int64)
# xyz_out_16block = xyz_out_16block.to(torch.int64)
xyz_out_ours = xyz_out_ours.to(torch.int64)

xyz_out_original_point = torch.gather(xyz_0, 1, xyz_out_original.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
# xyz_out_16block_point = torch.gather(xyz_0, 1, xyz_out_16block.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
xyz_out_ours_point = torch.gather(xyz_0, 1, xyz_out_ours.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)


for a in range(64):

    sum_16b = 0
    sum_our = 0
    for b in range(BlockNum_MAE):
        # sum_16b += abs(bn_ori[a][b]-bn_16b[a][b])
        sum_our += abs(bn_ori[a][b]-bn_our[a][b])
    # print('for batch', a, ', 16block MAE is ', sum_16b/16,  'our MAE is ', sum_our/16)
    # sum_16b_total += sum_16b
    sum_our_total += sum_our
    # pdb.set_trace()

    # md_16b_total += mahalanobis_distance(a, xyz_out_original_point[a,:,:], xyz_out_16block_point[a,:,:])
    # md_our_total += mahalanobis_distance(a, xyz_out_original_point[a,:,:], xyz_out_ours_point[a,:,:])

# print('for ave, 16block MAE is ', sum_16b_total/(BlockNum*64), 'mahalanobis is ', md_16b_total/64)
MAE = sum_our_total/(BlockNum_MAE*64)
IMD = md_our_total/64
Acc = MAE+IMD

print(MAE, IMD, Acc)