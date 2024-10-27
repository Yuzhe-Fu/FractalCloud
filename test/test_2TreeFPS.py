
import torch
# import numpy as np
# import torch.nn as nn
# from torch.autograd import Function
# import math
import sys
sys.path.append("/workspace/PointNeXt/")
# import os, sys
from openpoints.cpp.pointnet2_batch import pointnet2_cuda
import pdb
# import open3d as o3d
import asyncio
# import logging
import pdb
# import csv
from concurrent.futures import ThreadPoolExecutor
import time

def process_branch(xyz_2dL_L1, size_2dL_L1, PretNum_L1_checked, FPS_2dL_L1, i, j, tree_depth, global_index, executor=None):
    if size_2dL_L1[i][j] != 0:
        xyz_L1 = xyz_2dL_L1[i][j]
        SampNum_L1 = int(PretNum_L1_checked[j])
        if FPS_2dL_L1[i][j] == 1:
            indx_L1 = my_fps(xyz_L1, SampNum_L1)
            SampXyz_L1 = torch.gather(
                xyz_L1,
                1,
                indx_L1.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1])
            ).type_as(xyz_L1)
            global_index_update = torch.gather(global_index, 1, indx_L1.long())
        else:
            SampXyz_L1, global_index_update = TreeBlock_fps_recursive_config(
                xyz_L1, SampNum_L1, FPS_th, tree_depth, global_index, executor
            )
        return SampXyz_L1, global_index_update
    else:
        return None, None
    
def TreeBlock_fps_recursive_config(xyz, npoint, FPS_th, tree_depth=0, global_index=None, executor=None):
    if executor is None:
        # Create a single ThreadPoolExecutor to be shared
        with ThreadPoolExecutor(max_workers=64) as executor:
            return TreeBlock_fps_recursive_config(xyz, npoint, FPS_th, tree_depth, global_index, executor)

    if xyz.ndim == 3:
        B, N, _ = xyz.size()
    else:
        B = 1
        N = xyz.size(0)
        # xyz = xyz.unsqueeze(0)  # Add batch dimension if missing

    # Tree FPS: Partition and count
    if global_index is None:
        global_index = torch.arange(N).to(xyz.device).unsqueeze(0).repeat(B, 1)
    size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1, global_index = part2_and_count_with_index(xyz, B, FPS_th, global_index, tree_depth)
    
    SampXyz = []
    index_global = []

    for i in range(B):  # Loop over batches
        PretNum_L1 = [0 for _ in range(2)]
        for j in range(2):
            PretNum_L1[j] = round(npoint * size_2dL_L1[i][j] / N)
        PretNum_L1_checked = adjust_list_to_sum(PretNum_L1, npoint)

        # Submit tasks to the shared executor
        futures = [executor.submit(process_branch, xyz_2dL_L1, size_2dL_L1, PretNum_L1_checked, FPS_2dL_L1, i, j, tree_depth + 1, global_index[i][j], executor) for j in range(2)]
        SampXyz_batch = [future.result()[0] for future in futures if future.result()[0] is not None]
        index_batch = [future.result()[1] for future in futures if future.result()[1] is not None]

        # Concatenate the results from all branches
        SampXyz_batch = torch.cat(SampXyz_batch, dim=1)
        index_batch = torch.cat(index_batch, dim=1)
        SampXyz.append(SampXyz_batch)
        index_global.append(index_batch)

    # Concatenate all batches
    SampXyz = torch.cat(SampXyz, dim=0)
    index_global = torch.cat(index_global, dim=0)
    return SampXyz, index_global


def part2_and_count_with_index(xyz, batch_size, FPS_th, ori_index, TreeDepth):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    direction = TreeDepth % 3

    max_val = torch.max(xyz[:, :, direction])
    min_val = torch.min(xyz[:, :, direction])
    mid_val = (max_val + min_val)/2

    xyz_0_temp = xyz[:,:,direction] < mid_val
    
    size_2dTensor  = [[0 for _ in range(2)] for _ in range(batch_size)]
    xyz_2dList = [[0 for _ in range(2)] for _ in range(batch_size)]
    FPS_2dList = [[0 for _ in range(2)] for _ in range(batch_size)]
    global_index = [[0 for _ in range(2)] for _ in range(batch_size)]

    for b in range(batch_size):
        for i in range(2):
            indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape([1,-1]).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
            xyz_row = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx))
            xyz_2dList[b][i] = xyz_row
            size_2dTensor[b][i] = indx.size()[1]
            global_index[b][i] = torch.gather(ori_index[b].reshape([1,-1]), 1, indx[..., 0])
            if indx.size()[1] <= FPS_th:
                FPS_2dList[b][i] = 1

    return size_2dTensor, xyz_2dList, FPS_2dList, global_index



def compute_mean_cov(X):
    mean = torch.mean(X, dim=0)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    return mean, cov

def mahalanobis_distance(batch, a, b):
    try:
        a[0,:] += 1e-5
        mean_a, cov_a = compute_mean_cov(a)
        mean_b, cov_b = compute_mean_cov(b)
        diff = mean_a - mean_b
        cov_sum_inv = torch.inverse(cov_a + cov_b)
        dist = torch.sqrt(diff @ cov_sum_inv @ diff.T)
    except Exception as e:
        pdb.set_trace()
    return dist

def spar_tensor(arr, scale, starPoint=0):
    return arr[:,starPoint::scale,:]

def my_fps(xyz, npoint):
    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
    pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    return output

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
        # pdb.set_trace()
    
    return torch.tensor(c)



def find_indices_for_recursive(a, b, batch):
    try:
        c = []
        for i in range(0, batch):
            c_matrix = []
            # for b_row in b[i,:]:
            # 找到与b_row相等的行的索引
            a_row=a[i,:]
            for a_ele in a_row:
                idx = (a_ele == b[i]).all(dim=1).nonzero(as_tuple=True)[0]
                if idx.size()[0] == 1:
                    c_matrix.append(idx.item())
                else:
                    c_matrix.append(idx[0].item())
            c.append(c_matrix)
    except ValueError:
        print(ValueError)
        # # pdb.set_trace()
    
    return torch.tensor(c).to(b.device)

# ################ make sure the total predict num is consistant
def checkTotalNum(PredNumList, npoint):
    sumOfPN = sum(PredNumList)
    if sumOfPN != npoint:
        dif = npoint - sumOfPN
        max_value = max(PredNumList)
        max_position = PredNumList.index(max_value)
        PredNumList[max_position] = PredNumList[max_position] + dif
    return PredNumList

def adjust_list_to_sum(numbers, target_sum):
    current_sum = sum(numbers)
    if current_sum == target_sum:
        return numbers
    difference = target_sum - current_sum
    if(numbers[0] < numbers[1]):
        numbers[1] += difference
    else:
        numbers[0] += difference 
    return numbers


# ################ block partition with different directions and count the size
def part2_and_count(xyz, batch_size, FPS_th, TreeDepth):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    direction = TreeDepth % 3

    max_val = torch.max(xyz[:, :, direction])
    min_val = torch.min(xyz[:, :, direction])
    mid_val = (max_val + min_val)/2


    xyz_0_temp = xyz[:,:,direction] < mid_val
    
    size_2dTensor  = [[0 for _ in range(2)] for _ in range(batch_size)]
    xyz_2dList = [[0 for _ in range(2)] for _ in range(batch_size)]
    FPS_2dList = [[0 for _ in range(2)] for _ in range(batch_size)]

    for b in range(batch_size):
        for i in range(2):
                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape([1,-1]).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
                xyz_row = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx))
                xyz_2dList[b][i] = xyz_row
                size_2dTensor[b][i] = indx.size()[1]
                if indx.size()[1] <= FPS_th:
                    FPS_2dList[b][i] = 1

    return size_2dTensor, xyz_2dList, FPS_2dList

def TreeBlock_fps_depth10_config(xyz, npoint, FPS_th):
    B, N, _ = xyz.size()

    # tree FPS
    # firstly tree block partition

    size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count(xyz, B, FPS_th, 0)

    SampXyz = torch.tensor([], device='cuda')
    try:

        for i in range(0, B): # deep=1
            SampXyz_batch = torch.tensor([], device='cuda')
            PretNum_L1 = [0 for _ in range(2)]
            for j in range(0, 2):
                PretNum_L1[j] = round(npoint*size_2dL_L1[i][j]/N)
            PretNum_L1_checked = adjust_list_to_sum(PretNum_L1, npoint) # 确保累加起来是需要的值
            for j in range(0, 2):
                if (size_2dL_L1[i][j] != 0):
                    xyz_L1 = xyz_2dL_L1[i][j]
                    SampNum_L1 = int(PretNum_L1_checked[j])
                    if (FPS_2dL_L1[i][j]==1):
                        indx_L1 = my_fps(xyz_L1, SampNum_L1)
                        SampXyz_L1 = torch.gather(xyz_L1, 1, indx_L1.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                        SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L1), dim=1)
                    else: # 开始第二级树分块
                        size_2dL_L2, xyz_2dL_L2, FPS_2dL_L2 = part2_and_count(xyz_L1, 1, FPS_th, 1)
                        PretNum_L2 = [0 for _ in range(2)]
                        for k in range(0, 2):
                            PretNum_L2[k] = round(SampNum_L1*size_2dL_L2[0][k]/size_2dL_L1[i][j])
                        PretNum_L2_checked = adjust_list_to_sum(PretNum_L2, SampNum_L1) # 确保累加起来是需要的值
                        for k in range(0, 2):
                            if (size_2dL_L2[0][k] != 0):
                                # check the size of FPS-2dL-L2
                                # # pdb.set_trace()
                                xyz_L2 = xyz_2dL_L2[0][k]
                                SampNum_L2 = int(PretNum_L2_checked[k])
                                if (FPS_2dL_L2[0][k]==1):
                                    indx_L2_temp = my_fps(xyz_L2, SampNum_L2)
                                    SampXyz_L2 = torch.gather(xyz_L2, 1, indx_L2_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                    SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L2), dim=1)
                                else: # 开始第三级树分块
                                    size_2dL_L3, xyz_2dL_L3, FPS_2dL_L3 = part2_and_count(xyz_L2, 1, FPS_th, 2) # actually the FPS_th2 is useless here
                                    # 这里要计算SampNum_L1在L3各个层上的加权平均数了
                                    PretNum_L3 = [0 for _ in range(2)]
                                    for l in range(0, 2):
                                        PretNum_L3[l] = round(SampNum_L2*size_2dL_L3[0][l]/size_2dL_L2[0][k])
                                    PretNum_L3_checked = adjust_list_to_sum(PretNum_L3, SampNum_L2) # 确保累加起来是需要的值
                                    for l in range(0, 2):
                                        if (size_2dL_L3[0][l] != 0):
                                            xyz_L3 = xyz_2dL_L3[0][l]
                                            SampNum_L3 = int(PretNum_L3_checked[l])
                                            if (FPS_2dL_L3[0][l]==1):
                                                indx_L3_temp = my_fps(xyz_L3, SampNum_L3)
                                                SampXyz_L3 = torch.gather(xyz_L3, 1, indx_L3_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L3), dim=1)
                                            else: # deep = 4
                                                size_2dL_L4, xyz_2dL_L4, FPS_2dL_L4 = part2_and_count(xyz_L3, 1, FPS_th, 3) # actually the FPS_th2 is useless here
                                                PretNum_L4 = [0 for _ in range(2)]
                                                for m in range(0, 2):
                                                    PretNum_L4[m] = round(SampNum_L3*size_2dL_L4[0][m]/size_2dL_L3[0][l])
                                                PretNum_L4_checked = adjust_list_to_sum(PretNum_L4, SampNum_L3) # 确保累加起来是需要的值
                                                for m in range(0, 2):
                                                    if (size_2dL_L4[0][m] != 0):
                                                        xyz_L4 = xyz_2dL_L4[0][m]
                                                        SampNum_L4 = int(PretNum_L4_checked[m])
                                                        if (FPS_2dL_L4[0][m]==1):
                                                            indx_L4_temp = my_fps(xyz_L4, SampNum_L4)
                                                            SampXyz_L4 = torch.gather(xyz_L4, 1, indx_L4_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                            SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L4), dim=1)
                                                        else: #deep=5
                                                            size_2dL_L5, xyz_2dL_L5, FPS_2dL_L5 = part2_and_count(xyz_L4, 1, FPS_th, 4) # actually the FPS_th2 is useless here
                                                            PretNum_L5 = [0 for _ in range(2)]
                                                            for n in range(0, 2):
                                                                PretNum_L5[n] = round(SampNum_L4*size_2dL_L5[0][n]/size_2dL_L4[0][m])
                                                            PretNum_L5_checked = adjust_list_to_sum(PretNum_L5, SampNum_L4) # 确保累加起来是需要的值
                                                            for n in range(0, 2):
                                                                if (size_2dL_L5[0][n] != 0):
                                                                    xyz_L5 = xyz_2dL_L5[0][n]
                                                                    SampNum_L5 = int(PretNum_L5_checked[n])
                                                                    if (FPS_2dL_L5[0][n]==1):
                                                                        indx_L5_temp = my_fps(xyz_L5, SampNum_L5)
                                                                        SampXyz_L5 = torch.gather(xyz_L5, 1, indx_L5_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                        SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L5), dim=1)
                                                                    else:#deep 6
                                                                        size_2dL_L6, xyz_2dL_L6, FPS_2dL_L6 = part2_and_count(xyz_L5, 1, FPS_th, 5) # actually the FPS_th2 is useless here
                                                                        PretNum_L6 = [0 for _ in range(2)]
                                                                        for o in range(0, 2):
                                                                            PretNum_L6[o] = round(SampNum_L5*size_2dL_L6[0][o]/size_2dL_L5[0][n])
                                                                        PretNum_L6_checked = adjust_list_to_sum(PretNum_L6, SampNum_L5) # 确保累加起来是需要的值
                                                                        for o in range(0, 2):
                                                                            if (size_2dL_L6[0][o] != 0):
                                                                                xyz_L6 = xyz_2dL_L6[0][o]
                                                                                SampNum_L6 = int(PretNum_L6_checked[o])
                                                                                if (FPS_2dL_L6[0][o]==1):
                                                                                    indx_L6_temp = my_fps(xyz_L6, SampNum_L6)
                                                                                    SampXyz_L6 = torch.gather(xyz_L6, 1, indx_L6_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                    SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L6), dim=1)
                                                                                else:#deep 7
                                                                                    size_2dL_L7, xyz_2dL_L7, FPS_2dL_L7 = part2_and_count(xyz_L6, 1, FPS_th,6) # actually the FPS_th2 is useless here
                                                                                    PretNum_L7 = [0 for _ in range(2)]
                                                                                    for p in range(0, 2):
                                                                                        PretNum_L7[p] = round(SampNum_L6*size_2dL_L7[0][p]/size_2dL_L6[0][o])
                                                                                    PretNum_L7_checked = adjust_list_to_sum(PretNum_L7, SampNum_L6) # 确保累加起来是需要的值
                                                                                    for p in range(0, 2):
                                                                                        if (size_2dL_L7[0][p] != 0):
                                                                                            xyz_L7 = xyz_2dL_L7[0][p]
                                                                                            SampNum_L7 = int(PretNum_L7_checked[p])
                                                                                            if (FPS_2dL_L7[0][p]==1):
                                                                                                indx_L7_temp = my_fps(xyz_L7, SampNum_L7)
                                                                                                SampXyz_L7 = torch.gather(xyz_L7, 1, indx_L7_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L7), dim=1)
                                                                                            else:#deep 8
                                                                                                size_2dL_L8, xyz_2dL_L8, FPS_2dL_L8 = part2_and_count(xyz_L7, 1, FPS_th,7) # actually the FPS_th2 is useless here
                                                                                                PretNum_L8 = [0 for _ in range(2)]
                                                                                                for q in range(0, 2):
                                                                                                    PretNum_L8[q] = round(SampNum_L7*size_2dL_L8[0][q]/size_2dL_L7[0][p])
                                                                                                PretNum_L8_checked = adjust_list_to_sum(PretNum_L8, SampNum_L7) # 确保累加起来是需要的值
                                                                                                for q in range(0, 2):
                                                                                                    if (size_2dL_L8[0][q] != 0):
                                                                                                        xyz_L8 = xyz_2dL_L8[0][q]
                                                                                                        SampNum_L8 = int(PretNum_L8_checked[q])
                                                                                                        if (FPS_2dL_L8[0][q]==1):
                                                                                                            indx_L8_temp = my_fps(xyz_L8, SampNum_L8)
                                                                                                            SampXyz_L8 = torch.gather(xyz_L8, 1, indx_L8_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                            SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L8), dim=1)
                                                                                                        else:#deep 9
                                                                                                            size_2dL_L9, xyz_2dL_L9, FPS_2dL_L9 = part2_and_count(xyz_L8, 1, FPS_th,8) # actually the FPS_th2 is useless here
                                                                                                            PretNum_L9 = [0 for _ in range(2)]
                                                                                                            for r in range(0, 2):
                                                                                                                PretNum_L9[r] = round(SampNum_L8*size_2dL_L9[0][r]/size_2dL_L8[0][q])
                                                                                                            PretNum_L9_checked = adjust_list_to_sum(PretNum_L9, SampNum_L8) # 确保累加起来是需要的值
                                                                                                            for r in range(0, 2):
                                                                                                                if (size_2dL_L9[0][r] != 0):
                                                                                                                    xyz_L9 = xyz_2dL_L9[0][r]
                                                                                                                    SampNum_L9 = int(PretNum_L9_checked[r])
                                                                                                                    if (FPS_2dL_L9[0][r]==1):
                                                                                                                        indx_L9_temp = my_fps(xyz_L9, SampNum_L9)
                                                                                                                        SampXyz_L9 = torch.gather(xyz_L9, 1, indx_L9_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                                        SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L9), dim=1)
                                                                                                                    else:#deep 10
                                                                                                                        size_2dL_L10, xyz_2dL_L10, FPS_2dL_L10 = part2_and_count(xyz_L9, 1, FPS_th,9) # actually the FPS_th2 is useless here
                                                                                                                        PretNum_L10 = [0 for _ in range(2)]
                                                                                                                        for s in range(0, 2):
                                                                                                                            PretNum_L10[s] = round(SampNum_L9*size_2dL_L10[0][s]/size_2dL_L9[0][r])
                                                                                                                        PretNum_L10_checked = adjust_list_to_sum(PretNum_L10, SampNum_L9) # 确保累加起来是需要的值
                                                                                                                        for s in range(0, 2):
                                                                                                                            if (size_2dL_L10[0][s] != 0):
                                                                                                                                xyz_L10 = xyz_2dL_L10[0][s]
                                                                                                                                SampNum_L10 = int(PretNum_L10_checked[s])
                                                                                                                                indx_L10_temp = my_fps(xyz_L10, SampNum_L10)
                                                                                                                                SampXyz_L10 = torch.gather(xyz_L10, 1, indx_L10_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                                                SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L10), dim=1)



                            else:
                                continue
                else:
                    continue
            # # pdb.set_trace()
            SampXyz = torch.cat((SampXyz, SampXyz_batch), dim=0)
                
        SampXyz_indx = find_indices(xyz, SampXyz, B)
    except RuntimeError:
            print(RuntimeError)
            # pdb.set_trace()

    return SampXyz_indx

# async def TreeBlock_fps_recursive_config(xyz, npoint, FPS_th):
#     if xyz.ndim == 3:
#         B, N, _ = xyz.size()
#     else:
#         B = 1
#         N = xyz.size()[0]
#     # tree FPS
#     # firstly tree block partition

#     size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count(xyz, B, FPS_th, 0)
#     SampXyz = torch.tensor([], device='cuda')
#     # try:

#     async for i in range(0, B): # deep=1
#         SampXyz_batch = torch.tensor([], device='cuda')
#         PretNum_L1 = [0 for _ in range(2)]
#         for j in range(0, 2):
#             PretNum_L1[j] = round(npoint*size_2dL_L1[i][j]/N)
#         PretNum_L1_checked = adjust_list_to_sum(PretNum_L1, npoint) # 确保累加起来是需要的值
#         async for j in range(0, 2):
#             if (size_2dL_L1[i][j] != 0):
#                 xyz_L1 = xyz_2dL_L1[i][j]
#                 SampNum_L1 = int(PretNum_L1_checked[j])
#                 if (FPS_2dL_L1[i][j]==1):
#                     indx_L1 = my_fps(xyz_L1, SampNum_L1)
#                     SampXyz_L1 = torch.gather(xyz_L1, 1, indx_L1.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
#                     SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L1), dim=1)
#                 else: # 开始第二级树分块
#                     SampXyz_batch = await TreeBlock_fps_recursive_config(xyz_L1, SampNum_L1, FPS_th)
#             else:
#                 continue
#         # # pdb.set_trace()
#         SampXyz = torch.cat((SampXyz, SampXyz_batch), dim=0)
#     return SampXyz


def check_unique(a):
    return torch.tensor([row.unique().numel() == row.numel() for row in a]).int()

def blockNum_count(xyz_input, fps_out, blockNum):
    # # pdb.set_trace()
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


xyz_0    = torch.load('./data/PointNeXt-S/model_encoder_encoder_1_0_grouper/support_xyz.pt').cuda()
# [64,1024,3]
# xyz_0    = torch.load('./data/TensorData_S3DIS/support_xyz_grouper_train_24k.pt')
# # pdb.set_trace()
# xyz_out0 = torch.load('./data/TensorData_FilterPrune-v4-1-4/model_encoder_encoder_1_0_grouper/grouped_xyz.pt')


FPS_th = 32 # 这个值不能太小 容易产生负数。。。

xyz_out_original = my_fps(xyz_0, 512)

# 原本串行的方法，用来做对照
start_time = time.time()
xyz_out_ours = TreeBlock_fps_depth10_config(xyz_0, 512, FPS_th).to('cuda:0') #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)
duration = time.time() - start_time
print(f'depth10_duration: {duration}')

# 迭代方式
# # pdb.set_trace()
start_time = time.time()
_, xyz_out_ours = TreeBlock_fps_recursive_config(xyz_0, 512, FPS_th) #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)
duration = time.time() - start_time
print(f'recursive_duration: {duration}')
# xyz_out_ours = find_indices_for_recursive(xyz_out_ours, xyz_0, xyz_0.shape[0])

# # pdb.set_trace()

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
# # pdb.set_trace()
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
    # # pdb.set_trace()

    # md_16b_total += mahalanobis_distance(a, xyz_out_original_point[a,:,:], xyz_out_16block_point[a,:,:])
    md_our_total += mahalanobis_distance(a, xyz_out_original_point[a,:,:], xyz_out_ours_point[a,:,:])

# print('for ave, 16block MAE is ', sum_16b_total/(BlockNum*64), 'mahalanobis is ', md_16b_total/64)
MAE = sum_our_total/(BlockNum_MAE*64)
IMD = md_our_total/64
Acc = MAE+IMD

print(MAE, IMD, Acc)