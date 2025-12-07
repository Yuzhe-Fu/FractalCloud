from typing import List, Tuple
from torch.autograd import Function

import torch
import torch.nn as nn

from openpoints.cpp.pointnet2_batch import pointnet2_cuda
from openpoints.models.layers import create_convblock1d
import pdb
from concurrent.futures import ThreadPoolExecutor
import time
import logging

INTERP_COUNT_1D = 0

def find_indices(know: torch.Tensor, xyz_of_idx_all: torch.Tensor) -> torch.Tensor:
    """
    Args:
        know:           [B, N, 3]
        xyz_of_idx_all: [B, N, 3, 3], Each [3] is guaranteed to have a unique match in the know batch
    Returns:
        idx:            [B, N_xyz, M],  Each position stores the index of the matched know
    """
    B, N_know, _ = know.shape

    _, N_xyz, M, _ = xyz_of_idx_all.shape
    # (1) Reshape xyz_of_idx_all to [B, N_xyz * M, 3]
    xyz_flat = xyz_of_idx_all.view(B, N_xyz * M, 3)  # Flatten the second and third dimensions

    # 2) Broadcast comparison:
    #    The shape of xyz_flat.unsqueeze(2) is [B, N*M, 1, 3]
    #    The shape of know.unsqueeze(1) is [B, 1, N,   3]
    #    After broadcasting, we get [B, N*M, N, 3], and all(dim=-1) will compare the last 3 coordinates element-wise
    matches = (xyz_flat.unsqueeze(2) == know.unsqueeze(1)).all(dim=-1)  
    # matches shape: [B, N*M, N], where True means the xyz vector equals to the vector of some index in know

    # 3) Use argmax to find the matching index (guaranteed to have only one match)
    idx_flat = matches.long().argmax(dim=-1)  # [B, N*M]

    idx = idx_flat.view(B, N_xyz, M)
    return idx


def myinterp(unknown: torch.Tensor, known: torch.Tensor):
    """
    Find the three nearest neighbors of unknown in known
    :param unknown: (B, N, 3)
    :param known: (B, M, 3)
    :return:
        dist: (B, N, 3) l2 distance to the three nearest neighbors
        idx: (B, N, 3) index of 3 nearest neighbors
    """
    assert unknown.is_contiguous()
    assert known.is_contiguous()

    B, N, _ = unknown.size()
    m = known.size(1)
    
    dist2 = torch.cuda.FloatTensor(B, N, 3)
    idx = torch.cuda.IntTensor(B, N, 3)

    pointnet2_cuda.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
    return dist2, idx

def selectAfromB(A, B):
    A_2d = A.squeeze(0)
    B_2d = B.squeeze(0)
    matches = (B_2d[:, None] == A_2d).all(-1).any(1)
    indices = torch.where(matches)[0]
    C = B_2d[matches]
    count = C.shape[0]
    return C.unsqueeze(0), count, indices


def process_branch(chunk_size, xyz, xyz_2dL_L1, size_2dL_L1, known_xyz, unknown_xyz, FPS_2dL_L1, FPS_th, i, j, tree_depth, executor=None, unknown_idx_all_batch_list=[], neighbor_known_idx_all_batch_list=[], dist_top_3_all_batch_list=[]):
    if size_2dL_L1[j] != 0:
        xyz_L1 = xyz_2dL_L1[j]
        if FPS_2dL_L1[j] == 1:
            unknown_L1, unknown_L1_size, indices_unknown = selectAfromB_chunk(xyz_L1, unknown_xyz[i], chunk_size)
            known_L1, known_L1_size, _ = selectAfromB_chunk(xyz if tree_depth > 1 else xyz_L1, known_xyz[i], chunk_size)
            if(unknown_L1_size != 0):
                # print(tree_depth)
                dist_top_3, idx_three_known_neighbors = myinterp(unknown_L1, known_L1)
                
                idx_three_known_neighbors_expanded = idx_three_known_neighbors.unsqueeze(-1).expand(-1, -1, -1, 3).long()
                known_xyz_idx = torch.gather(known_L1.unsqueeze(2).expand(-1, -1, 3, -1), 1, idx_three_known_neighbors_expanded)

                return indices_unknown, known_xyz_idx, dist_top_3, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list
            else:
                return None, None, None, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list
        else:
            indices_unknown, known_xyz_idx, dist_top_3, new_unknown_idx_all_batch_list, new_neighbor_known_idx_all_batch_list, new_dist_top_3_all_batch_list = TreeBlock_interp_recursive_config(
                xyz_L1, known_xyz, unknown_xyz, FPS_th, chunk_size, tree_depth, executor, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list
            )
            return indices_unknown, known_xyz_idx, dist_top_3, new_unknown_idx_all_batch_list, new_neighbor_known_idx_all_batch_list, new_dist_top_3_all_batch_list
    else:
        return None, None, None, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list

def TreeBlock_interp_recursive_config(xyz, known_xyz, unknown_xyz, FPS_th, chunk_size, tree_depth=0, executor=None, unknown_idx_all_batch_list=[], neighbor_known_idx_all_batch_list=[], dist_top_3_all_batch_list=[]):
    if executor is None:
        # Create a single ThreadPoolExecutor to be shared
        unknown_idx_all_batch_list = []
        neighbor_known_idx_all_batch_list = []
        dist_top_3_all_batch_list = []
        with ThreadPoolExecutor(max_workers=5120) as executor:
            return TreeBlock_interp_recursive_config(xyz, known_xyz, unknown_xyz, FPS_th, chunk_size, tree_depth, executor, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list)

    # print(tree_depth)
    index_unknown_all_batch_list = unknown_idx_all_batch_list
    known_xyz_idx_all_batch_list = neighbor_known_idx_all_batch_list
    dist_top_3_all_batch_list = dist_top_3_all_batch_list
    B, _, _ = xyz.size()

    for i in range(B):  # Loop over batches
        size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count_with_index(xyz[i, :, :].unsqueeze(0), FPS_th, tree_depth)
        futures = [executor.submit(process_branch, chunk_size, xyz[i, :, :].unsqueeze(0), xyz_2dL_L1, size_2dL_L1, known_xyz, unknown_xyz, FPS_2dL_L1, FPS_th, i, j, tree_depth + 1, executor, [], [], []) for j in range(2)]
        index_unknown = [future.result()[0] for future in futures if future.result()[0] is not None]
        known_xyz_idx = [future.result()[1] for future in futures if future.result()[1] is not None]
        dist_top_3 = [future.result()[2] for future in futures if future.result()[2] is not None]

        # Concatenate the results from all branches
        if index_unknown != []:
            index_unknown_batch = torch.cat(index_unknown, dim=0)
            known_xyz_idx_batch = torch.cat(known_xyz_idx, dim=1)
            dist_top_3_batch = torch.cat(dist_top_3, dim=1)

            if index_unknown_batch.shape[0] == unknown_xyz.shape[1]:
                index_unknown_all_batch_list.append(index_unknown_batch)
                known_xyz_idx_all_batch_list.append(known_xyz_idx_batch)
                dist_top_3_all_batch_list.append(dist_top_3_batch)

    if len(index_unknown_all_batch_list) == unknown_xyz.shape[0]:
        index_unknown_all_batch_list = torch.stack(index_unknown_all_batch_list)
        known_xyz_idx_all_batch_list = torch.cat(known_xyz_idx_all_batch_list, dim=0)
        dist_top_3_all_batch_list = torch.cat(dist_top_3_all_batch_list, dim=0)
        return index_unknown_batch, known_xyz_idx_batch, dist_top_3_batch, index_unknown_all_batch_list, known_xyz_idx_all_batch_list, dist_top_3_all_batch_list
    else:
        return index_unknown_batch, known_xyz_idx_batch, dist_top_3_batch, index_unknown_all_batch_list, known_xyz_idx_all_batch_list, dist_top_3_all_batch_list

def find_indices_chunked(know: torch.Tensor, xyz_of_idx_all: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Args:
        know:           [B, N, 3]
        xyz_of_idx_all: [B, N, 3, 3], Each [3] is guaranteed to have a unique match in the know batch
        chunk_size:     The size of N to process each time, to prevent CUDA memory overflow
    Returns:
        idx:            [B, N, 3], Each position stores the index of the matched know
    """
    B, N_know, _ = know.shape
    _, N_xyz, M, _ = xyz_of_idx_all.shape

    # Initialize the result tensor
    idx = torch.zeros(B, N_xyz, M, dtype=torch.long, device=xyz_of_idx_all.device)
    
    # Split along the second dimension (N_xyz)
    num_chunks = (N_xyz + chunk_size - 1) // chunk_size  # Number of chunks
    xyz_chunks = torch.chunk(xyz_of_idx_all, chunks=num_chunks, dim=1)

    start = 0
    for xyz_chunk in xyz_chunks:
        chunk_size_actual = xyz_chunk.shape[1]  # Current chunk size along the N dimension
        xyz_flat = xyz_chunk.view(B, chunk_size_actual * M, 3)  # Flatten along N_chunk and M
        
        # Compute matches
        matches = (xyz_flat.unsqueeze(2) == know.unsqueeze(1)).all(dim=-1)
        
        # Find the indices
        idx_flat = matches.long().argmax(dim=-1)  # [B, chunk_size_actual * M]
        idx_chunk = idx_flat.view(B, chunk_size_actual, M)
        
        # Place the chunk result into the final result tensor
        idx[:, start:start + chunk_size_actual, :] = idx_chunk
        start += chunk_size_actual

    return idx

def selectAfromB_chunk(A, B, chunk_size=100):
    B_2d = B.squeeze(0)

    matches = torch.zeros(B_2d.shape[0], dtype=torch.bool, device=B.device)
    for j in range(0, B_2d.size(0), chunk_size):
        B_chunk = B_2d[j:j+chunk_size]
        chunk_matches = (B_chunk[:, None] == A).all(-1).any(1)
        matches[j:j+chunk_size] |= chunk_matches  # Aggregate results

    indices = torch.where(matches)[0]
    C = B_2d[matches]
    count = C.shape[0]
    return C.unsqueeze(0), count, indices

def part2_and_count_with_index(xyz, FPS_th, TreeDepth):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    direction = TreeDepth % 3
    xyz_onedirec = xyz[:, :, direction]
    # xyz_not5 = xyz_onedirec[xyz_onedirec != 5.0]
    max_val = torch.max(xyz_onedirec)
    min_val = torch.min(xyz_onedirec)
    mid_val = (max_val + min_val)/2

    if mid_val == max_val:
        xyz_onedirec = xyz[:, :, ((TreeDepth+1) % 3)]
        # xyz_not5 = xyz_onedirec[xyz_onedirec != 5.0]
        max_val = torch.max(xyz_onedirec)
        min_val = torch.min(xyz_onedirec)
        mid_val = (max_val + min_val)/2
        assert max_val != min_val

    xyz_lessMid_indx = torch.nonzero((xyz_onedirec < mid_val))
    xyz_largMid_indx = torch.nonzero((xyz_onedirec >= mid_val))
    
    size_2dTensor  = [0 for _ in range(2)]
    xyz_2dList = [0 for _ in range(2)]
    FPS_2dList = [0 for _ in range(2)]

    size_2dTensor[0] = xyz_lessMid_indx.size()[0]
    size_2dTensor[1] = xyz_largMid_indx.size()[0]
    FPS_2dList[0] = 1 if size_2dTensor[0] <= FPS_th else 0
    FPS_2dList[1] = 1 if size_2dTensor[1] <= FPS_th else 0

    xyz_2dList[0] = xyz[:, xyz_lessMid_indx[:,1], :]
    xyz_2dList[1] = xyz[:, xyz_largMid_indx[:,1], :]


    return size_2dTensor, xyz_2dList, FPS_2dList

def part2_and_count_with_index_kdTree(xyz, FPS_th, TreeDepth):
    direction = TreeDepth % 3
    xyz_onedirec = xyz[:, :, direction]

    # Sort the values in the selected dimension
    sorted_vals, sorted_indices = torch.sort(xyz_onedirec, dim=1)
    num_points = xyz_onedirec.size(1)

    # Calculate the mid-point index
    mid_idx = num_points // 2

    # The indices of the left and right halves
    left_indices = sorted_indices[:, :mid_idx]
    right_indices = sorted_indices[:, mid_idx:]

    size_2dTensor = [left_indices.size(1), right_indices.size(1)]

    FPS_2dList = [1 if size_2dTensor[i] <= FPS_th else 0 for i in range(2)]

    xyz_2dList = [xyz[:, left_indices[0, :], :], xyz[:, right_indices[0, :], :]]

    return size_2dTensor, xyz_2dList, FPS_2dList



def Fractal_interp_on_batch(know, unknow, FPS_th=128, chunk_size = 1000):
    # xyz = torch.cat((unknow, know), dim=1)
    xyz = unknow
    B = xyz.size(0)
    dist2 = []
    idx = []
    for batch in range(0, B):
        _, _, _, indices_unknown_global, known_xyz_idx, dist2_global =  TreeBlock_interp_recursive_forBatchIteration(xyz[batch,:].unsqueeze(0), know[batch,:].unsqueeze(0), unknow[batch,:].unsqueeze(0), FPS_th, chunk_size, 0, None) #(xyz, npoint, blockNum, scale, PMS=1, BMS=1)
        sorted_indices = torch.argsort(indices_unknown_global, dim=1) # [B, N]
        # use sorted_indices to sort known_xyz_idx and dist2_global, where known_xyz_idx is the xyz of known points with shape [B, N, 3, 3], dist2_global is the distance of 3 nearest neighbors with shape [B, N, 3]
        known_xyz_idx_sorted = torch.gather(known_xyz_idx.cuda(), 1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        dist2_global_sorted = torch.gather(dist2_global.cuda(), 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3))
        idx_all_recursive = find_indices_chunked(know[batch,:].unsqueeze(0), known_xyz_idx_sorted, chunk_size=1000)

        dist2.append(dist2_global_sorted)
        idx.append(idx_all_recursive)
    dist2 = torch.cat(dist2, dim=0)
    idx = torch.cat(idx, dim=0).int()
    return dist2, idx

def process_branch_forBatchIteration(chunk_size, xyz, xyz_2dL_L1, size_2dL_L1, known_xyz, unknown_xyz, FPS_2dL_L1, FPS_th, i, j, tree_depth, executor=None, unknown_idx_all_batch_list=[], neighbor_known_idx_all_batch_list=[], dist_top_3_all_batch_list=[]):
    # pdb.set_trace()
    if size_2dL_L1[j] != 0:
        xyz_L1 = xyz_2dL_L1[j]
        if FPS_2dL_L1[j] == 1:
            unknown_L1, unknown_L1_size, indices_unknown = selectAfromB_chunk(xyz_L1, unknown_xyz[i], chunk_size)
            known_L1, known_L1_size, _ = selectAfromB_chunk(xyz if tree_depth > 1 else xyz_L1, known_xyz[i], chunk_size)
            if(unknown_L1_size != 0):
                # print(tree_depth)
                dist_top_3, idx_three_known_neighbors = myinterp(unknown_L1, known_L1)
                # dist_top_3_batch = torch.cat((dist_top_3_batch, dist_top_3), dim=1)
                idx_three_known_neighbors_expanded = idx_three_known_neighbors.unsqueeze(-1).expand(-1, -1, -1, 3).long()
                known_xyz_idx = torch.gather(known_L1.unsqueeze(2).expand(-1, -1, 3, -1), 1, idx_three_known_neighbors_expanded)

                return indices_unknown, known_xyz_idx, dist_top_3, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list
            else:
                return None, None, None, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list
        else:
            indices_unknown, known_xyz_idx, dist_top_3, new_unknown_idx_all_batch_list, new_neighbor_known_idx_all_batch_list, new_dist_top_3_all_batch_list = TreeBlock_interp_recursive_forBatchIteration(
                xyz_L1, known_xyz, unknown_xyz, FPS_th, chunk_size, tree_depth, executor, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list
            )
            return indices_unknown, known_xyz_idx, dist_top_3, new_unknown_idx_all_batch_list, new_neighbor_known_idx_all_batch_list, new_dist_top_3_all_batch_list
    else:
        return None, None, None, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list

def TreeBlock_interp_recursive_forBatchIteration(xyz, known_xyz, unknown_xyz, FPS_th, chunk_size, tree_depth=0, executor=None, unknown_idx_all_batch_list=[], neighbor_known_idx_all_batch_list=[], dist_top_3_all_batch_list=[]):
    if executor is None:
        # Create a single ThreadPoolExecutor to be shared
        unknown_idx_all_batch_list = []
        neighbor_known_idx_all_batch_list = []
        dist_top_3_all_batch_list = []
        # pdb.set_trace()
        with ThreadPoolExecutor(max_workers=4096) as executor:
            return TreeBlock_interp_recursive_forBatchIteration(xyz, known_xyz, unknown_xyz, FPS_th, chunk_size, tree_depth, executor, unknown_idx_all_batch_list, neighbor_known_idx_all_batch_list, dist_top_3_all_batch_list)

    # print(tree_depth)
    index_unknown_all_batch_list = unknown_idx_all_batch_list
    known_xyz_idx_all_batch_list = neighbor_known_idx_all_batch_list
    dist_top_3_all_batch_list = dist_top_3_all_batch_list
    
    if xyz.ndim == 3:
        B, N, _ = xyz.size()
    else:
        B = 1
        N = xyz.size(0)
        # xyz = xyz.unsqueeze(0)  # Add batch dimension if missing

    # for i in range(B):  # Loop over batches
    i=0
    size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count_with_index(xyz[i, :, :].unsqueeze(0), FPS_th, tree_depth)
    # size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count_with_index_kdTree(xyz[i, :, :].unsqueeze(0), FPS_th, tree_depth)

    futures = [executor.submit(process_branch_forBatchIteration, chunk_size, xyz[i, :, :].unsqueeze(0), xyz_2dL_L1, size_2dL_L1, known_xyz, unknown_xyz, FPS_2dL_L1, FPS_th, i, j, tree_depth + 1, executor, [], [], []) for j in range(2)]
    index_unknown = [future.result()[0] for future in futures if future.result()[0] is not None]
    known_xyz_idx = [future.result()[1] for future in futures if future.result()[1] is not None]
    dist_top_3 = [future.result()[2] for future in futures if future.result()[2] is not None]

    # Concatenate the results from all branches
    # pdb.set_trace()
    if index_unknown != []:
        index_unknown_batch = torch.cat(index_unknown, dim=0)
        known_xyz_idx_batch = torch.cat(known_xyz_idx, dim=1)
        dist_top_3_batch = torch.cat(dist_top_3, dim=1)
        # index_global_center = torch.cat([index_global_center, indices_center_batch], dim=0)
        # index_global_group = torch.cat([index_global_group, indx_group_batch], dim=1)

        # pdb.set_trace()
        if index_unknown_batch.shape[0] == unknown_xyz.shape[1]:
            # pdb.set_trace()
            index_unknown_all_batch_list.append(index_unknown_batch)
            known_xyz_idx_all_batch_list.append(known_xyz_idx_batch)
            dist_top_3_all_batch_list.append(dist_top_3_batch)

    # pdb.set_trace()
    if len(index_unknown_all_batch_list) == unknown_xyz.shape[0]:
        # pdb.set_trace()
        index_unknown_all_batch_list = torch.stack(index_unknown_all_batch_list)
        known_xyz_idx_all_batch_list = torch.cat(known_xyz_idx_all_batch_list, dim=0)
        dist_top_3_all_batch_list = torch.cat(dist_top_3_all_batch_list, dim=0)
        executor.shutdown()
        return index_unknown_batch, known_xyz_idx_batch, dist_top_3_batch, index_unknown_all_batch_list, known_xyz_idx_all_batch_list, dist_top_3_all_batch_list
    else:
        # pdb.set_trace()
        return index_unknown_batch, known_xyz_idx_batch, dist_top_3_batch, index_unknown_all_batch_list, known_xyz_idx_all_batch_list, dist_top_3_all_batch_list



_FRACTAL_INTERP_STAGES = None
_FRACTAL_INTERP_TH = 64

def set_fractal_interp_config(stages=None, th=64):
    """set the fractal interp config
    Args:
        stages: list of int or None, for example [1, 2] means using fractal interp in stage 1 and 2, None means using original method
        th: int, the threshold of fractal interp, default 64
    """
    global _FRACTAL_INTERP_STAGES, _FRACTAL_INTERP_TH
    _FRACTAL_INTERP_STAGES = stages
    _FRACTAL_INTERP_TH = th

def get_fractal_interp_stages():
    return _FRACTAL_INTERP_STAGES

def get_fractal_interp_th():
    return _FRACTAL_INTERP_TH


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor, stage: int = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)


        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)
        # stage is not None and 
        # pdb.set_trace()
        # (3,2,1,0) stage setting for PontNet++ in seg, 
        # (-1,-2,-3,-4) stage setting for PointNeXt and PointVector in seg.
        fractal_interp_stages = get_fractal_interp_stages()
        use_fractal_interp = fractal_interp_stages is not None and stage in fractal_interp_stages
        if use_fractal_interp:
            fractal_interp_th = get_fractal_interp_th()
            dist2, idx = Fractal_interp_on_batch(known, unknown, fractal_interp_th, chunk_size = 1000)

        else:
            pointnet2_cuda.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)
        pointnet2_cuda.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.zeros([B, c, m], device='cuda', requires_grad=True)
        grad_out_data = grad_out.data.contiguous()

        pointnet2_cuda.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def three_interpolation(unknown_xyz, known_xyz, know_feat, stage=None):
    """
    input: known_xyz: (m, 3), unknown_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    # torch.cuda.synchronize()
    # time0 = time.time()
    dist, idx = three_nn(unknown_xyz, known_xyz, stage)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    # torch.cuda.synchronize()
    # time1 = time.time()
    # print(f"three_nn time: {time1 - time0}")
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats


if __name__ == "__main__":
    pass
