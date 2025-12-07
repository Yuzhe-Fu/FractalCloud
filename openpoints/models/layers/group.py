# group layer: find neighbors for each point
# knn, knn_sparse, ball query

# gather layer, gather features by index
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function
from openpoints.cpp import pointnet2_cuda
import pdb
from concurrent.futures import ThreadPoolExecutor
import logging
import time

GROUP_COUNT_1D = 0  # global variable to count the number of group operations

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply

# add TreeGroup here
# ################ block partition with different directions and count the size

def selectAfromB(A, B):
    A_2d = A.squeeze(0)
    B_2d = B.squeeze(0)
    matches = (B_2d[:, None] == A_2d).all(-1).any(1)
    indices = torch.where(matches)[0]
    C = B_2d[matches]
    count = C.shape[0]
    return C.unsqueeze(0), count, indices

def selectAfromB_chunk(A, B, chunk_size=100): # chunk here is for not OOM.
    matches = torch.zeros(B.shape[0], dtype=torch.bool, device=B.device)

    for j in range(0, B.size(0), chunk_size):
        B_chunk = B[j:j+chunk_size]
        chunk_matches = (B_chunk[:, None] == A).all(-1).any(1)
        matches[j:j+chunk_size] |= chunk_matches 

    indices = torch.where(matches)[0]
    C = B[matches]
    count = C.shape[0]

    return C.unsqueeze(0), count, indices




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


def part2_and_count_with_index(xyz, FPS_th, ori_index, TreeDepth):
    # a list with (3, 4) shape
    direction = TreeDepth % 3
    xyz_onedirec = xyz[:, :, direction]
    max_val = torch.max(xyz_onedirec)
    min_val = torch.min(xyz_onedirec)
    mid_val = (max_val + min_val)/2

    if mid_val == max_val:
        xyz_onedirec = xyz[:, :, ((TreeDepth+1) % 3)]
        max_val = torch.max(xyz_onedirec)
        min_val = torch.min(xyz_onedirec)
        mid_val = (max_val + min_val)/2
        assert max_val != min_val

    xyz_lessMid_indx = torch.nonzero((xyz_onedirec < mid_val))
    xyz_largMid_indx = torch.nonzero((xyz_onedirec >= mid_val))
    
    size_2dTensor  = [0 for _ in range(2)]
    xyz_2dList = [0 for _ in range(2)]
    FPS_2dList = [0 for _ in range(2)]
    global_index = [0 for _ in range(2)]

    size_2dTensor[0] = xyz_lessMid_indx.size()[0]
    size_2dTensor[1] = xyz_largMid_indx.size()[0]
    FPS_2dList[0] = 1 if size_2dTensor[0] <= FPS_th else 0
    FPS_2dList[1] = 1 if size_2dTensor[1] <= FPS_th else 0


    xyz_2dList[0] = xyz[:, xyz_lessMid_indx[:,1], :]
    xyz_2dList[1] = xyz[:, xyz_largMid_indx[:,1], :]
    global_index[0] = ori_index[:, xyz_lessMid_indx[:,1]]
    global_index[1] = ori_index[:, xyz_largMid_indx[:,1]]


    return size_2dTensor, xyz_2dList, FPS_2dList, global_index


def process_branch(xyz, xyz_2dL_L1, size_2dL_L1, nsample, new_xyz, radius, FPS_2dL_L1, FPS_th, i, j, tree_depth, global_index, ori_index, executor=None, center_idx_all_batch_list=[], group_idx_all_batch_list=[]):
    if size_2dL_L1[j] != 0:
        xyz_L1 = xyz_2dL_L1[j]
        if FPS_2dL_L1[j] == 1:
            centerXYZ, centerXYZ_size, indices_center = selectAfromB_chunk(xyz_L1, new_xyz[i], chunk_size=1000)
            # centerXYZ, centerXYZ_size, indices_center = selectAfromB(xyz_L1, new_xyz[i])

            if(centerXYZ_size != 0):
                group_from_xyz = xyz if tree_depth > 1 else xyz_L1 
                group_from_xyz_N = xyz.size(1) if tree_depth > 1 else xyz_L1.size(1) 
                idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz_L1.device).zero_()
                pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)

                if tree_depth > 1:
                    global_index_group = torch.gather(global_index.unsqueeze(1).unsqueeze(1).expand(-1, idx.size(1), idx.size(2), -1), dim=-1, index=idx.long().unsqueeze(-1)).squeeze(-1)
                else:
                    global_index_group = torch.gather(ori_index.unsqueeze(1).unsqueeze(1).expand(-1, idx.size(1), idx.size(2), -1), dim=-1, index=idx.long().unsqueeze(-1)).squeeze(-1) #FIXME: maybe need to fix the ori_index
                # indx_batch = torch.cat((indx_batch, global_index), dim=1)
                return indices_center, global_index_group, center_idx_all_batch_list, group_idx_all_batch_list
            else:
                return None, None, center_idx_all_batch_list, group_idx_all_batch_list
        else:
            indices_center, global_index_group, new_center_idx_all_batch_list, new_group_idx_all_batch_list = Fractal_group_recursive_config(
                xyz_L1, nsample, new_xyz[i].unsqueeze(0), radius, FPS_th, tree_depth, ori_index, executor, center_idx_all_batch_list=center_idx_all_batch_list, group_idx_all_batch_list=group_idx_all_batch_list
            )
            return indices_center, global_index_group, new_center_idx_all_batch_list, new_group_idx_all_batch_list
    else:
        return None, None, center_idx_all_batch_list, group_idx_all_batch_list

def Fractal_group_recursive_config(xyz, nsample, new_xyz, radius, FPS_th, tree_depth=0, global_index=None, executor=None, index_global_group=None, index_global_center=None, center_idx_all_batch_list=[], group_idx_all_batch_list=[]):
    if executor is None:
        # Create a single ThreadPoolExecutor to be shared
        center_idx_all_batch_list = []
        group_idx_all_batch_list = []
        with ThreadPoolExecutor(max_workers=4096) as executor:
            return Fractal_group_recursive_config(xyz, nsample, new_xyz, radius, FPS_th, tree_depth, global_index, executor, index_global_group, index_global_center, center_idx_all_batch_list, group_idx_all_batch_list)

    if xyz.ndim == 3:
        B, N, _ = xyz.size()
    else:
        B = 1
        N = xyz.size(0)
        # xyz = xyz.unsqueeze(0)  # Add batch dimension if missing

    # Tree FPS: Partition and count
    if global_index is None:
        global_index = torch.arange(N).to(xyz.device).unsqueeze(0).repeat(B, 1)
    
    if index_global_center is None:
        index_global_center = torch.tensor([]).cuda()
        index_global_group = torch.tensor([]).cuda()

    for i in range(B):  # Loop over batches
        size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1, split_index = part2_and_count_with_index(xyz[i, :, :].unsqueeze(0), FPS_th, global_index[i:i+1], tree_depth)

        # Submit tasks to the shared executor
        futures = [executor.submit(process_branch, xyz[i, :, :].unsqueeze(0), xyz_2dL_L1, size_2dL_L1, nsample, new_xyz, radius, FPS_2dL_L1, FPS_th, i, j, tree_depth + 1, global_index[i:i+1], split_index[j], executor, [], []) for j in range(2)]
        index_center = [future.result()[0] for future in futures if future.result()[0] is not None]
        index_group = [future.result()[1] for future in futures if future.result()[1] is not None]
 
        # Concatenate the results from all branches
        if index_center != []:
            indices_center_batch = torch.cat(index_center, dim=0)
            indx_group_batch = torch.cat(index_group, dim=1)

            if indices_center_batch.shape[0] == new_xyz.shape[1]:
                center_idx_all_batch_list.append(indices_center_batch)
                group_idx_all_batch_list.append(indx_group_batch)
    if len(center_idx_all_batch_list) == new_xyz.shape[0]:
        center_idx_all_batch_list = torch.stack(center_idx_all_batch_list)
        group_idx_all_batch_list = torch.cat(group_idx_all_batch_list, dim=0)
        return indices_center_batch, indx_group_batch, center_idx_all_batch_list, group_idx_all_batch_list
    else:
        return indices_center_batch, indx_group_batch, center_idx_all_batch_list, group_idx_all_batch_list

_FRACTAL_GROUP_STAGES = None
_FRACTAL_GROUP_TH = 64

def set_fractal_group_config(stages=None, th=64):
    """set the fractal group config
    Args:
        stages: list of int or None, for example [1, 2] means using fractal group in stage 1 and 2, None means using original method
        th: int, the threshold of fractal group, default 64
    """
    global _FRACTAL_GROUP_STAGES, _FRACTAL_GROUP_TH
    _FRACTAL_GROUP_STAGES = stages
    _FRACTAL_GROUP_TH = th

def get_fractal_group_stages():
    return _FRACTAL_GROUP_STAGES

def get_fractal_group_th():
    return _FRACTAL_GROUP_TH

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, stage: int = None) -> torch.Tensor:
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
        # pointNext and pointvector is (1, 2, 3, 4)
        # pointNet++ is (0, 1) just two layer
        fractal_group_stages = get_fractal_group_stages()
        use_fractal_group = fractal_group_stages is not None and stage in fractal_group_stages
        # print(f'stage: {stage}, fractal_group_stages: {fractal_group_stages}, use_fractal_group: {use_fractal_group}')

        if use_fractal_group:
            Fractal_group_th = get_fractal_group_th()
            _, _, indices_global, indx_global = Fractal_group_recursive_config(xyz, nsample, new_xyz, radius, Fractal_group_th)
            sorted_indices = torch.argsort(indices_global, dim=1)
            sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, indx_global.size(2))
            xyz_out_recursive = torch.gather(indx_global, dim=1, index=sorted_indices_expanded)
            idx=xyz_out_recursive.int()

        else:
            idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
            pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, stage: int, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.stage = stage
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # torch.cuda.synchronize()
        # timeg0 = time.time()
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz, self.stage)
        # torch.cuda.synchronize()
        # timeg1 = time.time()
        # print(f"Group time: {timeg1 - timeg0}")

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj





def create_grouper(stage, group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)
    # pdb.set_trace()
    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(stage, radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper


if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    # --------------- debug downsampling
    from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample

    npoints = 10000
    # rs = RandomSample(num_to_sample=npoints)
    # query, _= rs(points)
    idx = random_sample(points, npoints)
    # torch gather is faster then operation gather. 
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    # ------------- debug ball query
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        # ball querying is 40 times faster then KNN 
        features = query_group(query, points)
    print(time.time() - st)
    # print(features.shape)
