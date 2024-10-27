# import torch
# import pdb
# support_xyz = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/support_xyz.pth') # ([64, 1024, 3])
# query_xyz = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/query_xyz.pth')   # ([64, 512, 3]) FPS之后的xyz
# idx = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/idx.pth')  # ([64, 512, 32] group操作之后的idx)

# xyz_trans    = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/xyz_trans.pth') # ([64, 3, 1024]
# grouped_xyz = torch.load('./data/PointNeXt-S/firstlayer_grouping_in_out/grouped_xyz.pth') # ([64, 3, 512, 32]) group之后的xyz坐标

# print(xyz_trans.size())
# print(query_xyz.size())
# print(grouped_xyz.size())
# print(support_xyz.size())
# print(idx.size())

# print(support_xyz[0, idx[0, 0, 0], :].tolist())
# print(query_xyz[0, 0, :].tolist())
# print(grouped_xyz[0, :, 0, 0].tolist())

# pdb.set_trace() 
# save the printed result into txt file
          
# with open('./test/output.txt', 'w') as f:
#     for i in range(512):
#         f.write(f'---i={i}\n')
#         f.write(f'{support_xyz[0, idx[0, i, 0], :]}\n')
#         f.write(f'{query_xyz[0, i, :]}\n')
#         f.write(f'{grouped_xyz[0, :, i, 0]}\n')

import numpy as np

# Example tensors
A = np.array([[[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]])  # Shape: [1, 3, 3]
B = np.array([2, 1, 3])     # Shape: [3]

# Step 1: Get the sorted indices of B
sorted_indices = torch.argsort(B)

# Step 2: Reorder A using these indices along the second dimension
A_reordered = A[:, sorted_indices, :]
print(sorted_indices)
print("Reordered A:\n", A_reordered)
