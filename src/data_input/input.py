from plyfile import PlyData
import numpy as np
import torch
from src.TransformationModel.transModel import (DeformationNetworkSeparate, DeformationNetworkBilinearCombination,
                                                DeformationNetworkCompletelyConnected)

# read .ply file
ply_path = 'F:\\Files\\Project\\point_cloud.ply'
plydata = PlyData.read(ply_path)

# Extracting point cloud data
vertex_data = plydata['vertex'].data
points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
colors = np.vstack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T
opacities = vertex_data['opacity']
scales = np.vstack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T
rotations = np.vstack([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T

# Converting Data to PyTorch Tensor
points_tensor = torch.tensor(points, dtype=torch.float32).cuda()
colors_tensor = torch.tensor(colors, dtype=torch.float32).cuda()
opacities_tensor = torch.tensor(opacities, dtype=torch.float32).cuda()
scales_tensor = torch.tensor(scales, dtype=torch.float32).cuda()
rotations_tensor = torch.tensor(rotations, dtype=torch.float32).cuda()

# Model Instantiation
model = DeformationNetworkSeparate()  # or DeformationNetworkConnected()
model = model.cuda()

# Setting of time t
t = 0

# Passing data to the transmodel
output_x, output_q = model(points_tensor, rotations_tensor, t)

# output
print("Output positions:", output_x)
print("Output quaternions:", output_q)

