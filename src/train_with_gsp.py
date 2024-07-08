import numpy as np
from gsplat.rendering import  rasterization
import torch
import os
import torch.nn.functional as F
from TransformationModel.transModel import (DeformationNetworkSeparate, DeformationNetworkBilinearCombination,
                                            DeformationNetworkCompletelyConnected)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_ply_data(ply_path, device='cpu'):
    from plyfile import PlyData
    # read .ply file
    plydata = PlyData.read(ply_path)

    # Extracting point cloud data
    vertex_data = plydata['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.vstack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T
    opacities = vertex_data['opacity']
    scales = np.vstack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T
    rotations = np.vstack([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T

    # Converting Data to PyTorch Tensor
    points_tensor = torch.tensor(points, dtype=torch.float32, requires_grad=True).to(device)
    colors_tensor = torch.tensor(colors, dtype=torch.float32, requires_grad=True).to(device)
    opacities_tensor = torch.tensor(opacities, dtype=torch.float32, requires_grad=True).to(device)
    scales_tensor = torch.tensor(scales, dtype=torch.float32, requires_grad=True).to(device)
    rotations_tensor = torch.tensor(rotations, dtype=torch.float32, requires_grad=True).to(device)

    return points_tensor, colors_tensor, opacities_tensor, scales_tensor, rotations_tensor

# read ply file
ply_path = os.path.join('..', 'gaussian_ply_files', 'sample_video', 'frame_1', 'splat.ply')
points_tensor, colors_tensor, opacities_tensor, scales_tensor, rotations_tensor = load_ply_data(ply_path, device)

# initialize model and optimizer
model = DeformationNetworkSeparate().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# camera config and render size
viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
w, h = 300, 200

# render ground truth gaussian
rendered_rgb_gt, rendered_alphas_gt, meta_gt = rasterization(points_tensor, rotations_tensor, scales_tensor, opacities_tensor, colors_tensor, viewmats, Ks, w, h)

# Training loop
epochs = 1
for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()

    t = 1
    output_x, output_q = model(points_tensor.detach(), rotations_tensor.detach(), t)

    # Update means and quats
    means = points_tensor + output_x
    quats = rotations_tensor + output_q
    scales = scales_tensor
    colors = colors_tensor
    opacities = opacities_tensor

    # Randomly generate rendered_rgb and rendered_alphas for testing
    # rendered_rgb = torch.rand((1, h, w, 3), device=device, requires_grad=True)
    # rendered_alphas = torch.rand((1, h, w, 1), device=device, requires_grad=True)
    rendered_rgb, rendered_alphas, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, w, h)

    keys = ['means2d', 'opacities', 'radii']
    losses = [F.l1_loss(meta_gt[key], meta[key], reduction='none') for key in keys if key in meta_gt and key in meta]
    losses = torch.stack(losses)
    total_loss = torch.mean(losses)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item()}')