import numpy as np
from PIL import Image
import torch
import os
from TransformationModel.transModel import (DeformationNetworkSeparate, DeformationNetworkBilinearCombination,
                                            DeformationNetworkCompletelyConnected)
from gsplat.rendering import rasterization
import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def extract_image_data(image_path):
    image = Image.open(image_path)
    width, height = image.size
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        rgb_data = image_np[:, :, :3]
        alpha_data = image_np[:, :, 3]
    else:
        rgb_data = image_np
        alpha_data = None

    if alpha_data is not None:
        gt_image = np.dstack((rgb_data, alpha_data))
    else:
        gt_image = rgb_data

    gt_image_tensor = torch.tensor(gt_image, dtype=torch.float32).to(device)

    return gt_image_tensor, width, height


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

ply_path = os.path.join('..', 'gaussian_splats', 'gaussians', 'frame_00002', 'splat.ply')
points_tensor, colors_tensor, opacities_tensor, scales_tensor, rotations_tensor = load_ply_data(ply_path, device)

# timestamp and ground truth image
t = 1
# !!!!!  here! replace by path of 1st frame ground truth image !!!!!!!
'''
for testing, we are using the second frame from video 0
small code block just to get correct viewing mat, width and height from the JSON files from colmap for frame 1, camera from 0
'''
image_path = os.path.join('..', 'gaussian_splats', 'ordered_output', 'vid_frame00002', 'frame_from_0.png')
gt_image_tensor = extract_image_data(image_path)[0]
'''
start of code block fror JSON file extraction
'''
json_path = os.path.join('..', 'gaussian_splats', 'colmap_output', 'colmap_00002', 'transforms.json')
f = open(json_path)
num_channels = gt_image_tensor.shape[-1]

data = json.load(f)
f.close()
del f
w = data['w']
h = data['h']
Ks = torch.Tensor([[[data['fl_x'], 0, data['cx']], [0, data['fl_y'], data['cy']], [0, 0, 1]]]).to(device)
viewmats = torch.Tensor([data['frames'][0]['transform_matrix']]).to(device)
'''
end of codeblock
'''

model = DeformationNetworkSeparate().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

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

    print('tiles_per_gauss shape  is: ', meta['tiles_per_gauss'].shape)
    print('isect_ids shape  is: ', meta['isect_ids'].shape)
    print('flatten_ids shape  is: ', meta['flatten_ids'].shape)
    print('isect_offsets shape  is: ', meta['isect_offsets'].shape)

    rendered_rgb = rendered_rgb.squeeze(0)
    rendered_alphas = rendered_alphas.squeeze(0)
    if num_channels == 4:
        # image with alpha channel
        rendered_image = torch.cat((rendered_rgb, rendered_alphas), dim=-1)
    else:
        rendered_image = rendered_rgb.squeeze(0)

    # Ensure shapes match
    assert gt_image_tensor.shape == rendered_rgb.shape, "Shapes of ground truth image and rendered image do not match"

    l1_loss = torch.mean(torch.abs(gt_image_tensor - rendered_image))

    l1_loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {l1_loss.item()}')
