import numpy as np
from PIL import Image
import torch
import os
from TransformationModel.transModel import (DeformationNetworkSeparate, DeformationNetworkBilinearCombination,
                                            DeformationNetworkCompletelyConnected)


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


ply_path = os.path.join('..', 'gaussian_ply_files', 'sample_video', 'frame_1', 'splat.ply')
points_tensor, colors_tensor, opacities_tensor, scales_tensor, rotations_tensor = load_ply_data(ply_path, device)

# timestamp and ground truth image
t = 1
image_path = 'gt_image.png'  # !!!!!  here! replace by path of 1st frame ground truth image !!!!!!!
gt_image_tensor, w, h = extract_image_data(image_path)
num_channels = gt_image_tensor.shape[-1]



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
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]

    # Randomly generate rendered_rgb and rendered_alphas for testing
    # rendered_rgb = torch.rand((1, h, w, 3), device=device, requires_grad=True)
    # rendered_alphas = torch.rand((1, h, w, 1), device=device, requires_grad=True)
    rendered_rgb, rendered_alphas, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, w, h)

    rendered_rgb = rendered_rgb.squeeze(0)
    rendered_alphas = rendered_alphas.squeeze(0)
    if num_channels == 4 :
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







