import numpy as np
import torch
import torch.nn.functional as F

from TransformationModel.transModel import DeformationNetworkCompletelyConnected
from gsplat.rendering import rasterization
from LossFunction.utils import calc_ssim


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = DeformationNetworkCompletelyConnected().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# randomly generate original parameters
x = np.random.randn(100, 3)
q = np.random.randn(100, 4)
t = np.arange(1, 101)
x_tensor = torch.tensor(x, dtype=torch.float).to(device)
q_tensor = torch.tensor(q, dtype=torch.float).to(device)
t_tensor = torch.tensor(t, dtype=torch.float).to(device)

# randomly generate target data
target_colors = torch.rand((1, 3, 200, 300), device=device)
target_alphas = torch.rand((1, 1, 200, 300), device=device)


output_x, output_q_normalized, output_q = model(x_tensor, q_tensor, t_tensor)
means = output_x + x_tensor
quats = output_q_normalized + q_tensor

# rendering process
scales = torch.rand((100, 3), device=device) * 0.1
colors = torch.rand((100, 3), device=device)
opacities = torch.rand((100,), device=device)
viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor([
    [300., 0., 150.],
    [0., 300., 100.],
    [0., 0., 1.]
], device=device)[None, :, :]
colors, alphas, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 300, 200)
print(colors.shape, alphas.shape)
print(meta)

# adjust dimension
rendered_colors = colors.permute(0, 3, 1, 2)  # [B, C, H, W]
rendered_alphas = alphas.permute(0, 3, 1, 2)

# L1 loss
l1_color_loss = F.l1_loss(rendered_colors, target_colors)
l1_alpha_loss = F.l1_loss(rendered_alphas, target_alphas)
print("l1_color_loss：", l1_color_loss)
print("l1_alpha_loss：", l1_alpha_loss)
print("l1 loss -- done")

# SSIM loss
ssim_color_loss = 1.0 - calc_ssim(rendered_colors, target_colors)
ssim_alpha_loss = 1.0 - calc_ssim(rendered_alphas, target_alphas)
print("ssim_color_loss:", ssim_color_loss)
print("ssim_alpha_loss:", ssim_alpha_loss)
print("ssim loss -- done")

# Total loss
total_loss = l1_color_loss + l1_alpha_loss + ssim_color_loss + ssim_alpha_loss

if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
    print("Total loss contains NaN or Inf!")
else:
    print("Total loss is valid.")

print("Total loss before backward:", total_loss)






