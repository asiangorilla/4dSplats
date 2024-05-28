"""
The losses calculated are:
1. Image Loss (im): Combines L1 loss and SSIM loss for the rendered image.
2. Segmentation Loss (seg): Combines L1 loss and SSIM loss for the segmentation map.
3. Rigid Loss (rigid): Measures the deviation in positions between neighboring points.
4. Rotation Loss (rot): Measures the consistency in rotations between neighboring points.
5. Isometric Loss (iso): Ensures distances between neighboring points remain consistent.
6. Floor Constraint Loss (floor): Ensures points do not go below a certain plane (y=0).
7. Background Points Position and Rotation Loss (bg): Ensures stability of background points' positions and rotations.
8. Soft Color Consistency Loss (soft_col_cons): Ensures color consistency over time.

The forward method compute  these losses, and sums them up using specified weights.
"""


import torch.nn.functional as F
import torch.nn as nn
import torch

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils import params2rendervar, calc_ssim, quat_mult, build_rotation,l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2
from math import exp

class DeformationLoss(nn.Module):
    def __init__(self):
        super(DeformationLoss, self).__init__()

    def forward(self, params, curr_data, variables, is_initial_timestep):
        losses = {}

        # Transform parameters into renderable variable, setting gradient computation
        rendervar = self.get_rendervar(params)

        # Compute rendered image, adjusts the brightness and contrast.
        im, radius = self.get_rendered_image(rendervar, curr_data)

        # Comput L1-Loss and SSIM-Loss
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        # Saving intermediate results
        variables['means2D'] = rendervar['means2D'] 

        # Segmentation-Loss
        seg = self.get_segmentation_image(params, curr_data)
        losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))
        
        # compute other loss
        if not is_initial_timestep:
            
            # Compute foreground parameters
            curr_offset_in_prev_coord, rel_rot, curr_offset, fg_pts, is_fg = self.get_foreground_params(params, rendervar, variables)

            # Rigid-Loss
            losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                                variables["neighbor_weight"])
            # Rotation-Loss
            losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                                variables["neighbor_weight"])


            curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
            # Isometric-Loss
            losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

            # Floor-Constraint-Loss
            losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

            bg_pts = rendervar['means3D'][~is_fg]
            bg_rot = rendervar['rotations'][~is_fg]
            # Background Points Position and Rotation loss
            losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
            # Soft Color Consistency-Loss
            losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])


        # compute Sum of weighted loss
        loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                        'soft_col_cons': 0.01}
        loss = sum([loss_weights[k] * v for k, v in losses.items()]) # FINAL LOSS

        # compute seen points
        seen = radius > 0
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
        variables['seen'] = seen

        return loss, variables
    
    def get_rendervar(self, params):
        rendervar = params2rendervar(params)
        rendervar['means2D'].retain_grad()

        return rendervar
    
    def get_rendered_image(self, rendervar, curr_data):
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        return im, radius
    
    def get_segmentation_image(self, params, curr_data):
        segrendervar = self.get_rendervar(params)
        segrendervar['colors_precomp'] = params['seg_colors']
        seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
        return seg
    
    def get_foreground_params(params, rendervar, variables):
        
        # Extraction foreground
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        # Compute rotation matrix for foreground
        fg_rot = rendervar['rotations'][is_fg]
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        # Compute offset for foreground
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

        return curr_offset_in_prev_coord, rel_rot, curr_offset, fg_pts, is_fg

    

    

    
    




