import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
import numpy as np


class GaussianModel:

    def __init__(self):
        self.max_sh_degree = 3

        self.xyz = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.features_dc = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.features_rest = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.scaling = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.rotation = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.opacity = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))

    def forward(self):
        return {
            'xyz': self.xyz,
            'features_dc': self.features_dc,
            'features_rest': self.features_rest,
            'scaling': self.scaling,
            'rotation': self.rotation,
            'opacity': self.opacity
        }

    def load_ply(self, path):
        try:
            plydata = PlyData.read(path)
        except Exception as e:
            print(f"Error reading PLY file: {e}")
            return

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc(0，1，2)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        print('features_dc shape:', features_dc.shape)

        # features_rest( ... ... )
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # scale
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # rotation
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        print('Number of points from ply:', xyz.shape[0])

        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def get_feature_vector(self, t=1.0):

        num_points = self.xyz.shape[0]
        t = torch.ones((num_points, 1), device='cuda') * t

        # Collect and view features as needed
        xyz = self.xyz
        features_dc = self.features_dc.view(num_points, -1)
        features_rest = self.features_rest.view(num_points, -1)
        scaling = self.scaling
        rotation = self.rotation

        # Combine all features into a single tensor
        fv = torch.cat([xyz, features_dc, features_rest, scaling, rotation, t], dim=1)

        return fv
