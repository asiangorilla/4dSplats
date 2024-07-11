from plyfile import PlyData, PlyElement
from scene.gaussian_utils import BasicPointCloud, RGB2SH,inverse_sigmoid,mkdir_p

import os
import torch
import torch.nn as nn
import numpy as np
from simple_knn._C import distCUDA2

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
        print('Load Gaussian Points with ply... ...')
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

    def save_ply(self, path):
        ''' Save the model data into a ply file '''

        mkdir_p(os.path.dirname(path))

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self.features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        # f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation),
                                    axis=1)  # -> ply file content
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        '''  --> Build a list of attributes '''
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1]*self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1]*self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        # Get PointCloud data
        # Creating the internal state of a model from PointCloud data

        # print('reading spatial_lr_scale:')
        # self.spatial_lr_scale = spatial_lr_scale
        print('Creating Gaussian Points with PointCloud... ...')

        # get points and color
        print('Reading points:')
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        print('Num of points:', fused_point_cloud.shape[0])

        print('Reading fused_color:')
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())   # RGB2SH: --> Converts colour data from RGB to spherical harmonic (SH) representation
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # --> Reserve space for features according to max_sh_degree
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print('Shape of features:', features.shape)

        print('Initial scaling')
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        print('Scaling shape:', scales.shape)

        # initialize rotation quaternion (1,0,0,0)
        print('Initial rotation')
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        print('Rotation shape:', rots.shape)

        print('Initial opacities')
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        print('Opacities shape:', opacities.shape)

        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.features_dc = nn.Parameter(features[:, :, 0:1].contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:, :, 1:].contiguous().requires_grad_(True))
        # self.features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self.features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scaling = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))

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
