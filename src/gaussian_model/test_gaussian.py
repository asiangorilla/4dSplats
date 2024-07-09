import torch
import numpy as np
from gaussian_model import GaussianModel

sh_degree = 3
args = {}


# Create form Point Cloud
class BasicPointCloud:
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors

# 示例点云数据
points = np.random.rand(10, 3)  # 10 个点的坐标
colors = np.random.rand(10, 3)  # 10 个点的颜色

pcd = BasicPointCloud(points, colors)
spatial_lr_scale = 1.0
time_line = 0

model1 = GaussianModel(sh_degree, args)
model1.create_from_pcd(pcd, spatial_lr_scale, time_line)
print('Create from Point Cloud --- done')



ply_file_path = 'myproj/splat.ply'
model2 = GaussianModel(sh_degree, args)
model2.load_ply(ply_file_path)
print('Load from PLY --- done')





