'''
Transform data formate
return dic Camera(
    colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
    image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
    mask=mask)
gt_alpha_mask=None --> opacity matrix
'''

from torch.utils.data import Dataset
from scene.cameras import Camera
from scene.scene_read import focal2fov

import torch


class FourDGSdataset(Dataset):

    def __init__(
            self,
            dataset,
            args,
            dataset_type
    ):
    ''' 传进来的 dataset 是一个相机列表'''
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type

    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R, T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask = None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time

                mask = caminfo.mask

            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                          mask=mask)
        else:
            return self.dataset[index]

    def __len__(self):

        return len(self.dataset)