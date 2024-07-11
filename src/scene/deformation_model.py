import torch
from torch import nn
import pytorch_lightning as pl


class TimeAwareEncoder(nn.Module):

    def __init__(self, input_dim, depth, width):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, width))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BaseDecoder(nn.Module):
    def __init__(self, output_dim, width):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class XYZDecoder(BaseDecoder):
    def __init__(self, output_dim, width):
        super().__init__(output_dim, width)


class ColorDecoder(BaseDecoder):
    def __init__(self, output_dim, width):
        super().__init__(output_dim, width)


class ScaleDecoder(BaseDecoder):
    def __init__(self, output_dim, width):
        super().__init__(output_dim, width)


class RotationDecoder(BaseDecoder):
    def __init__(self, output_dim, width):
        super().__init__(output_dim, width)


# class OpacityDecoder(BaseDecoder):
#     def __init__(self, feature_dim, output_dim):
#         super().__init__(feature_dim, output_dim)


class DeformationModel(pl.LightningModule):

    def __init__(self, input_dim, depth, common_width):
        super().__init__()
        self.encoder = TimeAwareEncoder(input_dim, depth, common_width)
        self.xyz_decoder = XYZDecoder(3, common_width)
        self.color_decoder = ColorDecoder(48, common_width)
        self.scale_decoder = ScaleDecoder(3, common_width)
        self.rotation_decoder = RotationDecoder(4, common_width)

    def forward(self, data):
        encoded = self.encoder(data)
        xyz = self.xyz_decoder(encoded)
        color = self.color_decoder(encoded)
        scale = self.scale_decoder(encoded)
        rotation = self.rotation_decoder(encoded)

        output = {
            'xyz': xyz,
            'color': color,
            'scale': scale,
            'rotation': rotation
        }
        return output


    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = sum(torch.nn.functional.mse_loss(pred, target) for pred, target in zip(pred, y))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

