# combine all elements

import torch
from LossFunction.DeformationLoss import DeformationLoss
import LossFunction.utils
from TransformationModel.transModel import DeformationNetworkConnected, DeformationNetworkSeparate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss = DeformationLoss()
model_connected = DeformationNetworkConnected()
model_separate = DeformationNetworkSeparate()