import numpy as np
from torch import nn
import torch

device = 'cuda:0'
coordinate_L = 10
quaternion_L = 15
pos_dim = 3
quat_dim = 4


def build_model():
    model = nn.Sequential(

    )
    return 0


class DeformationNetworkSeperate(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_x_1 = torch.Linear(coordinate_L * 2 * pos_dim + 1, 256)
        self.linear_x_2 = torch.nn.Linear(256, 256)
        self.linear_x_3 = torch.nn.Linear(256, 256)
        self.linear_x_4 = torch.nn.Linear(256, 256)
        self.linear_x_5 = torch.nn.Linear(256, 256)
        self.linear_x_6 = torch.nn.Linear(256, 256)
        self.linear_x_7 = torch.nn.Linear(256, 256)
        self.linear_x_8 = torch.nn.Linear(256, 3)
        self.linear_q_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + 1, 256)
        self.linear_q_2 = torch.nn.Linear(256, 256)
        self.linear_q_3 = torch.nn.Linear(256, 256)
        self.linear_q_4 = torch.nn.Linear(256, 256)
        self.linear_q_5 = torch.nn.Linear(256, 256)
        self.linear_q_6 = torch.nn.Linear(256, 256)
        self.linear_q_7 = torch.nn.Linear(256, 256)
        self.linear_q_8 = torch.nn.Linear(256, 4)
        self.relu_x_1 = torch.nn.ReLU()
        self.relu_x_2 = torch.nn.ReLU()
        self.relu_x_3 = torch.nn.ReLU()
        self.relu_x_4 = torch.nn.ReLU()
        self.relu_x_5 = torch.nn.ReLU()
        self.relu_x_6 = torch.nn.ReLU()
        self.relu_x_7 = torch.nn.ReLU()
        self.relu_q_1 = torch.nn.ReLU()
        self.relu_q_2 = torch.nn.ReLU()
        self.relu_q_3 = torch.nn.ReLU()
        self.relu_q_4 = torch.nn.ReLU()
        self.relu_q_5 = torch.nn.ReLU()
        self.relu_q_6 = torch.nn.ReLU()
        self.relu_q_7 = torch.nn.ReLU()

    def forward(self, x, q, t):
        if t == 0:
            return np.zeros(pos_dim + quat_dim)
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_x = torch.tensor(higher_x).to(device)
        input_x = torch.cat((higher_x, t), 0)
        input_x = self.linear_x_1(input_x)
        input_x = self.relu_x_1(input_x)
        input_x = self.linear_x_2(input_x)
        input_x = self.relu_x_2(input_x)
        input_x = self.linear_x_3(input_x)
        input_x = self.relu_x_3(input_x)
        input_x = self.linear_x_4(input_x)
        input_x = self.relu_x_4(input_x)
        input_x = self.linear_x_5(input_x)
        input_x = self.relu_x_5(input_x)
        input_x = self.linear_x_6(input_x)
        input_x = self.relu_x_6(input_x)
        input_x = self.linear_x_7(input_x)
        input_x = self.relu_x_7(input_x)
        input_x = self.linear_x_8(input_x)

        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_q = torch.tensor(higher_q).to(device)
        input_q = torch.cat((higher_q, t), 0)
        input_q = torch.cat((x, t), 0)
        input_q = self.linear_q_1(input_q)
        input_q = self.relu_q_1(input_q)
        input_q = self.linear_q_2(input_q)
        input_q = self.relu_q_2(input_q)
        input_q = self.linear_q_3(input_q)
        input_q = self.relu_q_3(input_q)
        input_q = self.linear_q_4(input_q)
        input_q = self.relu_q_4(input_q)
        input_q = self.linear_q_5(input_q)
        input_q = self.relu_q_5(input_q)
        input_q = self.linear_q_6(input_q)
        input_q = self.relu_q_6(input_q)
        input_q = self.linear_q_7(input_q)
        input_q = self.relu_q_7(input_q)
        input_q = self.linear_q_8(input_q)

        #normalization of q
        input_q = normalize_q(input_q)

        return input_x, input_q


def higher_dim_gamma(p, length_ar):
    sins = np.sin(np.outer(2 ** np.array(list(range(length_ar))) * np.pi, p))
    coss = np.cos(np.outer(2 ** np.array(list(range(length_ar))) * np.pi, p))
    return np.column_stack((sins, coss))


def normalize_q(q):
    norm_q = torch.sum(q ** 2)
    norm_q = torch.sqrt(norm_q)
    return q / norm_q
