import numpy as np
from torch import nn
import torch
'''
deformation Network model classes.
Both deformation Networks are based upon nerf MLP architectures. Separate Model is comprised of two separate MLP's.
Connected model is one MLP. Both Networks take position and quaternion together with a time t. Netoworks calulate
the deformation or change delta_x and delta_q s.t. the original positions have to be added. i.e. x(t) = x(0) + delta_x
and q(t) = delta_q * q(0), as per coordinate and quaternion arithmetic. 
It is assumed that both position and quaternions are normalized. Normalization of delta_q is built in. 
TODO, check normalization for delta_x.
'''
device = 'cuda:0'
coordinate_L = 10  #as per original nerf paper
quaternion_L = 15  #no paper exists for this value, trial and error
pos_dim = 3
quat_dim = 4


class DeformationNetworkSeparate(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_x_1 = torch.nn.Linear(coordinate_L * 2 * pos_dim + 1, 256, device=device)
        self.linear_x_2 = torch.nn.Linear(256, 256, device=device)
        self.linear_x_3 = torch.nn.Linear(256, 256, device=device)
        self.linear_x_4 = torch.nn.Linear(256, 256, device=device)
        self.linear_x_5 = torch.nn.Linear(256, 256, device=device)
        self.linear_x_6 = torch.nn.Linear(256, 256, device=device)
        self.linear_x_7 = torch.nn.Linear(256, 128, device=device)
        self.linear_x_8 = torch.nn.Linear(256, 128, device=device)
        self.linear_x_9 = torch.nn.Linear(128, 3, device=device)
        self.linear_q_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + 1, 256, device=device)
        self.linear_q_2 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_3 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_4 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_5 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_6 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_7 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_8 = torch.nn.Linear(256, 128, device=device)
        self.linear_q_9 = torch.nn.Linear(128, 4, device=device)
        self.relu_x_1 = torch.nn.ReLU()
        self.relu_x_2 = torch.nn.ReLU()
        self.relu_x_3 = torch.nn.ReLU()
        self.relu_x_4 = torch.nn.ReLU()
        self.relu_x_5 = torch.nn.ReLU()
        self.relu_x_6 = torch.nn.ReLU()
        self.relu_x_7 = torch.nn.ReLU()
        self.relu_x_8 = torch.nn.ReLU()
        self.relu_q_1 = torch.nn.ReLU()
        self.relu_q_2 = torch.nn.ReLU()
        self.relu_q_3 = torch.nn.ReLU()
        self.relu_q_4 = torch.nn.ReLU()
        self.relu_q_5 = torch.nn.ReLU()
        self.relu_q_6 = torch.nn.ReLU()
        self.relu_q_7 = torch.nn.ReLU()
        self.relu_q_8 = torch.nn.ReLU()

    def forward(self, x, q, t):
        if t == 0:
            return torch.zeros(pos_dim), torch.zeros(quat_dim)
        t = torch.tensor([t], dtype=torch.float).to(device)
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_x = torch.tensor(higher_x.flatten(), dtype=torch.float).to(device)
        input_x = torch.cat((higher_x.clone().detach(), t.clone().detach()), 0)
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
        input_x = self.relu_x_8(input_x)
        input_x = self.linear_x_9(input_x)


        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_q = torch.tensor(higher_q.flatten(), dtype=torch.float).to(device)
        input_q = torch.cat((higher_q.clone().detach(), t.clone().detach()), 0)
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
        input_q = self.relu_q_8(input_q)
        input_q = self.linear_q_9(input_q)

        #normalization of q
        input_q = normalize_q_torch(input_q)

        return input_x, input_q


class DeformationNetworkConnected(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + coordinate_L * 2 * pos_dim + 1, 256,
                                        device=device)
        self.linear_2 = torch.nn.Linear(256, 256, device=device)
        self.linear_3 = torch.nn.Linear(256, 256, device=device)
        self.linear_4 = torch.nn.Linear(256, 256, device=device)
        self.linear_5 = torch.nn.Linear(256, 256, device=device)
        self.linear_6 = torch.nn.Linear(256, 256, device=device)
        self.linear_7 = torch.nn.Linear(256, 256, device=device)
        self.linear_8 = torch.nn.Linear(256, 128, device=device)
        self.linear_9 = torch.nn.Linear( 128, pos_dim + quat_dim, device=device)
        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        self.relu_3 = torch.nn.ReLU()
        self.relu_4 = torch.nn.ReLU()
        self.relu_5 = torch.nn.ReLU()
        self.relu_6 = torch.nn.ReLU()
        self.relu_7 = torch.nn.ReLU()
        self.relu_8 = torch.nn.ReLU()

    def forward(self, x, q, t):
        if t == 0:
            return torch.zeros(pos_dim), torch.zeros(quat_dim)
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_x = torch.tensor(higher_x.flatten(), dtype=torch.float).to(device)
        higher_q = torch.tensor(higher_q.flatten(), dtype=torch.float).to(device)
        t = torch.tensor([t], dtype=torch.float).to(device)
        input_total = torch.cat((higher_x.clone().detach(), higher_q.clone().detach(), t.clone().detach()), 0)
        input_total = self.linear_1(input_total)
        input_total = self.relu_1(input_total)
        input_total = self.linear_2(input_total)
        input_total = self.relu_2(input_total)
        input_total = self.linear_3(input_total)
        input_total = self.relu_3(input_total)
        input_total = self.linear_4(input_total)
        input_total = self.relu_4(input_total)
        input_total = self.linear_5(input_total)
        input_total = self.relu_5(input_total)
        input_total = self.linear_6(input_total)
        input_total = self.relu_6(input_total)
        input_total = self.linear_7(input_total)
        input_total = self.relu_7(input_total)
        input_total = self.linear_8(input_total)
        input_total = self.relu_8(input_total)
        input_total = self.linear_9(input_total)
        out_x, out_q1, out_q2 = torch.split(input_total, pos_dim)
        out_q = torch.cat((out_q1, out_q2))
        out_q = normalize_q_torch(out_q)
        return out_x, out_q


def higher_dim_gamma(p, length_ar):  #as per original nerf paper
    sins = np.sin(np.outer(2 ** np.array(list(range(length_ar))) * np.pi, p))
    coss = np.cos(np.outer(2 ** np.array(list(range(length_ar))) * np.pi, p))
    return np.column_stack((sins, coss))


def normalize_q_torch(q):  #normalization of quaternion. Only unit quaternions can be interpreted as rotations.
    norm_q = torch.sum(q ** 2)
    norm_q = torch.sqrt(norm_q)
    return q / norm_q
