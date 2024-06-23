import numpy as np
from torch import nn
import torch

"""
    deformation Network model classes.
    Both deformation Networks are based upon nerf MLP architectures. Separate Model is comprised of two separate MLP's.
    Connected model is one MLP. Both Networks take position and quaternion together with a time t. Networks calculate
    the deformation or change delta_x and delta_q s.t. the original positions have to be added. i.e. x(t) = x(0) + delta_x(t)
    and q(t) = delta_q(t) * q(0), as per coordinate and quaternion arithmetic. model calculates delta_x(t) and delta_q(t)
    It is assumed that both position and quaternions are normalized. Normalization of delta_q is built in. 
    TODO, check normalization for delta_x.
    
    since derivative for quaternion is quite simple (dq/dt = 1/2 w q), a simple second deformation network should be 
    sufficient to learn quaternion deformation. rotation does not necessarily depend on location, 
"""

# device = 'cuda:0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

        self.relu = torch.nn.ReLU

    def forward(self, x, q, t):

        if t == 0:
            # return torch.zeros(pos_dim), torch.zeros(quat_dim)
            return torch.zeros(pos_dim).to(device), torch.zeros(quat_dim).to(device)
        
        t = torch.tensor(t, dtype=torch.float).to(device)
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_x = torch.tensor(higher_x.flatten(), dtype=torch.float).to(device)
        input_x = torch.cat((higher_x.clone().detach(), t.clone().detach()), 0)
        input_x = self.linear_x_1(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_2(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_3(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_4(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_5(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_6(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_7(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_8(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_9(input_x)

        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_q = torch.tensor(higher_q.flatten(), dtype=torch.float).to(device)
        input_q = torch.cat((higher_q.clone().detach(), t.clone().detach()), 0)
        input_q = self.linear_q_1(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_2(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_3(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_4(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_5(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_6(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_7(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_8(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_9(input_q)

        #normalization of q
        input_q_2 = nn.functional.normalize(input_q, dim=0)

        return input_x, input_q_2, input_q


class DeformationNetworkCompletelyConnected(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + coordinate_L * 2 * pos_dim + 1, 512,
                                        device=device)
        self.linear_2 = torch.nn.Linear(512, 512, device=device)
        self.linear_3 = torch.nn.Linear(512, 512, device=device)
        self.linear_4 = torch.nn.Linear(512, 512, device=device)
        self.linear_5 = torch.nn.Linear(512, 512, device=device)
        self.linear_6 = torch.nn.Linear(512, 512, device=device)
        self.linear_7 = torch.nn.Linear(512, 512, device=device)
        self.linear_8 = torch.nn.Linear(512, 128, device=device)
        self.linear_9 = torch.nn.Linear(128, pos_dim + quat_dim, device=device)
        self.relu = torch.nn.ReLU()

    def forward(self, x, q, t):

        if t == 0:
            # return torch.zeros(pos_dim), torch.zeros(quat_dim)
            return torch.zeros(pos_dim).to(device), torch.zeros(quat_dim).to(device)
        
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_x = torch.tensor(higher_x.flatten(), dtype=torch.float).to(device)
        higher_q = torch.tensor(higher_q.flatten(), dtype=torch.float).to(device)
        t = torch.tensor([t], dtype=torch.float).to(device)
        input_total = torch.cat((higher_x.clone().detach(), higher_q.clone().detach(), t.clone().detach()), 0)
        input_total = self.linear_1(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_2(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_3(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_4(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_5(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_6(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_7(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_8(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_9(input_total)
        out_x, out_q1, out_q2 = torch.split(input_total, pos_dim)
        out_q = torch.cat((out_q1, out_q2))
        out_q_2 = nn.functional.normalize(out_q, dim=0)

        return out_x, out_q_2, out_q


class DeformationNetworkBilinearCombination(torch.nn.Module):

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

        self.linear_q_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + 1, 256, device=device)
        self.linear_q_2 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_3 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_4 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_5 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_6 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_7 = torch.nn.Linear(256, 256, device=device)
        self.linear_q_8 = torch.nn.Linear(256, 128, device=device)

        self.bilin_end = torch.nn.Bilinear(128, 128, pos_dim + quat_dim, device=device)

        self.relu = torch.nn.ReLU()

    def forward(self, x, q, t):
        if t == 0:
            # return torch.zeros(pos_dim), torch.zeros(quat_dim)
            return torch.zeros(pos_dim).to(device), torch.zeros(quat_dim).to(device)

        t = torch.tensor(t, dtype=torch.float).to(device)
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_x = torch.tensor(higher_x.flatten(), dtype=torch.float).to(device)
        input_x = torch.cat((higher_x.clone().detach(), t.clone().detach()), 0)
        input_x = self.linear_x_1(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_2(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_3(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_4(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_5(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_6(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_7(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_8(input_x)
        input_x = self.relu(input_x)

        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_q = torch.tensor(higher_q.flatten(), dtype=torch.float).to(device)
        input_q = torch.cat((higher_q.clone().detach(), t.clone().detach()), 0)
        input_q = self.linear_q_1(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_2(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_3(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_4(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_5(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_6(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_7(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_8(input_q)
        input_q = self.relu(input_q)

        input_total = self.bilin_end(input_x, input_q)

        out_x, out_q1, out_q2 = torch.split(input_total, pos_dim)
        out_q = torch.cat((out_q1, out_q2))
        out_q_2 = nn.functional.normalize(out_q, dim=0)

        return out_x, out_q_2, out_q


def higher_dim_gamma(p, length_ar):  #as per original nerf paper

    # move tensor into cpu and transform in numpy array
    if isinstance(p, torch.Tensor):
        p = p.cpu().numpy()  
        
    sins = np.sin(np.outer(2 ** np.array(list(range(length_ar))) * np.pi, p))
    coss = np.cos(np.outer(2 ** np.array(list(range(length_ar))) * np.pi, p))
    return np.column_stack((sins, coss))

