import unittest

import torch

from transModel import DeformationNetworkSeparate
from transModel import DeformationNetworkConnected
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestModel(unittest.TestCase):

    def test_DeformationNetworkSeparate(self):
        networkx = DeformationNetworkSeparate()
        data_pos = np.random.randn(3)
        data_quat = np.random.randn(4)
        t = np.random.randn(1)
        test = networkx.forward(data_pos, data_quat, 0)
        assert torch.equal(test[0], torch.zeros(3).to(DEVICE))
        assert torch.equal(test[1], torch.zeros(4).to(DEVICE))
        test2 = networkx.forward(data_pos, data_quat, t)
        quat = test2[1].clone().to('cpu').detach().numpy()
        norm = np.linalg.norm(quat)
        print(norm)
        print(test2)  # sometimes, norm does not equal 1 but gets very close to 1, problem is probably rounding errors
        assert norm == 1
        return 0

    def test_DeformationNetworkConnected(self):
        networkx = DeformationNetworkConnected()
        data_pos = np.random.randn(3)
        data_quat = np.random.randn(4)
        test = networkx.forward(data_pos, data_quat, 0)
        assert torch.equal(test[0], torch.zeros(3).to(DEVICE))
        assert torch.equal(test[1], torch.zeros(4).to(DEVICE))
        test2 = networkx.forward(data_pos, data_quat, 1)
        quat = test2[1].clone().to('cpu').detach().numpy()
        norm = np.linalg.norm(quat)
        print(norm)
        print(test2)  # sometimes, norm does not equal 1 but gets very close to 1, problem is probably rounding errors
        assert norm == 1
        return 0
