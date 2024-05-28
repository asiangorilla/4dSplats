import unittest

import torch

from transModel import DeformationNetworkSeparate
from transModel import DeformationNetworkConnected
import numpy as np

class TestModel(unittest.TestCase):

    def test_DeformationNetworkSeparate(self):
        networkx = DeformationNetworkSeparate()
        data_pos = np.random.randn(3)
        data_quat = np.random.randn(4)
        test = networkx.forward(data_pos, data_quat, 0)
        assert torch.equal(test[0], torch.zeros(3))
        assert torch.equal(test[1], torch.zeros(4))
        test2 = networkx.forward(data_pos, data_quat, 1)
        print(test2)
        return 0

    def test_DeformationNetworkConnected(self):
        networkx = DeformationNetworkConnected()
        data_pos = np.random.randn(3)
        data_quat = np.random.randn(4)
        test = networkx.forward(data_pos, data_quat, 0)
        assert torch.equal(test[0], torch.zeros(3))
        assert torch.equal(test[1], torch.zeros(4))
        test2 = networkx.forward(data_pos, data_quat, 1)
        print(test2)
        return 0