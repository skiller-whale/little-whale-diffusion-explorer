import unittest

import numpy as np

from dataset import WhaleDataset, WhaleDatasetConfig, render_whale
from model import WhaleUNet


class DatasetTests(unittest.TestCase):
    def test_render_is_deterministic_and_rgb(self):
        config = WhaleDatasetConfig(length=10)
        first = np.asarray(render_whale(3, config))
        second = np.asarray(render_whale(3, config))
        self.assertEqual(first.shape, (64, 64, 3))
        self.assertTrue(np.array_equal(first, second))

    def test_dataset_range_and_shape(self):
        sample, condition = WhaleDataset(WhaleDatasetConfig(length=1))[0]
        self.assertEqual(tuple(sample.shape), (3, 64, 64))
        self.assertEqual(tuple(condition.shape), (16,))
        self.assertEqual(float(condition.sum()), 1)
        self.assertGreaterEqual(float(sample.min()), -1)
        self.assertLessEqual(float(sample.max()), 1)

    def test_model_contract(self):
        import torch
        output = WhaleUNet()(torch.randn(1, 3, 64, 64), torch.randn(1, 192), torch.eye(16)[:1])
        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))
