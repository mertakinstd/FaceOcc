import unittest
from unittest import mock

import torch

from Dataset.transforms import RandomAffine
from loss import IoU, Precision
from meter import AverageValueMeter


class RandomAffineTranslationTests(unittest.TestCase):
    def test_translation_is_independent_symmetric_and_fractional(self):
        transform = RandomAffine(scale=0.0, angle=0.0, flip=0.0, translate=0.1)
        with mock.patch("Dataset.transforms.torch.rand", side_effect=[torch.tensor([0.0]), torch.tensor([1.0])]):
            tx, ty = transform._sample_translation(height=128, width=256)

        self.assertAlmostEqual(tx, -2.0 * (0.1 * 256) / 255, places=7)
        self.assertAlmostEqual(ty, 2.0 * (0.1 * 128) / 127, places=7)
        self.assertLess(tx, 0.0)
        self.assertGreater(ty, 0.0)


class AverageValueMeterTests(unittest.TestCase):
    def test_weighted_update_uses_n_as_weight(self):
        meter = AverageValueMeter()
        meter.add(2.0, n=2)
        meter.add(10.0, n=1)
        self.assertEqual(meter.n, 3)
        self.assertAlmostEqual(meter.mean, 14.0 / 3.0)


class SegmentationMetricTests(unittest.TestCase):
    def test_iou_is_macro_averaged_per_image(self):
        logits = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[-1.0, -1.0], [-1.0, -1.0]]],
        ])
        targets = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ])
        self.assertAlmostEqual(IoU()(logits, targets).item(), 0.5, places=7)

    def test_precision_is_macro_averaged_per_image(self):
        logits = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, -1.0], [-1.0, -1.0]]],
        ])
        targets = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ])
        self.assertAlmostEqual(Precision()(logits, targets).item(), 0.5, places=7)

    def test_empty_union_is_finite(self):
        logits = torch.full((1, 1, 2, 2), -1.0)
        targets = torch.zeros_like(logits)
        score = IoU()(logits, targets)
        self.assertTrue(torch.isfinite(score))
        self.assertEqual(score.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
