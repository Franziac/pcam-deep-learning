import math
import unittest

from metrics import binary_metrics


class BinaryMetricsTest(unittest.TestCase):
    def test_confusion_counts_and_zero_one_error(self):
        result = binary_metrics(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.8, 0.9, 0.2],
            threshold=0.5,
        )
        self.assertEqual(result["true_negatives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["zero_one_error"], 0.5)
        self.assertTrue(math.isfinite(result["binary_cross_entropy"]))

    def test_rejects_invalid_labels(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, 2], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
