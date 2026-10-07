import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metrics import binary_curve_metrics
from visualizations import plot_validation_comparison


class CurveMetricsTest(unittest.TestCase):
    def test_perfect_classifier_has_unit_roc_auc_and_average_precision(self):
        result = binary_curve_metrics(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.2, 0.8, 0.9],
        )
        self.assertAlmostEqual(float(result["roc_auc"]), 1.0)
        self.assertAlmostEqual(float(result["average_precision"]), 1.0)

    def test_curves_require_both_classes(self):
        with self.assertRaises(ValueError):
            binary_curve_metrics([1, 1], [0.2, 0.9])


class PlotSmokeTest(unittest.TestCase):
    def test_validation_comparison_plot_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "comparison.csv"
            csv_path.write_text(
                "model,training_bce,validation_bce,training_zero_one_error,validation_zero_one_error\n"
                "cnn,0.2,0.3,0.1,0.15\n"
                "mlp,0.3,0.4,0.2,0.25\n",
                encoding="utf-8",
            )
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps({"selected_model": "cnn"}), encoding="utf-8"
            )
            output_path = root / "figure.png"
            plot_validation_comparison(csv_path, selection_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
