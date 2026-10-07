import json
import tempfile
import unittest
from pathlib import Path

from compare_models import compare_runs
from training_utils import select_best_epoch


COMMON = {
    "dataset_revision": "revision",
    "training_split": "train",
    "validation_split": "valid",
    "epochs": 2,
    "batch_size": 64,
    "seed": 42,
    "learning_rate": 0.001,
    "training_loss": "binary_cross_entropy",
    "test_split_loaded": False,
}


class SelectionTest(unittest.TestCase):
    def test_best_epoch_prefers_validation_loss(self):
        history = {
            "loss": [0.4, 0.3],
            "accuracy": [0.8, 0.9],
            "precision": [0.8, 0.9],
            "recall": [0.8, 0.9],
            "val_loss": [0.35, 0.36],
            "val_accuracy": [0.82, 0.88],
            "val_precision": [0.82, 0.88],
            "val_recall": [0.82, 0.88],
        }
        result = select_best_epoch(history)
        self.assertEqual(result["best_epoch"], 1)
        self.assertAlmostEqual(result["validation_bce"], 0.35)


    def test_best_epoch_exact_validation_loss_tie_keeps_earliest(self):
        history = {
            "loss": [0.5, 0.4],
            "accuracy": [0.75, 0.9],
            "val_loss": [0.35, 0.35],
            "val_accuracy": [0.80, 0.95],
        }
        result = select_best_epoch(history)
        self.assertEqual(result["best_epoch"], 1)

    def test_compare_runs_uses_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cnn = root / "cnn"
            mlp = root / "mlp"
            out = root / "out"
            cnn.mkdir()
            mlp.mkdir()

            cnn_summary = {
                **COMMON,
                "model": "cnn",
                "parameter_count": 100,
                "best_epoch": 2,
                "training_bce": 0.2,
                "validation_bce": 0.3,
                "training_accuracy": 0.9,
                "validation_accuracy": 0.85,
                "training_zero_one_error": 0.1,
                "validation_zero_one_error": 0.15,
                "training_precision": 0.9,
                "validation_precision": 0.85,
                "training_recall": 0.9,
                "validation_recall": 0.85,
            }
            mlp_summary = {
                **COMMON,
                "model": "mlp",
                "parameter_count": 200,
                "best_epoch": 1,
                "training_bce": 0.25,
                "validation_bce": 0.4,
                "training_accuracy": 0.88,
                "validation_accuracy": 0.8,
                "training_zero_one_error": 0.12,
                "validation_zero_one_error": 0.2,
                "training_precision": 0.88,
                "validation_precision": 0.8,
                "training_recall": 0.88,
                "validation_recall": 0.8,
            }
            (cnn / "validation_summary.json").write_text(json.dumps(cnn_summary))
            (mlp / "validation_summary.json").write_text(json.dumps(mlp_summary))
            (cnn / "best.weights.h5").touch()
            (mlp / "best.weights.h5").touch()

            selection_path = compare_runs(cnn, mlp, out)
            selection = json.loads(selection_path.read_text())
            self.assertEqual(selection["selected_model"], "cnn")
            self.assertFalse(selection["test_data_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
