import json
import unittest
from pathlib import Path


class ColabNotebookTests(unittest.TestCase):
    def test_colab_notebook_packages_outputs_with_manifest(self):
        notebook = json.loads(Path("notebooks/colab_train.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

        self.assertIn("package_artifacts", source)
        self.assertIn("outputs/colab_artifact_manifest.json", source)
        self.assertIn("colab_top100_outputs.zip", source)
        self.assertIn("GRAD_CLIP_NORM", source)
        self.assertIn("--grad-clip-norm", source)
        self.assertIn("MAX_NOTES", source)
        self.assertIn("--max-notes {MAX_NOTES}", source)
        self.assertIn('"--max-notes", str(MAX_NOTES)', source)
        self.assertIn("HUBER_DELTA", source)
        self.assertIn("HUBER_DELTAS", source)
        self.assertIn("--huber-delta", source)
        self.assertIn("--huber-deltas", source)
        self.assertIn("calibration_columns", source)
        self.assertIn("prediction_summary.csv", source)
        self.assertIn("cv_prediction_summary.csv", source)
        self.assertIn("calibration_worst_p90_target", source)
        self.assertIn("calibration_worst_p90_abs_error", source)
        self.assertIn("prediction_errors.png", source)
        self.assertIn("cv_prediction_errors.png", source)
        self.assertIn("calibration_warning", source)
        self.assertIn("revision_columns", source)
        self.assertIn("git_commit", source)
        self.assertIn("git_branch", source)
        self.assertIn("git_dirty", source)
        self.assertIn("git_status_entries", source)
        self.assertIn("Artifact source revision", source)
        self.assertIn("manifest.get(\"source\", {})", source)
        self.assertIn("TRAIN_DEVICE", source)
        self.assertIn("LOADER_WORKERS", source)
        self.assertIn("AMP_MODE", source)
        self.assertIn('"--device", TRAIN_DEVICE', source)
        self.assertIn('"--loader-workers", str(LOADER_WORKERS)', source)


if __name__ == "__main__":
    unittest.main()
