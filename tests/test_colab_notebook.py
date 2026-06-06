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
        self.assertIn("calibration_columns", source)
        self.assertIn("prediction_summary.csv", source)
        self.assertIn("cv_prediction_summary.csv", source)
        self.assertIn("calibration_warning", source)
        self.assertIn("TRAIN_DEVICE", source)
        self.assertIn("LOADER_WORKERS", source)
        self.assertIn("AMP_MODE", source)
        self.assertIn('"--device", TRAIN_DEVICE', source)
        self.assertIn('"--loader-workers", str(LOADER_WORKERS)', source)


if __name__ == "__main__":
    unittest.main()
