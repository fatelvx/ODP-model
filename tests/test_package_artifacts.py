import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from mania_difficulty.tools.package_artifacts import package_artifacts


class PackageArtifactsTests(unittest.TestCase):
    def test_package_artifacts_writes_manifest_and_skips_missing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            run_dir = outputs / "runs" / "run_a"
            run_dir.mkdir(parents=True)
            (run_dir / "run_report.html").write_text("<html>run</html>", encoding="utf-8")
            (run_dir / "metrics.json").write_text('{"mean_acc": {"mae": 0.1}}', encoding="utf-8")
            (outputs / "dashboard.html").write_text("<html>dashboard</html>", encoding="utf-8")

            out_zip = root / "colab_outputs.zip"
            manifest_path = outputs / "artifact_manifest.json"
            git_metadata = {
                "git_commit": "abc1234",
                "git_branch": "main",
                "git_dirty": False,
                "git_status_entries": 0,
            }
            with patch(
                "mania_difficulty.tools.package_artifacts.git_environment_metadata",
                return_value=git_metadata,
            ):
                manifest = package_artifacts(
                    [
                        outputs / "dashboard.html",
                        run_dir,
                        outputs / "missing_report.html",
                    ],
                    out_zip,
                    manifest_path=manifest_path,
                    root=root,
                )

            manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
            with zipfile.ZipFile(out_zip) as archive:
                names = set(archive.namelist())

        self.assertEqual(manifest["output_zip"], "colab_outputs.zip")
        self.assertEqual(manifest_json["output_zip"], "colab_outputs.zip")
        self.assertIn("outputs/missing_report.html", manifest_json["missing"])
        self.assertIn("outputs/dashboard.html", names)
        self.assertIn("outputs/runs/run_a/run_report.html", names)
        self.assertIn("outputs/runs/run_a/metrics.json", names)
        self.assertIn("outputs/artifact_manifest.json", names)
        self.assertEqual(manifest_json["file_count"], 4)
        self.assertGreater(manifest_json["total_size_bytes"], 0)
        self.assertEqual(manifest["source"]["git_commit"], "abc1234")
        self.assertEqual(manifest_json["source"]["git_branch"], "main")
        self.assertFalse(manifest_json["source"]["git_dirty"])


if __name__ == "__main__":
    unittest.main()
