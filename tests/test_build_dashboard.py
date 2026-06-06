import json
import tempfile
import unittest
from pathlib import Path

from mania_difficulty.tools.build_dashboard import write_dashboard


class BuildDashboardTests(unittest.TestCase):
    def test_write_dashboard_links_runs_reports_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "runs" / "run_a"
            audit_dir = root / "outputs" / "dataset_audit"
            sweep_dir = root / "outputs" / "forest_sweep"
            run_dir.mkdir(parents=True)
            audit_dir.mkdir(parents=True)
            sweep_dir.mkdir(parents=True)
            (run_dir / "run_report.html").write_text("<html>run</html>", encoding="utf-8")
            (run_dir / "learning_curve.png").write_bytes(b"png")
            (run_dir / "prediction_scatter.png").write_bytes(b"png")
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mean_acc": {"mae": 0.1, "spearman": 0.5, "pairwise_order_accuracy": 0.75},
                        "_run": {"model_name": "summary", "evaluation": "holdout", "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (audit_dir / "dataset_audit.html").write_text("<html>audit</html>", encoding="utf-8")
            (audit_dir / "dataset_audit.json").write_text(
                json.dumps({"usable_rows": 10, "missing_sequence_count": 1, "group_count": 3}),
                encoding="utf-8",
            )
            (sweep_dir / "sweep_report.html").write_text("<html>sweep</html>", encoding="utf-8")
            (sweep_dir / "best_params.json").write_text(
                json.dumps({"candidate_id": "best_forest", "selection_metric": "mean_mae"}),
                encoding="utf-8",
            )

            out_html = root / "outputs" / "dashboard.html"
            write_dashboard(
                out_html,
                run_dirs=[run_dir],
                audit_dir=audit_dir,
                forest_sweep_dir=sweep_dir,
            )

            html = out_html.read_text(encoding="utf-8")

        self.assertIn("run_a", html)
        self.assertIn("run_report.html", html)
        self.assertIn("learning_curve.png", html)
        self.assertIn("mean_acc", html)
        self.assertIn("Dataset Audit", html)
        self.assertIn("best_forest", html)


if __name__ == "__main__":
    unittest.main()
