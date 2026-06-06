import tempfile
import unittest
from pathlib import Path

from mania_difficulty.train import (
    checkpoint_backup_run_dir,
    restore_checkpoint_backup,
    sync_checkpoint_backup,
)


class CheckpointBackupTests(unittest.TestCase):
    def test_checkpoint_backup_run_dir_uses_run_name_under_base_dir(self):
        path = checkpoint_backup_run_dir(Path("drive/checkpoints"), "colab/lstm")

        self.assertEqual(str(path).replace("\\", "/"), "drive/checkpoints/colab_lstm")

    def test_sync_and_restore_checkpoint_backup_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "runs" / "run_a"
            backup_dir = root / "drive" / "run_a"
            run_dir.mkdir(parents=True)
            (run_dir / "last_checkpoint.pt").write_bytes(b"last")
            (run_dir / "best_model.pt").write_bytes(b"best")
            (run_dir / "history.csv").write_text("epoch,train_loss,val_loss\n1,0.1,0.2\n", encoding="utf-8")

            copied = sync_checkpoint_backup(run_dir, backup_dir)

            self.assertEqual(
                sorted(path.name for path in copied),
                ["best_model.pt", "history.csv", "last_checkpoint.pt"],
            )
            self.assertTrue((backup_dir / "last_checkpoint.pt").exists())

            for path in run_dir.iterdir():
                path.unlink()

            restored = restore_checkpoint_backup(run_dir, backup_dir)

            self.assertEqual(
                sorted(path.name for path in restored),
                ["best_model.pt", "history.csv", "last_checkpoint.pt"],
            )
            self.assertEqual((run_dir / "last_checkpoint.pt").read_bytes(), b"last")
            history_header = (run_dir / "history.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(history_header, "epoch,train_loss,val_loss")


if __name__ == "__main__":
    unittest.main()
