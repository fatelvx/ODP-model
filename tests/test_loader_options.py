import argparse
import unittest
from types import SimpleNamespace

import torch

from mania_difficulty.train import (
    dataloader_options,
    gradient_accumulation_steps,
    latest_checkpoint_path,
    mixed_precision_enabled,
    positive_int,
    runtime_environment_metadata,
)


class LoaderOptionTests(unittest.TestCase):
    def test_dataloader_options_keep_cpu_safe_by_default(self):
        args = SimpleNamespace(loader_workers=0, pin_memory="auto", loader_prefetch_factor=2)

        options = dataloader_options(args, torch.device("cpu"))

        self.assertEqual(options["num_workers"], 0)
        self.assertFalse(options["pin_memory"])
        self.assertNotIn("persistent_workers", options)
        self.assertNotIn("prefetch_factor", options)

    def test_dataloader_options_enable_cuda_prefetch_when_workers_are_set(self):
        args = SimpleNamespace(loader_workers=2, pin_memory="auto", loader_prefetch_factor=3)

        options = dataloader_options(args, torch.device("cuda"))

        self.assertEqual(options["num_workers"], 2)
        self.assertTrue(options["pin_memory"])
        self.assertTrue(options["persistent_workers"])
        self.assertEqual(options["prefetch_factor"], 3)

    def test_mixed_precision_auto_only_enables_cuda(self):
        args = SimpleNamespace(amp="auto")

        self.assertFalse(mixed_precision_enabled(args, torch.device("cpu")))
        self.assertTrue(mixed_precision_enabled(args, torch.device("cuda")))

    def test_mixed_precision_on_rejects_cpu(self):
        args = SimpleNamespace(amp="on")

        with self.assertRaises(RuntimeError):
            mixed_precision_enabled(args, torch.device("cpu"))

    def test_mixed_precision_off_disables_cuda(self):
        args = SimpleNamespace(amp="off")

        self.assertFalse(mixed_precision_enabled(args, torch.device("cuda")))

    def test_gradient_accumulation_defaults_to_one(self):
        args = SimpleNamespace()

        self.assertEqual(gradient_accumulation_steps(args), 1)

    def test_gradient_accumulation_uses_positive_step_count(self):
        args = SimpleNamespace(grad_accum_steps=4)

        self.assertEqual(gradient_accumulation_steps(args), 4)

    def test_positive_int_rejects_zero_and_negative_values(self):
        self.assertEqual(positive_int("3"), 3)

        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("-1")

    def test_latest_checkpoint_path_is_stable_inside_run_dir(self):
        self.assertEqual(
            str(latest_checkpoint_path("outputs/runs/demo")).replace("\\", "/"),
            "outputs/runs/demo/last_checkpoint.pt",
        )

    def test_runtime_environment_metadata_records_device_and_torch(self):
        args = SimpleNamespace(device="")

        metadata = runtime_environment_metadata(args, torch.device("cpu"))

        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["requested_device"], "auto")
        self.assertEqual(metadata["torch_version"], torch.__version__)
        self.assertEqual(metadata["cuda_available"], torch.cuda.is_available())
        self.assertIn("cuda_device_count", metadata)


if __name__ == "__main__":
    unittest.main()
