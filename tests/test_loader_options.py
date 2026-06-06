import unittest
from types import SimpleNamespace

import torch

from mania_difficulty.train import dataloader_options, mixed_precision_enabled


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


if __name__ == "__main__":
    unittest.main()
