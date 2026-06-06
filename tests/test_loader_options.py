import unittest
from types import SimpleNamespace

import torch

from mania_difficulty.train import dataloader_options


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


if __name__ == "__main__":
    unittest.main()
