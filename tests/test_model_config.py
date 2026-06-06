import unittest
from types import SimpleNamespace

from mania_difficulty.train import model_config_from_args


class ModelConfigTests(unittest.TestCase):
    def test_lstm_config_uses_cli_values(self):
        args = SimpleNamespace(
            model="lstm",
            lstm_embed_dim=32,
            lstm_hidden_dim=64,
            lstm_layers=1,
            lstm_dropout=0.15,
            lstm_head_dropout=0.25,
            summary_hidden_dim=128,
            summary_dropout=0.2,
        )

        config = model_config_from_args(args)

        self.assertEqual(config["embed_dim"], 32)
        self.assertEqual(config["hidden_dim"], 64)
        self.assertEqual(config["num_layers"], 1)
        self.assertEqual(config["dropout"], 0.15)
        self.assertEqual(config["head_dropout"], 0.25)

    def test_summary_config_uses_cli_values(self):
        args = SimpleNamespace(
            model="summary",
            lstm_embed_dim=32,
            lstm_hidden_dim=64,
            lstm_layers=1,
            lstm_dropout=0.15,
            lstm_head_dropout=0.25,
            summary_hidden_dim=96,
            summary_dropout=0.35,
        )

        config = model_config_from_args(args)

        self.assertEqual(config["hidden_dim"], 96)
        self.assertEqual(config["dropout"], 0.35)

    def test_transformer_config_uses_cli_values(self):
        args = SimpleNamespace(
            model="transformer",
            max_notes=512,
            transformer_embed_dim=48,
            transformer_heads=6,
            transformer_layers=2,
            transformer_ff_dim=192,
            transformer_dropout=0.15,
            transformer_head_dropout=0.25,
        )

        config = model_config_from_args(args)

        self.assertEqual(config["embed_dim"], 48)
        self.assertEqual(config["num_heads"], 6)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["ff_dim"], 192)
        self.assertEqual(config["dropout"], 0.15)
        self.assertEqual(config["head_dropout"], 0.25)
        self.assertEqual(config["max_positions"], 512)

    def test_tabular_forest_has_no_torch_model_config(self):
        args = SimpleNamespace(model="tabular_forest")

        self.assertEqual(model_config_from_args(args), {})


if __name__ == "__main__":
    unittest.main()
