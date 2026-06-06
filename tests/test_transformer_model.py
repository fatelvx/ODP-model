import unittest

import torch

from mania_difficulty.models.factory import create_model
from mania_difficulty.models.transformer import TransformerDifficultyModel


class TransformerModelTests(unittest.TestCase):
    def test_transformer_forward_encode_and_attention_shapes(self):
        model = TransformerDifficultyModel(
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            output_dim=3,
            max_positions=8,
            dropout=0.0,
            head_dropout=0.0,
        )
        features = torch.rand(2, 5, 6)
        lengths = torch.tensor([5, 3], dtype=torch.long)

        pred = model(features, lengths)
        encoded = model.encode(features, lengths)
        attention = model.attention_importance(features, lengths)

        self.assertEqual(tuple(pred.shape), (2, 3))
        self.assertEqual(tuple(encoded.shape), (2, 16))
        self.assertEqual(tuple(attention.shape), (2, 5))
        self.assertTrue(torch.isfinite(attention).all())
        self.assertEqual(float(attention[1, 3:].sum().detach()), 0.0)

    def test_factory_creates_transformer(self):
        model = create_model(
            "transformer",
            output_dim=3,
            config={
                "embed_dim": 16,
                "num_heads": 4,
                "num_layers": 1,
                "ff_dim": 32,
                "max_positions": 8,
            },
        )

        self.assertIsInstance(model, TransformerDifficultyModel)


if __name__ == "__main__":
    unittest.main()
