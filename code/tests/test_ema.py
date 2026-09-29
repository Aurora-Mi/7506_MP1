import unittest

import torch

from ema import ModelEMA
from student import build_model


class EMATests(unittest.TestCase):
    def test_average_updates_tied_weights_once_and_leaves_live_model_untouched(self):
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256))
        initial_rng = torch.get_rng_state().clone()
        average = ModelEMA(model, decay=0.9)
        torch.testing.assert_close(torch.get_rng_state(), initial_rng)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        live = {name: value.detach().clone() for name, value in model.named_parameters()}
        average.update(model)
        for name, parameter in average.model.named_parameters():
            torch.testing.assert_close(parameter, initial[name] + 0.1)
            self.assertFalse(parameter.requires_grad)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, live[name])
        self.assertIs(average.model.token.weight, average.model.head.weight)
        self.assertTrue(model.training)
        self.assertFalse(average.model.training)
        self.assertEqual(average.updates, 1)


if __name__ == '__main__':
    unittest.main()
