import unittest

import torch
from torch.nn import functional as F

from student import build_model
from calibrate_checkpoint import cache_target_probability, mix_target_logp, search_grid
from evaluate import score


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = build_model(dict(vocab=2048, context=256, width=32,
                                     heads=4, depth=1, dropout=0.1)).eval()

    def test_target_only_calculation_matches_predictor(self):
        ids = torch.tensor([[7, 7, 3, 7, 3, 4], [3, 2, 3, 2, 9, 2]])
        targets = torch.tensor([[7, 3, 7, 3, 4, 6], [2, 3, 2, 9, 2, 1]])
        with torch.no_grad():
            features = self.model.features(ids)
            for temperature in (1.0, 1.08):
                for theta in (5.0, 20.0):
                    cache = cache_target_probability(features, ids, targets, theta)
                    base = F.log_softmax(self.model.head(features).float() / temperature, -1)
                    base_target = base.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    for weight in (0.0, 0.07):
                        self.model.logit_temperature = temperature
                        self.model.cache_theta = theta
                        self.model.cache_lambda = weight
                        expected = self.model.predict_log_probs(ids).gather(
                            -1, targets.unsqueeze(-1)).squeeze(-1)
                        actual = mix_target_logp(base_target, cache, weight)
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_grid_matches_scorer_including_padded_last_window(self):
        tokens = torch.tensor([3, 3, 4, 3, 4, 9, 3])
        candidates = [(1.08, 0.07, 10.0), (1.0, 0.0, 5.0)]
        rows = search_grid(self.model, tokens, 20, torch.device('cpu'), candidates, 2)
        for row in rows:
            for key in ('logit_temperature', 'cache_lambda', 'cache_theta'):
                setattr(self.model, key, row[key])
            result = score(self.model, tokens, 20, torch.device('cpu'), 'fp32', 2)
            self.assertAlmostEqual(result['bpb'], row['bpb'], places=6)


if __name__ == '__main__':
    unittest.main()
