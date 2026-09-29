import unittest

import torch

from export_bigram import fit_bigram_log_probs
from student import StudentGPT
from student_bigram import build_model


class BigramTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                           cache_lambda=0.07, cache_theta=10, logit_temperature=1.16,
                           bigram_lambda=0.02)
        self.model = build_model(self.config).eval()
        self.train = torch.tensor([2, 3, 2, 4, 2, 3, 3, 9])
        self.model.bigram_log_probs.copy_(fit_bigram_log_probs(self.train, 2048, 20))

    def test_statistics_match_search_formula_and_are_normalized(self):
        counts = torch.zeros(2048, 2048, dtype=torch.float64)
        for a, b in zip(self.train[:-1], self.train[1:]):
            counts[a, b] += 1
        unigram_counts = torch.bincount(self.train[1:], minlength=2048).double()
        unigram = (unigram_counts + 1) / (len(self.train) - 1 + 2048)
        expected = (counts + 20 * unigram) / (counts.sum(1, keepdim=True) + 20)
        torch.testing.assert_close(self.model.bigram_log_probs.double().exp(),
                                   expected, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(self.model.bigram_log_probs.logsumexp(-1),
                                   torch.zeros(2048), atol=1e-6, rtol=0)

    def test_mixture_matches_target_only_search(self):
        ids = torch.tensor([[2, 3, 2, 4, 2]])
        targets = torch.tensor([[3, 2, 4, 2, 9]])
        with torch.no_grad():
            base_logp = StudentGPT.predict_log_probs(self.model, ids)
            neural_p = base_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).double().exp()
            bigram_p = self.model.bigram_log_probs[ids].gather(
                -1, targets.unsqueeze(-1)).squeeze(-1).double().exp()
            expected = (0.98 * neural_p + 0.02 * bigram_p).log()
            actual = self.model.predict_log_probs(ids).gather(
                -1, targets.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)

    def test_causality_normalization_independence_and_reset(self):
        ids = torch.randint(0, 2048, (2, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            first = self.model.predict_log_probs(ids)
            modified = self.model.predict_log_probs(changed)
            alone = self.model.predict_log_probs(ids[:1])
            again = self.model.predict_log_probs(ids)
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=0)
        torch.testing.assert_close(first[:, :7], modified[:, :7], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first, again, atol=0, rtol=0)

    def test_zero_weight_and_checkpoint_roundtrip(self):
        ids = torch.randint(0, 2048, (1, 6))
        cloned = build_model(self.config).eval()
        cloned.load_state_dict(self.model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(self.model.predict_log_probs(ids),
                                       cloned.predict_log_probs(ids), atol=0, rtol=0)
            cloned.bigram_lambda = 0
            torch.testing.assert_close(cloned.predict_log_probs(ids),
                                       StudentGPT.predict_log_probs(cloned, ids), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
