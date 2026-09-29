"""Student GPT plus a fixed, train-derived bigram distribution (inference only)."""
import math

import torch

from student import StudentGPT


class StudentBigram(StudentGPT):
    def __init__(self, config):
        super().__init__(config)
        self.bigram_lambda = float(config.get('bigram_lambda', 0.02))
        if not math.isfinite(self.bigram_lambda) or not 0 <= self.bigram_lambda < 1:
            raise ValueError('bigram_lambda must be finite and in [0, 1).')
        vocab = config['vocab']
        # A normalized default permits contract tests; export replaces it with
        # train-only statistics. Persistent buffer travels inside checkpoint.pt.
        self.register_buffer('bigram_log_probs', torch.full((vocab, vocab), -math.log(vocab)))

    def predict_log_probs(self, ids):
        neural_logp = super().predict_log_probs(ids)
        if self.bigram_lambda == 0:
            return neural_logp
        # At position t, ids[t] is already observed. No reference target, state,
        # neighboring example or evaluation answer is used in this lookup.
        return torch.logaddexp(
            neural_logp + math.log1p(-self.bigram_lambda),
            self.bigram_log_probs[ids].float() + math.log(self.bigram_lambda),
        )


def build_model(config):
    return StudentBigram(config)
