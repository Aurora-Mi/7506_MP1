import unittest

import torch

from student import build_model


class DropoutCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = dict(vocab=2048, width=32, heads=4, depth=2,
                           context=256, dropout=0.1, mtp_heads=2, mtp_weight=0.2,
                           cache_lambda=0.07, logit_temperature=1.08)

    def test_explicit_legacy_rates_preserve_training_outputs(self):
        legacy = build_model(self.config).train()
        explicit = build_model(self.config | dict(attention_dropout=0.1,
                                                  embedding_dropout=0.1,
                                                  residual_dropout=0.1)).train()
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        batch = torch.randint(0, 2048, (2, 17))
        torch.manual_seed(123)
        legacy_loss = legacy.training_loss(batch)
        torch.manual_seed(123)
        explicit_loss = explicit.training_loss(batch)
        torch.testing.assert_close(legacy_loss, explicit_loss, atol=0, rtol=0)

    def test_split_rates_do_not_change_inference_or_checkpoint_structure(self):
        legacy = build_model(self.config).eval()
        split = build_model(self.config | dict(attention_dropout=0.05,
                                               embedding_dropout=0.1,
                                               residual_dropout=0.1)).eval()
        split.load_state_dict(legacy.state_dict(), strict=True)
        ids = torch.randint(0, 2048, (2, 17))
        with torch.no_grad():
            torch.testing.assert_close(legacy.predict_log_probs(ids),
                                       split.predict_log_probs(ids), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
