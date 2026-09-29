import json
import unittest

import torch

from common import ROOT
from student import build_model


class WideConfigTests(unittest.TestCase):
    def test_wide_config_only_changes_width_and_heads_from_calibrated_recipe(self):
        reference = json.loads((ROOT / 'configs/student_192x8_mtp2.json').read_text())
        reference['logit_temperature'] = 1.12
        wide = json.loads((ROOT / 'configs/student_224x8_mtp2.json').read_text())
        differences = {key for key in reference if reference[key] != wide[key]}
        self.assertEqual(differences, {'width', 'heads'})
        self.assertEqual(wide['width'] // wide['heads'], 32)
        self.assertEqual(wide['width'] % wide['heads'], 0)

    def test_wide_model_loss_gradients_and_inference(self):
        torch.set_num_threads(2)
        config = json.loads((ROOT / 'configs/student_224x8_mtp2.json').read_text())
        model = build_model(config)
        self.assertEqual(len(model.blocks), 8)
        self.assertEqual(model.blocks[0].mlp.down.in_features, 600)
        self.assertIs(model.token.weight, model.head.weight)
        batch = torch.randint(0, 2048, (1, 9))
        loss = model.training_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.blocks[0].qkv.weight.grad)
        self.assertIsNotNone(model.mtp_heads[0].weight.grad)
        inference = build_model(config | {'mtp_heads': 0, 'mtp_weight': 0.0}).eval()
        state = {key: value for key, value in model.state_dict().items()
                 if not key.startswith('mtp_heads.')}
        inference.load_state_dict(state, strict=True)
        with torch.no_grad():
            logp = inference.predict_log_probs(batch[:, :-1])
        self.assertEqual(logp.shape, (1, 8, 2048))
        self.assertTrue(torch.isfinite(logp).all())
        torch.testing.assert_close(torch.logsumexp(logp, -1),
                                   torch.zeros(1, 8), atol=1e-5, rtol=0)

    def test_deeper_config_only_adds_one_block(self):
        reference = json.loads((ROOT / 'configs/student_224x8_mtp2.json').read_text())
        deeper = json.loads((ROOT / 'configs/student_224x9_mtp2.json').read_text())
        differences = {key for key in reference if reference[key] != deeper[key]}
        self.assertEqual(differences, {'depth'})
        self.assertEqual(reference['depth'], 8)
        self.assertEqual(deeper['depth'], 9)
        self.assertEqual(deeper['width'] // deeper['heads'], 32)

    def test_deeper_model_builds_nine_blocks(self):
        config = json.loads((ROOT / 'configs/student_224x9_mtp2.json').read_text())
        model = build_model(config)
        self.assertEqual(len(model.blocks), 9)
        self.assertEqual(model.blocks[0].mlp.down.in_features, 600)
        self.assertIs(model.token.weight, model.head.weight)
        batch = torch.randint(0, 2048, (1, 5))
        loss = model.training_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.blocks[8].qkv.weight.grad)


if __name__ == '__main__':
    unittest.main()
