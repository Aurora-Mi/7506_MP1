import unittest

import torch

from student import build_model
from train import make_parameter_groups


class WeightDecayTests(unittest.TestCase):
    def make_model(self):
        return build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                context=256, mtp_heads=2, mtp_weight=0.2))

    def test_matrix_groups_cover_parameters_once_including_tied_weights_and_mtp(self):
        model = self.make_model()
        groups, summary = make_parameter_groups(model, 0.1, 'matrix')
        grouped = [parameter for group in groups for parameter in group['params']]
        identities = [id(parameter) for parameter in grouped]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(set(identities), {id(p) for p in model.parameters()})
        self.assertEqual(identities.count(id(model.token.weight)), 1)
        self.assertIs(model.token.weight, model.head.weight)
        for group in groups:
            for parameter in group['params']:
                self.assertEqual(group['weight_decay'], 0.1 if parameter.ndim >= 2 else 0.0)
        decay_names = summary[0]['parameter_names']
        self.assertIn('token.weight', decay_names)
        self.assertIn('mtp_heads.0.weight', decay_names)
        self.assertIn('mtp_heads.1.weight', decay_names)
        self.assertIn('norm.weight', summary[1]['parameter_names'])

    def test_zero_gradients_decay_only_matrices(self):
        model = self.make_model()
        groups, _ = make_parameter_groups(model, 0.1, 'matrix')
        optimizer = torch.optim.AdamW(groups, lr=0.1)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            expected = before[name] * 0.99 if parameter.ndim >= 2 else before[name]
            torch.testing.assert_close(parameter, expected)

    def test_default_policy_keeps_original_all_parameter_decay(self):
        model = self.make_model()
        groups, _ = make_parameter_groups(model, 0.1, 'all')
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['weight_decay'], 0.1)
        self.assertEqual({id(p) for p in groups[0]['params']}, {id(p) for p in model.parameters()})


if __name__ == '__main__':
    unittest.main()
