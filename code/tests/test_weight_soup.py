import unittest

import torch

from tune_weight_soup import interpolate_state_dict


class WeightSoupTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        state_a = {'weight': torch.tensor([1.0, 3.0]),
                   'counter': torch.tensor(2, dtype=torch.long)}
        state_b = {'weight': torch.tensor([5.0, 7.0]),
                   'counter': torch.tensor(2, dtype=torch.long)}
        torch.testing.assert_close(
            interpolate_state_dict(state_a, state_b, 0.0)['weight'], state_a['weight'])
        torch.testing.assert_close(
            interpolate_state_dict(state_a, state_b, 1.0)['weight'], state_b['weight'])
        torch.testing.assert_close(
            interpolate_state_dict(state_a, state_b, 0.5)['weight'],
            torch.tensor([3.0, 5.0]))

    def test_rejects_incompatible_states(self):
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            interpolate_state_dict({'a': torch.ones(1)}, {'b': torch.ones(1)}, 0.5)
        with self.assertRaisesRegex(ValueError, 'Non-floating'):
            interpolate_state_dict({'a': torch.tensor(1)}, {'a': torch.tensor(2)}, 0.5)
        with self.assertRaisesRegex(ValueError, r'\[0, 1\]'):
            interpolate_state_dict({'a': torch.ones(1)}, {'a': torch.ones(1)}, 1.5)


if __name__ == '__main__':
    unittest.main()
