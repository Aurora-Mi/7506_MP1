"""Validation-only search for the strictly causal within-window neural cache."""
import argparse
import json
import math

import torch
from torch.nn import functional as F

from common import make_model, setup, windows
from dev_data import load_development_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_development_data(('validation',))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    thetas = (2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0)
    lambdas = (0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
    totals = {(theta, weight): 0.0 for theta in thetas for weight in lambdas}

    with torch.no_grad():
        for x, y in windows(data['validation'][0]):
            features = model.features(x)
            base_logp = F.log_softmax(model.head(features).float(), dim=-1)
            valid = y != -100
            target_logp = base_logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            similarity = F.normalize(features.float(), dim=-1) @ F.normalize(
                features.float(), dim=-1).transpose(-1, -2)
            length = x.shape[1]
            causal = torch.ones(length, length, dtype=torch.bool).tril(-1)
            outcomes = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            matches = outcomes[:, None, :] == y.clamp_min(0)[:, :, None]
            for theta in thetas:
                scores = (theta * similarity).masked_fill(~causal, float('-inf'))
                scores[:, 0, 0] = 0.0
                cache_target_p = (scores.softmax(-1) * matches).sum(-1)
                for weight in lambdas:
                    if weight == 0.0:
                        mixed_logp = target_logp
                    else:
                        mixed_logp = torch.logaddexp(
                            target_logp + math.log1p(-weight),
                            cache_target_p.clamp_min(1e-30).log() + math.log(weight),
                        )
                        mixed_logp[:, 0] = target_logp[:, 0]
                    totals[theta, weight] -= mixed_logp[valid].double().sum().item()

    byte_count = data['validation'][1]
    rows = [
        {'theta': theta, 'lambda': weight, 'bpb': nll / math.log(2.0) / byte_count}
        for (theta, weight), nll in totals.items()
    ]
    rows.sort(key=lambda row: row['bpb'])
    print(json.dumps(rows[:20], indent=2))


if __name__ == '__main__':
    main()
