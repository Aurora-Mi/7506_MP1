"""Validation-only joint search for logit temperature and cache weight."""
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

    temperatures = (0.90, 0.95, 1.00, 1.03, 1.05, 1.08, 1.10, 1.15)
    lambdas = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
    theta = 10.0
    totals = {(temperature, weight): 0.0
              for temperature in temperatures for weight in lambdas}

    with torch.no_grad():
        for x, y in windows(data['validation'][0]):
            features = model.features(x)
            logits = model.head(features).float()
            valid = y != -100
            targets = y.clamp_min(0)
            normalized = F.normalize(features.float(), dim=-1)
            similarity = normalized @ normalized.transpose(-1, -2)
            length = x.shape[1]
            causal = torch.ones(length, length, dtype=torch.bool).tril(-1)
            scores = (theta * similarity).masked_fill(~causal, float('-inf'))
            scores[:, 0, 0] = 0.0
            outcomes = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            matches = outcomes[:, None, :] == targets[:, :, None]
            cache_target_p = (scores.softmax(-1) * matches).sum(-1)

            for temperature in temperatures:
                target_logp = F.log_softmax(logits / temperature, dim=-1).gather(
                    -1, targets.unsqueeze(-1)).squeeze(-1)
                for weight in lambdas:
                    mixed = torch.logaddexp(
                        target_logp + math.log1p(-weight),
                        cache_target_p.clamp_min(1e-30).log() + math.log(weight),
                    )
                    mixed[:, 0] = target_logp[:, 0]
                    totals[temperature, weight] -= mixed[valid].double().sum().item()

    byte_count = data['validation'][1]
    rows = [
        {'temperature': temperature, 'lambda': weight,
         'bpb': nll / math.log(2.0) / byte_count}
        for (temperature, weight), nll in totals.items()
    ]
    rows.sort(key=lambda row: row['bpb'])
    print(json.dumps(rows[:20], indent=2))


if __name__ == '__main__':
    main()
