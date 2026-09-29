"""Validation-only FP32 calibration; reuse model features across the parameter grid."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, device_metrics, make_model, setup, sha, windows
from dev_data import load_development_data
from evaluate import score


def cache_target_probability(features, ids, targets, theta):
    """Targets gather scoring probabilities, never construct cache keys/outcomes."""
    normalized = F.normalize(features.float(), dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    length = ids.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril(-1)
    scores = (theta * similarity).masked_fill(~causal, float('-inf'))
    scores[:, 0, 0] = 0.0
    outcomes = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
    matches = outcomes[:, None, :] == targets[:, :, None]
    return (scores.softmax(-1) * matches).sum(-1)


def mix_target_logp(base_target_logp, cache_target_p, weight):
    if weight == 0.0:
        return base_target_logp
    mixed = torch.logaddexp(
        base_target_logp + math.log1p(-weight),
        cache_target_p.clamp_min(1e-30).log() + math.log(weight),
    )
    mixed[:, 0] = base_target_logp[:, 0]
    return mixed


@torch.no_grad()
def search_grid(model, tokens, byte_count, device, candidates, batch_size=32):
    model.eval()
    totals = {candidate: 0.0 for candidate in candidates}
    temperatures = sorted({candidate[0] for candidate in candidates})
    thetas = sorted({candidate[2] for candidate in candidates})
    for batch_index, (x, y) in enumerate(windows(tokens, batch_size)):
        x, y = x.to(device), y.to(device)
        features = model.features(x)
        logits = model.head(features).float()
        targets, valid = y.clamp_min(0), y != -100
        base = {temperature: F.log_softmax(logits / temperature, dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1) for temperature in temperatures}
        cache = {theta: cache_target_probability(features, x, targets, theta)
                 for theta in thetas}
        for temperature, weight, theta in candidates:
            logp = mix_target_logp(base[temperature], cache[theta], weight)
            totals[temperature, weight, theta] -= logp[valid].double().sum().item()
        if (batch_index + 1) % 10 == 0:
            print(json.dumps({'validation_batches_scored': batch_index + 1}), flush=True)
    rows = [dict(logit_temperature=t, cache_lambda=w, cache_theta=h,
                 bpb=nll / math.log(2) / byte_count)
            for (t, w, h), nll in totals.items()]
    return sorted(rows, key=lambda row: row['bpb'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1.04, 1.08, 1.12])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.04, 0.07, 0.10])
    parser.add_argument('--thetas', type=float, nargs='+', default=[5.0, 10.0, 20.0])
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Output directory is not empty. Use a new --run-dir; never overwrite results.')
    if args.batch_size < 1 or any(not math.isfinite(v) or v <= 0 for v in args.temperatures + args.thetas):
        parser.error('Batch size, temperatures and thetas must be positive and finite.')
    if any(not math.isfinite(v) or not 0 <= v < 1 for v in args.lambdas):
        parser.error('Cache lambdas must be finite and in [0, 1).')
    started = time.perf_counter()
    device, _ = setup(args.device, 'fp32', args.threads)
    source_hash = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'student':
        parser.error('This search requires a student checkpoint from the fixed course protocol.')
    model, implementation_hash = make_model('student', checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'], strict=True)
    original = (model.logit_temperature, model.cache_lambda, model.cache_theta)
    candidates = sorted(set(itertools.product(args.temperatures, args.lambdas, args.thetas)) | {original})
    data = load_development_data(('validation',))
    rows = search_grid(model, *data['validation'], device, candidates, args.batch_size)
    winner = rows[0]
    model.logit_temperature = winner['logit_temperature']
    model.cache_lambda = winner['cache_lambda']
    model.cache_theta = winner['cache_theta']
    # Verify the fast target-only calculation using the unchanged official scorer.
    verified = score(model, *data['validation'], device, 'fp32', args.batch_size)
    verified.pop('window_nll_nats')
    if abs(verified['bpb'] - winner['bpb']) > 1e-6:
        raise RuntimeError('Search disagrees with the official scorer; refusing export.')
    provenance = dict(source_checkpoint_sha256=source_hash, split='validation',
                      settings={key: winner[key] for key in
                                ('logit_temperature', 'cache_lambda', 'cache_theta')})
    exported = dict(checkpoint)
    exported['config'] = dict(checkpoint['config']) | provenance['settings']
    exported['calibration'] = provenance
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / 'checkpoint.pt'
    # Preserve all weights and training provenance: this is not additional training.
    torch.save(exported, output)
    report = dict(protocol=PROTOCOL, split='validation', precision='fp32',
                  source_checkpoint=str(args.checkpoint.resolve()),
                  source_checkpoint_sha256=source_hash, checkpoint_sha256=sha(output),
                  implementation_sha256=implementation_hash, search_sha256=sha(__file__),
                  original_settings=dict(zip(('logit_temperature', 'cache_lambda', 'cache_theta'), original)),
                  original_grid_bpb=next(row['bpb'] for row in rows if
                      (row['logit_temperature'], row['cache_lambda'], row['cache_theta']) == original),
                  selected=winner, verified_validation=verified, candidates=rows,
                  process_seconds=time.perf_counter() - started, **device_metrics(device))
    (args.run_dir / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'candidates'}, indent=2))


if __name__ == '__main__':
    main()
