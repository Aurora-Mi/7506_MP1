"""Validation-only weight interpolation for two aligned student checkpoints."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from common import PROTOCOL, device_metrics, make_model, setup, sha
from dev_data import load_development_data
from evaluate import score


def interpolate_state_dict(state_a, state_b, alpha_b):
    """Return (1-alpha_b)*A + alpha_b*B after strict compatibility checks."""
    if not math.isfinite(alpha_b) or not 0.0 <= alpha_b <= 1.0:
        raise ValueError('Interpolation alpha must be finite and in [0, 1].')
    if state_a.keys() != state_b.keys():
        raise ValueError('Checkpoint parameter keys differ; models are not aligned.')
    mixed = {}
    for name, tensor_a in state_a.items():
        tensor_b = state_b[name]
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            raise ValueError(f'Incompatible checkpoint tensor: {name}')
        if torch.is_floating_point(tensor_a) or torch.is_complex(tensor_a):
            mixed[name] = torch.lerp(tensor_a, tensor_b, alpha_b)
        else:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(f'Non-floating checkpoint tensor differs: {name}')
            mixed[name] = tensor_a.clone()
    return mixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-a', required=True, type=Path)
    parser.add_argument('--checkpoint-b', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.125, 0.25, 0.375, 0.5,
                                 0.625, 0.75, 0.875, 1.0],
                        help='Weight assigned to checkpoint B; checkpoint A gets 1-alpha.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Output directory is not empty; use a new --run-dir.')
    alphas = sorted(set(args.alphas))
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in alphas):
        parser.error('Every alpha must be finite and in [0, 1].')

    started = time.perf_counter()
    device, _ = setup(args.device, 'fp32', args.threads)
    checkpoint_a = torch.load(args.checkpoint_a, map_location='cpu', weights_only=True)
    checkpoint_b = torch.load(args.checkpoint_b, map_location='cpu', weights_only=True)
    for label, checkpoint in (('A', checkpoint_a), ('B', checkpoint_b)):
        if checkpoint.get('protocol') != PROTOCOL:
            parser.error(f'Checkpoint {label} belongs to a different protocol.')
        if checkpoint.get('implementation') != 'student':
            parser.error(f'Checkpoint {label} must use the neural student implementation.')
    if checkpoint_a['config'] != checkpoint_b['config']:
        parser.error('Checkpoint configs differ; refusing weight interpolation.')

    # This path can load validation only; test is inaccessible by construction.
    validation = load_development_data(('validation',))['validation']
    model, implementation_hash = make_model('student', checkpoint_a['config'], device)
    rows = []
    states = {}
    for alpha_b in alphas:
        state = interpolate_state_dict(checkpoint_a['model'], checkpoint_b['model'], alpha_b)
        model.load_state_dict(state, strict=True)
        result = score(model, *validation, device, 'fp32')
        result.pop('window_nll_nats')
        rows.append({'alpha_a': 1.0 - alpha_b, 'alpha_b': alpha_b, **result})
        states[alpha_b] = state
        print(json.dumps({'weight_soup_validation': rows[-1]}), flush=True)
    rows.sort(key=lambda row: row['bpb'])
    winner = rows[0]

    exported = dict(checkpoint_a)
    exported['model'] = states[winner['alpha_b']]
    provenance = {
        'selection_split': 'validation',
        'checkpoint_a': str(args.checkpoint_a.resolve()),
        'checkpoint_a_sha256': sha(args.checkpoint_a),
        'checkpoint_b': str(args.checkpoint_b.resolve()),
        'checkpoint_b_sha256': sha(args.checkpoint_b),
        'alpha_a': winner['alpha_a'],
        'alpha_b': winner['alpha_b'],
    }
    exported['weight_soup'] = provenance
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / 'checkpoint.pt'
    torch.save(exported, output)

    report = {
        'protocol': PROTOCOL,
        'split': 'validation',
        'precision': 'fp32',
        'development_test_lock': True,
        **provenance,
        'selected': winner,
        'candidates': rows,
        'checkpoint_sha256': sha(output),
        'implementation_sha256': implementation_hash,
        'search_sha256': sha(__file__),
        'process_seconds': time.perf_counter() - started,
        **device_metrics(device),
    }
    (args.run_dir / 'weight_soup.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'candidates'}, indent=2))


if __name__ == '__main__':
    main()
