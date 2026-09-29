"""Development scorer restricted to validation; it cannot load the test corpus."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL, ROOT, device_metrics, make_model, setup, sha
from dev_data import load_development_data
from evaluate import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    device, precision = setup(args.device, args.precision, args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    model, implementation_sha = make_model(
        checkpoint['implementation'], checkpoint['config'], device
    )
    model.load_state_dict(checkpoint['model'])
    validation = load_development_data(('validation',))['validation']
    result = score(model, *validation, device, precision)
    losses = result.pop('window_nll_nats')
    output = args.output or args.checkpoint.parent / f'validation_{device.type}_{precision}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix('.window-nll.npy'), np.asarray(losses))
    result.update(
        protocol=PROTOCOL,
        split='validation',
        development_test_lock=True,
        precision=precision,
        checkpoint_sha256=sha(args.checkpoint),
        implementation_sha256=implementation_sha,
        evaluator_sha256=sha(Path(__file__)),
        score_function_sha256=sha(ROOT / 'evaluate.py'),
        tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'),
        **device_metrics(device),
    )
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
