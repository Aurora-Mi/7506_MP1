"""Validation-only search for train-frequency-binned bigram mixture weights."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from dev_data import load_development_data


ALPHAS = (0.5, 1.0, 2.0, 5.0, 20.0)
LAMBDAS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10)
# Fixed before validation scoring. Bins are outgoing train-bigram row counts:
# [0,4), [4,16), [16,64), [64,256), [256,1024), [1024,+inf).
COUNT_BOUNDARIES = (4, 16, 64, 256, 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Output already exists; use a new path to preserve provenance.')

    started = time.perf_counter()
    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_development_data()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'student':
        parser.error('Search requires a neural student checkpoint before bigram export.')
    model, implementation_hash = make_model('student', checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    vocab = checkpoint['config']['vocab']
    train = data['train'][0]
    pair_counts = torch.bincount(
        train[:-1] * vocab + train[1:], minlength=vocab * vocab
    ).view(vocab, vocab).double()
    row_counts = pair_counts.sum(1)
    unigram_counts = torch.bincount(train[1:], minlength=vocab).double()
    unigram = (unigram_counts + 1.0) / (unigram_counts.sum() + vocab)
    boundaries = torch.tensor(COUNT_BOUNDARIES, dtype=row_counts.dtype)
    bin_count = len(COUNT_BOUNDARIES) + 1
    totals = torch.zeros(len(ALPHAS), bin_count, len(LAMBDAS), dtype=torch.float64)
    targets_per_bin = torch.zeros(bin_count, dtype=torch.long)

    with torch.no_grad():
        for x, y in windows(data['validation'][0]):
            logp = model.predict_log_probs(x)
            valid = y != -100
            targets = y[valid]
            previous = x[valid]
            neural_p = logp[valid].gather(1, targets[:, None]).squeeze(1).double().exp()
            observed = pair_counts[previous, targets]
            denominator_counts = row_counts[previous]
            bin_ids = torch.bucketize(denominator_counts, boundaries)
            targets_per_bin += torch.bincount(bin_ids, minlength=bin_count)
            for alpha_index, alpha in enumerate(ALPHAS):
                bigram_p = ((observed + alpha * unigram[targets]) /
                            (denominator_counts + alpha))
                for bin_index in range(bin_count):
                    in_bin = bin_ids == bin_index
                    if not in_bin.any():
                        continue
                    neural_bin = neural_p[in_bin]
                    bigram_bin = bigram_p[in_bin]
                    for lambda_index, weight in enumerate(LAMBDAS):
                        mixed = (1.0 - weight) * neural_bin + weight * bigram_bin
                        totals[alpha_index, bin_index, lambda_index] -= mixed.log().sum()

    candidates = []
    byte_count = data['validation'][1]
    for alpha_index, alpha in enumerate(ALPHAS):
        best_indices = totals[alpha_index].argmin(1)
        selected_nll = sum(
            totals[alpha_index, bin_index, best_indices[bin_index]].item()
            for bin_index in range(bin_count)
        )
        candidates.append({
            'alpha': alpha,
            'bpb': selected_nll / math.log(2.0) / byte_count,
            'lambdas_by_count_bin': [LAMBDAS[index] for index in best_indices.tolist()],
        })
    candidates.sort(key=lambda row: row['bpb'])
    report = {
        'protocol': PROTOCOL,
        'split': 'validation',
        'statistics_split': 'train',
        'source_checkpoint': str(args.checkpoint.resolve()),
        'source_checkpoint_sha256': sha(args.checkpoint),
        'implementation_sha256': implementation_hash,
        'search_sha256': sha(__file__),
        'count_boundaries': COUNT_BOUNDARIES,
        'count_bin_semantics': '[0,b0), [b0,b1), ..., [blast,+inf)',
        'targets_per_count_bin': targets_per_bin.tolist(),
        'lambda_grid': LAMBDAS,
        'selected': candidates[0],
        'candidates': candidates,
        'train_text_sha256': sha(ROOT / 'data/wikitext_train.txt'),
        'tokenizer_sha256': sha(ROOT / 'data/tokenizer.json'),
        'process_seconds': time.perf_counter() - started,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
