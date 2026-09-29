"""Validation-only search for a train-derived bigram interpolation."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from dev_data import load_development_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path,
                        help='Save all validation candidates and provenance to a new JSON file.')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Output file already exists; use a new --output to preserve prior results.')
    started = time.perf_counter()
    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_development_data()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'student':
        parser.error('Search requires a student checkpoint, before adding a bigram predictor.')
    model, implementation_hash = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    vocab = checkpoint['config']['vocab']
    train = data['train'][0]
    pairs = train[:-1] * vocab + train[1:]
    pair_counts = torch.bincount(pairs, minlength=vocab * vocab).view(vocab, vocab).double()
    row_counts = pair_counts.sum(1)
    unigram_counts = torch.bincount(train[1:], minlength=vocab).double()
    unigram = (unigram_counts + 1.0) / (unigram_counts.sum() + vocab)

    alphas = (1.0, 5.0, 20.0, 100.0, 500.0, 2000.0)
    lambdas = (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
    totals = {(alpha, weight): 0.0 for alpha in alphas for weight in lambdas}

    with torch.no_grad():
        for x, y in windows(data['validation'][0]):
            logp = model.predict_log_probs(x)
            valid = y != -100
            targets = y[valid]
            previous = x[valid]
            neural_p = logp[valid].gather(1, targets[:, None]).squeeze(1).double().exp()
            observed = pair_counts[previous, targets]
            for alpha in alphas:
                bigram_p = (observed + alpha * unigram[targets]) / (row_counts[previous] + alpha)
                for weight in lambdas:
                    mixed = (1.0 - weight) * neural_p + weight * bigram_p
                    totals[alpha, weight] -= mixed.log().sum().item()

    byte_count = data['validation'][1]
    rows = [
        {'alpha': alpha, 'lambda': weight,
         'bpb': nll / math.log(2.0) / byte_count}
        for (alpha, weight), nll in totals.items()
    ]
    rows.sort(key=lambda row: row['bpb'])
    if args.output is not None:
        report = dict(protocol=PROTOCOL, split='validation', precision='fp32',
                      source_checkpoint=str(Path(args.checkpoint).resolve()),
                      source_checkpoint_sha256=sha(args.checkpoint),
                      implementation_sha256=implementation_hash, search_sha256=sha(__file__),
                      statistics_split='train', training_pairs=len(train)-1,
                      train_text_sha256=sha(ROOT/'data/wikitext_train.txt'),
                      tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                      selected=rows[0], candidates=rows, threads=args.threads,
                      process_seconds=time.perf_counter()-started,
                      note='Search only; export and confirm with the unchanged CPU FP32 evaluator.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(rows[:20], indent=2))


if __name__ == '__main__':
    main()
