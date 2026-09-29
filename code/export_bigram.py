"""Export frozen neural weights plus bigram statistics fitted only to training tokens."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch

from common import PROTOCOL, ROOT, make_model, setup, sha
from dev_data import load_development_data


def fit_bigram_log_probs(tokens, vocab, alpha):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    if len(tokens) < 2:
        raise ValueError('At least two training tokens are required.')
    pairs = tokens[:-1] * vocab + tokens[1:]
    counts = torch.bincount(pairs, minlength=vocab * vocab).view(vocab, vocab).double()
    row_counts = counts.sum(1)
    unigram_counts = torch.bincount(tokens[1:], minlength=vocab).double()
    unigram = (unigram_counts + 1.0) / (unigram_counts.sum() + vocab)
    probabilities = (counts + alpha * unigram[None, :]) / (row_counts[:, None] + alpha)
    return probabilities.log().float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=20.0)
    parser.add_argument('--lambda', type=float, dest='weight', default=0.02)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error('alpha must be positive and finite.')
    if not math.isfinite(args.weight) or not 0 <= args.weight < 1:
        parser.error('lambda must be finite and in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Output directory is not empty; use a new --run-dir.')
    started = time.perf_counter()
    device, _ = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'student':
        parser.error('Input must be a student checkpoint from the course protocol.')
    config = dict(checkpoint['config']) | {'bigram_alpha': args.alpha, 'bigram_lambda': args.weight}
    model, implementation_sha = make_model('student_bigram', config, device)
    loaded = model.load_state_dict(checkpoint['model'], strict=False)
    if loaded.missing_keys != ['bigram_log_probs'] or loaded.unexpected_keys:
        raise ValueError('Input neural weights do not match the bigram model.')
    data = load_development_data(('train',))
    train = data['train'][0]
    model.bigram_log_probs.copy_(fit_bigram_log_probs(train, config['vocab'], args.alpha))
    if not torch.isfinite(model.bigram_log_probs).all():
        raise ValueError('Bigram log probabilities must be finite.')
    if model.bigram_log_probs.logsumexp(-1).abs().max().item() > 1e-6:
        raise ValueError('Bigram rows are not normalized.')
    # No optimizer step: verify every original tensor is bit-for-bit preserved.
    for name, original in checkpoint['model'].items():
        if not torch.equal(original, model.state_dict()[name]):
            raise ValueError(f'Unexpected modification to neural weights: {name}')
    provenance = dict(source_checkpoint_sha256=sha(args.checkpoint),
                      statistics_split='train', selection_split='validation',
                      train_text_sha256=sha(ROOT / 'data/wikitext_train.txt'),
                      tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'),
                      alpha=args.alpha, weight=args.weight, training_pairs=len(train) - 1)
    exported = dict(checkpoint)
    exported.update(implementation='student_bigram', config=config, model=model.state_dict(),
                    bigram=provenance)
    # Store exactly one predictor and its required implementation sources.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / 'checkpoint.pt'
    torch.save(exported, output)
    for name in ('student_bigram.py', 'student.py'):
        shutil.copyfile(ROOT / name, args.run_dir / name)
    asset_bytes = sum(path.stat().st_size for path in
                      (output, args.run_dir / 'student_bigram.py', args.run_dir / 'student.py'))
    if asset_bytes > 64 * 1024 ** 2:
        raise ValueError('Export exceeds the 64 MiB uncompressed inference-asset limit.')
    report = dict(protocol=PROTOCOL, source_checkpoint=str(args.checkpoint.resolve()),
                  **provenance, checkpoint_sha256=sha(output),
                  implementation_sha256=implementation_sha,
                  base_implementation_sha256=sha(ROOT / 'student.py'),
                  exporter_sha256=sha(__file__), neural_weights_unchanged=True,
                  inference_asset_bytes=asset_bytes, inference_asset_mib=asset_bytes / 1024 ** 2,
                  bigram_table_bytes=model.bigram_log_probs.numel() * model.bigram_log_probs.element_size(),
                  process_seconds=time.perf_counter() - started,
                  note='No new neural training. Validate with the unchanged CPU FP32 evaluator.')
    (args.run_dir / 'bigram_export.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
