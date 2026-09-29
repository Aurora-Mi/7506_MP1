"""Development-only data loader that cannot access the test corpus.

Training, validation, calibration, and search scripts must use this module.
The unchanged course evaluator remains the only code path that can load test.
"""
import json

import torch
from tokenizers import Tokenizer

from common import ROOT, sha


DEVELOPMENT_SPLITS = frozenset({'train', 'validation'})


class DevelopmentData(dict):
    """Mapping that gives an explicit error if development code asks for test."""

    def __getitem__(self, key):
        if key == 'test':
            raise RuntimeError(
                'Test data is locked during development. Use train/validation only, '
                'then freeze the final predictor before official evaluation.'
            )
        return super().__getitem__(key)


def load_development_data(splits=('train', 'validation')):
    """Load only explicitly allowed development splits; never open the test file."""
    requested = tuple(splits)
    forbidden = set(requested) - DEVELOPMENT_SPLITS
    if forbidden:
        raise ValueError(f'Development loader forbids split(s): {sorted(forbidden)}')
    if not requested:
        raise ValueError('At least one development split is required.')

    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    required_files = ['tokenizer.json', *(f'wikitext_{split}.txt' for split in requested)]
    for name in required_files:
        expected = manifest['sha256'][name]
        if sha(ROOT / 'data' / name) != expected:
            raise ValueError(f'Changed benchmark file: {name}')

    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    data = DevelopmentData()
    for split in requested:
        raw = (ROOT / 'data' / f'wikitext_{split}.txt').read_bytes()
        ids = tokenizer.encode(raw.decode('utf-8')).ids
        data[split] = (torch.tensor(ids, dtype=torch.long), len(raw))
    return data
