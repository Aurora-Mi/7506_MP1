import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dev_data


class FakeTokenizer:
    @staticmethod
    def from_file(_path):
        return FakeTokenizer()

    def encode(self, text):
        return SimpleNamespace(ids=[len(text)])


class DevelopmentDataTests(unittest.TestCase):
    def make_fixture(self, root):
        data_dir = root / 'data'
        data_dir.mkdir()
        files = {
            'tokenizer.json': b'fake tokenizer',
            'wikitext_train.txt': b'train only',
            'wikitext_validation.txt': b'validation only',
        }
        for name, payload in files.items():
            (data_dir / name).write_bytes(payload)
        manifest = {'sha256': {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in files.items()
        }}
        # A test hash may be listed, but no test file exists. Successful loading
        # therefore proves the development path never tries to open it.
        manifest['sha256']['wikitext_test.txt'] = 'not-read-during-development'
        (data_dir / 'manifest.json').write_text(json.dumps(manifest))

    def test_default_loader_never_opens_test(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            with patch.object(dev_data, 'ROOT', root), patch.object(
                    dev_data, 'Tokenizer', FakeTokenizer):
                data = dev_data.load_development_data()
        self.assertEqual(set(data), {'train', 'validation'})

    def test_test_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'forbids'):
            dev_data.load_development_data(('test',))

    def test_test_key_is_rejected(self):
        data = dev_data.DevelopmentData()
        with self.assertRaisesRegex(RuntimeError, 'locked'):
            _ = data['test']


if __name__ == '__main__':
    unittest.main()
